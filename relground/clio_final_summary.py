"""Build and validate the lightweight final Clio benchmark summary."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "0.3"
STAGE = "clio-final-benchmark-summary"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source(root: Path, path: Path) -> dict[str, Any]:
    try:
        reference = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("Clio final-summary source escapes project root") from error
    return {
        "path": reference,
        "sha256": _sha256(path),
        "retention": "local_ignored_full_report",
    }


def _scene(
    *,
    role: str,
    geometry_frames: int,
    grounding: Mapping[str, Any],
    association: Mapping[str, Any],
    relation: Mapping[str, Any],
) -> dict[str, Any]:
    q0 = grounding["metrics"]["q0_top1"]
    q1 = grounding["metrics"]["q1_top5_a2"]
    q1f = grounding["metrics"]["q1f_top5_a2_with_q0_fallback"]
    assoc_a1 = association["metrics"]["A1"]
    assoc_a2 = association["metrics"]["A2"]
    rel = relation["metrics"]
    association_counts = association["counts"]
    semantic_embedding_count = int(
        association_counts.get("observations_with_semantic_embedding", 0)
    )
    mixed_class_task_count = int(
        association_counts.get("tasks_with_multiple_class_texts", 0)
    )
    return {
        "role": role,
        "geometry_frames": geometry_frames,
        "task_count": int(q0["task_count"]),
        "object_grounding": {
            "metric": "predicted center inside any official oriented GT OBB; nearest GT is diagnostic only",
            "official_clio_metric_claim": False,
            "primary_policy": "Q1F frozen Top-5+A2 with deterministic Q0 fallback",
            "q0_top1": q0,
            "q1_top5_a2_diagnostic": q1,
            "q1f_top5_a2_with_q0_fallback": q1f,
            "delta_q1_diagnostic_minus_q0": grounding["metrics"]["delta_q1_minus_q0"],
            "delta_q1f_minus_q0": grounding["metrics"]["delta_q1f_minus_q0"],
        },
        "association": {
            "label_rule": "same padded target GT OBB; background/background pairs excluded as unknown",
            "frozen_tasks": association["counts"]["frozen_tasks"],
            "runtime_name": "task-internal geometry+quality complete-link association",
            "semantic_input_audit": {
                "observations_total": int(association_counts.get("observations_total", 0)),
                "observations_with_semantic_embedding": semantic_embedding_count,
                "tasks_with_multiple_class_texts": mixed_class_task_count,
                "multimodal_semantic_association_claim": False,
            },
            "associable_tasks": association["counts"]["associable_tasks"],
            "unknown_background_pairs_excluded": association["counts"]["unknown_background_pairs_excluded"],
            "A1": assoc_a1,
            "A2": assoc_a2,
            "delta_A2_minus_A1": {
                key: float(assoc_a2[key]) - float(assoc_a1[key])
                for key in ("precision", "recall", "f1", "accuracy")
            },
        },
        "relation_and_abstention": {
            "query_count": rel["query_count"],
            "positive_count": rel["positive_count"],
            "negative_count": rel["negative_count"],
            "positive_pair_grounding_acc_at_1_strict": rel["positive_pair_grounding_acc_at_1_strict"],
            "positive_pair_grounding_acc_at_1_alignment_rmse_padded": rel["positive_pair_grounding_acc_at_1_alignment_rmse_padded"],
            "negative_rejection_accuracy": rel["negative_rejection_accuracy"],
            "reason_matched_negative_rejection_accuracy": rel["reason_matched_negative_rejection_accuracy"],
            "relation_aware_negative_rejection_accuracy": rel["relation_aware_negative_rejection_accuracy"],
            "end_to_end_task_accuracy_alignment_rmse_padded": rel["end_to_end_task_accuracy_alignment_rmse_padded"],
            "pair_grounded_task_accuracy_alignment_rmse_padded": rel["pair_grounded_task_accuracy_alignment_rmse_padded"],
            "answer_coverage": rel["answer_coverage"],
            "answer_aurc_discrete": rel["answer_aurc_discrete"],
            "answerability_proxy_brier": rel["answerability_proxy_brier"],
            "answerability_proxy_ece_10": rel["answerability_proxy_ece_10"],
            "calibration_status": relation["contract"]["calibration_status"],
        },
    }


def build_summary(
    *,
    project_root: Path,
    apartment_grounding_path: Path,
    apartment_association_path: Path,
    apartment_relation_path: Path,
    cubicle_grounding_path: Path,
    cubicle_association_path: Path,
    cubicle_relation_path: Path,
    protocol_path: Path,
    created_at: str | None = None,
) -> dict[str, Any]:
    root = project_root.resolve()
    paths = {
        "apartment_grounding": apartment_grounding_path,
        "apartment_association": apartment_association_path,
        "apartment_relation": apartment_relation_path,
        "cubicle_grounding": cubicle_grounding_path,
        "cubicle_association": cubicle_association_path,
        "cubicle_relation": cubicle_relation_path,
        "relation_protocol": protocol_path,
    }
    payloads = {
        key: json.loads(path.read_text(encoding="utf-8"))
        for key, path in paths.items()
        if key != "relation_protocol"
    }
    if any(value.get("status") != "PASS" for value in payloads.values()):
        raise ValueError("all source benchmark reports must pass")
    apartment = _scene(
        role="development",
        geometry_frames=192,
        grounding=payloads["apartment_grounding"],
        association=payloads["apartment_association"],
        relation=payloads["apartment_relation"],
    )
    cubicle = _scene(
        role="fixed-confirmatory",
        geometry_frames=172,
        grounding=payloads["cubicle_grounding"],
        association=payloads["cubicle_association"],
        relation=payloads["cubicle_relation"],
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "stage": STAGE,
        "project_positioning": "VGGT-SLAM geometry plus an auditable semantic-localization reliability layer",
        "scenes": {"apartment": apartment, "cubicle": cubicle},
        "headline_result": {
            "scope": "Cubicle 18-task fixed-confirmatory frozen-Q1F object-grounding benchmark",
            "policy_id": "Q1F",
            "q0_strict_acc_at_1": cubicle["object_grounding"]["q0_top1"]["grounding_acc_at_1"],
            "q1f_strict_acc_at_1": cubicle["object_grounding"]["q1f_top5_a2_with_q0_fallback"]["grounding_acc_at_1"],
            "strict_delta_percentage_points": 100.0 * cubicle["object_grounding"]["delta_q1f_minus_q0"]["grounding_acc_at_1"],
            "alignment_rmse_padded_delta_percentage_points": 100.0 * cubicle["object_grounding"]["delta_q1f_minus_q0"]["grounding_acc_at_1_with_alignment_rmse_margin"],
            "attribution": "Frozen Q1F system: Top-K multiframe retrieval/lifting, available A2 object memory, and deterministic Q0 fallback; not evidence that A2 association alone improves",
        },
        "claim_boundary": {
            "cubicle_object_grounding_policy_frozen_before_data_inspection": True,
            "cubicle_q1f_policy_manifest_validated": True,
            "association_and_relation_reports_are_fixed_confirmatory_not_untouched_held_out": True,
            "a2_pairwise_improvement_on_cubicle": False,
            "real_calibration_complete": False,
            "official_clio_metric_reproduced": False,
            "clio_a2_multimodal_semantic_association": False,
            "found_it_comparison_available": False,
            "found_it_comparison_scope": "OUT_OF_SCOPE_BY_PROJECT_DEFINITION",
            "closed_loop_navigation": False,
            "sota_or_superiority_claim": False,
        },
        "sources": {key: _source(root, path) for key, path in paths.items()},
        "public_evidence_policy": {
            "summary_contains": "aggregate JSON metrics, sample sizes, hashes, and claim boundaries",
            "summary_omits": "Clio RGB/depth/bag, task YAML, masks, point clouds, videos, and pair-level rows",
            "full_report_replay": "requires locally acquired public Clio scenes",
        },
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }


def validate_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    try:
        if payload.get("schema_version") != SCHEMA_VERSION or payload.get("status") != "PASS":
            raise ValueError("unsupported or failing Clio summary")
        if payload.get("claim_boundary", {}).get("found_it_comparison_scope") != "OUT_OF_SCOPE_BY_PROJECT_DEFINITION":
            raise ValueError("FOUND-IT must remain explicitly outside project scope")
        apartment = payload["scenes"]["apartment"]
        cubicle = payload["scenes"]["cubicle"]
        if apartment["role"] != "development" or cubicle["role"] != "fixed-confirmatory":
            raise ValueError("scene roles changed")
        for scene in (apartment, cubicle):
            if scene["object_grounding"]["primary_policy"] != "Q1F frozen Top-5+A2 with deterministic Q0 fallback":
                raise ValueError("final summary primary policy is not frozen Q1F")
            q0 = scene["object_grounding"]["q0_top1"]
            q1f = scene["object_grounding"]["q1f_top5_a2_with_q0_fallback"]
            delta = scene["object_grounding"]["delta_q1f_minus_q0"]
            for key in (
                "coverage", "grounding_acc_at_1",
                "grounding_acc_at_1_with_alignment_rmse_margin",
            ):
                if abs(float(delta[key]) - (float(q1f[key]) - float(q0[key]))) > 1e-11:
                    raise ValueError(f"grounding delta mismatch: {key}")
            a1 = scene["association"]["A1"]
            a2 = scene["association"]["A2"]
            assoc_delta = scene["association"]["delta_A2_minus_A1"]
            semantic_audit = scene["association"]["semantic_input_audit"]
            if (
                scene["association"]["runtime_name"]
                != "task-internal geometry+quality complete-link association"
                or int(semantic_audit["observations_with_semantic_embedding"]) != 0
                or int(semantic_audit["tasks_with_multiple_class_texts"]) != 0
                or semantic_audit["multimodal_semantic_association_claim"] is not False
            ):
                raise ValueError(
                    "Clio A2 runtime must remain geometry+quality complete-link "
                    "without a multimodal semantic claim"
                )
            for key in ("precision", "recall", "f1", "accuracy"):
                if abs(float(assoc_delta[key]) - (float(a2[key]) - float(a1[key]))) > 1e-11:
                    raise ValueError(f"association delta mismatch: {key}")
        headline = payload["headline_result"]
        if headline.get("policy_id") != "Q1F":
            raise ValueError("headline policy is not Q1F")
        if abs(
            float(headline["strict_delta_percentage_points"])
            - 100.0 * float(cubicle["object_grounding"]["delta_q1f_minus_q0"]["grounding_acc_at_1"])
        ) > 1e-9:
            raise ValueError("headline strict delta does not match Cubicle metrics")
        if payload["claim_boundary"].get("cubicle_q1f_policy_manifest_validated") is not True:
            raise ValueError("Cubicle frozen Q1F manifest validation was removed")
        if payload["claim_boundary"].get("clio_a2_multimodal_semantic_association") is not False:
            raise ValueError("Clio A2 cannot be described as multimodal semantic association")

        if payload["claim_boundary"]["a2_pairwise_improvement_on_cubicle"] is not False:
            raise ValueError("A2 Cubicle association regression was hidden")
        serialized = json.dumps(payload["sources"], sort_keys=True)
        if '"path": "/' in serialized or ".." in serialized:
            raise ValueError("summary contains unsafe source paths")
    except (KeyError, TypeError, ValueError) as error:
        failures.append(str(error))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not failures else "FAIL",
        "stage": f"{STAGE}-validation",
        "checks": {
            "scene_roles_explicit": not failures,
            "grounding_deltas_recomputed": not failures,
            "association_deltas_recomputed": not failures,
            "headline_matches_cubicle": not failures,
            "negative_result_not_hidden": not failures,
            "a2_runtime_scope_honest": not failures,
            "source_paths_safe": not failures,
        },
        "failures": failures,
    }
