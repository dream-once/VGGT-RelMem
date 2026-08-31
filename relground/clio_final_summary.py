"""Build and validate the lightweight final Clio benchmark summary."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "0.1"
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
    assoc_a1 = association["metrics"]["A1"]
    assoc_a2 = association["metrics"]["A2"]
    rel = relation["metrics"]
    return {
        "role": role,
        "geometry_frames": geometry_frames,
        "task_count": int(q0["task_count"]),
        "object_grounding": {
            "metric": "predicted center inside official oriented GT OBB",
            "official_clio_metric_claim": False,
            "q0_top1": q0,
            "q1_top5_a2": q1,
            "delta_q1_minus_q0": grounding["metrics"]["delta_q1_minus_q0"],
        },
        "association": {
            "label_rule": "same padded target GT OBB; background/background pairs excluded as unknown",
            "frozen_tasks": association["counts"]["frozen_tasks"],
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
            "positive_grounding_acc_at_1_strict": rel["positive_grounding_acc_at_1_strict"],
            "positive_grounding_acc_at_1_alignment_rmse_padded": rel["positive_grounding_acc_at_1_alignment_rmse_padded"],
            "negative_rejection_accuracy": rel["negative_rejection_accuracy"],
            "relation_aware_negative_rejection_accuracy": rel["relation_aware_negative_rejection_accuracy"],
            "task_accuracy_alignment_rmse_padded": rel["task_accuracy_alignment_rmse_padded"],
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
            "scope": "Cubicle 18-task frozen object-grounding benchmark",
            "q0_strict_acc_at_1": cubicle["object_grounding"]["q0_top1"]["grounding_acc_at_1"],
            "q1_strict_acc_at_1": cubicle["object_grounding"]["q1_top5_a2"]["grounding_acc_at_1"],
            "strict_delta_percentage_points": 100.0 * cubicle["object_grounding"]["delta_q1_minus_q0"]["grounding_acc_at_1"],
            "alignment_rmse_padded_delta_percentage_points": 100.0 * cubicle["object_grounding"]["delta_q1_minus_q0"]["grounding_acc_at_1_with_alignment_rmse_margin"],
            "attribution": "Top-K multiframe retrieval/lifting plus available A2 object memory; not evidence that A2 pairwise association alone improves",
        },
        "claim_boundary": {
            "cubicle_object_grounding_policy_frozen_before_data_inspection": True,
            "association_and_relation_reports_are_fixed_confirmatory_not_untouched_held_out": True,
            "a2_pairwise_improvement_on_cubicle": False,
            "real_calibration_complete": False,
            "official_clio_metric_reproduced": False,
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
            q0 = scene["object_grounding"]["q0_top1"]
            q1 = scene["object_grounding"]["q1_top5_a2"]
            delta = scene["object_grounding"]["delta_q1_minus_q0"]
            for key in (
                "coverage", "grounding_acc_at_1",
                "grounding_acc_at_1_with_alignment_rmse_margin",
            ):
                if abs(float(delta[key]) - (float(q1[key]) - float(q0[key]))) > 1e-11:
                    raise ValueError(f"grounding delta mismatch: {key}")
            a1 = scene["association"]["A1"]
            a2 = scene["association"]["A2"]
            assoc_delta = scene["association"]["delta_A2_minus_A1"]
            for key in ("precision", "recall", "f1", "accuracy"):
                if abs(float(assoc_delta[key]) - (float(a2[key]) - float(a1[key]))) > 1e-11:
                    raise ValueError(f"association delta mismatch: {key}")
        headline = payload["headline_result"]
        if abs(
            float(headline["strict_delta_percentage_points"])
            - 100.0 * float(cubicle["object_grounding"]["delta_q1_minus_q0"]["grounding_acc_at_1"])
        ) > 1e-9:
            raise ValueError("headline strict delta does not match Cubicle metrics")
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
            "source_paths_safe": not failures,
        },
        "failures": failures,
    }
