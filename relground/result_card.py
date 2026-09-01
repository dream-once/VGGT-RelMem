"""D21 final result card and external-claim boundary audit."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
import json
import re


D21_SCHEMA_VERSION = "0.2"
D21_STATUS = "GPU_AND_CPU_COMPLETE_WITH_EXTERNAL_PACKAGING_GAPS"
PROJECT_POSITIONING = "VGGT-SLAM 几何之上的可审计语义定位可靠性层"
FINAL_CONCLUSION = (
    "完成可复现基准、查询策略与关联策略隔离、"
    "显式拒答协议、跨场景固定确认和失败分析"
)
LIMITED_PERFORMANCE_CLAIM = {
    "scope": "Cubicle 18-task fixed-confirmatory frozen-Q1F object-grounding benchmark",
    "policy_id": "Q1F",
    "metric": "strict predicted-center-in-oriented-GT-OBB Acc@1",
    "q0": 0.277777777778,
    "q1f": 0.388888888889,
    "delta_percentage_points": 11.1111111111,
    "a2_pairwise_attribution": False,
}
CLAIM_TERMS = (
    "官方",
    "复现",
    "改进",
    "导航",
    "优于",
    "SOTA",
    "state-of-the-art",
    "性能提升",
)
MANIFEST_FIELDS = (
    "schema_version",
    "stage",
    "status",
    "repository_base_commit",
    "project_positioning",
    "final_conclusion",
    "inputs",
    "claim_terms",
    "required_gaps",
    "result_ids",
    "frozen_at",
)
INPUT_FIELDS = ("input_id", "path", "sha256")
RESULT_IDS = (
    "D15.5-scene-visualization",
    "D16-clio-readiness",
    "Clio-apartment-GPU-acceptance",
    "D17-relation-reliability",
    "D18-QxA-protocol",
    "D19-ablation-audit",
    "D20-reproduction-package",
    "Clio-final-object-grounding-association",
    "Clio-final-relation-abstention",
)
REQUIRED_GAPS = {
    "clio_download": "APARTMENT_AND_CUBICLE_LOCAL_COMPLETE_RAW_NOT_REDISTRIBUTED",
    "data_license": "DATA_LICENSE_UNVERIFIED",
    "clio_held_out": "CUBICLE_Q1F_FROZEN_ASSOCIATION_RELATION_FIXED_CONFIRMATORY",
    "real_calibration": "REAL_DATA_CALIBRATION_PENDING",
    "query_policy_ablation": "LABELLED_QUERY_POLICY_ABLATION_PENDING",
    "instance_metrics": "INSTANCE_RECALL_DUPLICATE_RATE_COUNT_ERROR_PENDING",
    "statistical_intervals": "STATISTICAL_CONFIDENCE_INTERVALS_PENDING",
    "final_costs": "FULL_36_TASK_LATENCY_AND_PEAK_VRAM_PENDING",
    "new_gpu_inference": "CLIO_APARTMENT_AND_CUBICLE_COMPLETE",
    "found_it_comparison": "OUT_OF_SCOPE_BY_PROJECT_DEFINITION",
    "optional_binary_release": "OPTIONAL_BINARY_RELEASE_PENDING",
    "demo_recording": "DEMO_RECORDING_PENDING",
    "release_tag": "RELEASE_TAG_PENDING_UNCOMMITTED_FIXES",
}


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _relative(value: Any, name: str) -> str:
    path = Path(str(value))
    if not str(value) or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be repository-relative")
    return path.as_posix()


def validate_result_card_manifest(
    payload: Mapping[str, Any],
    *,
    project_root: str | Path | None = None,
) -> None:
    if set(payload) != set(MANIFEST_FIELDS):
        raise ValueError("D21 manifest fields are not frozen")
    if payload["schema_version"] != D21_SCHEMA_VERSION:
        raise ValueError("unsupported D21 schema")
    if payload["stage"] != "D21-final-result-card":
        raise ValueError("D21 stage changed")
    if payload["status"] != D21_STATUS:
        raise ValueError("D21 source status changed")
    if re.fullmatch(
        r"[0-9a-f]{40}", str(payload["repository_base_commit"])
    ) is None:
        raise ValueError("D21 base commit must be a Git hash")
    if payload["project_positioning"] != PROJECT_POSITIONING:
        raise ValueError("D21 project positioning changed")
    if payload["final_conclusion"] != FINAL_CONCLUSION:
        raise ValueError("D21 final conclusion changed")
    inputs = payload["inputs"]
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("D21 inputs must be a non-empty list")
    identifiers: list[str] = []
    for item in inputs:
        if not isinstance(item, Mapping) or set(item) != set(INPUT_FIELDS):
            raise ValueError("D21 input fields are not frozen")
        identifiers.append(str(item["input_id"]))
        reference = _relative(item["path"], "D21 input path")
        digest = str(item["sha256"])
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("D21 input hash must be SHA-256")
        if project_root is not None:
            root = Path(project_root).resolve()
            if sha256_file(root / reference) != digest:
                raise ValueError(f"D21 input hash mismatch: {reference}")
    if identifiers != [
        "readme",
        "presentation_material",
        "d16_config",
        "d16_report",
        "d16_acquisition",
        "clio_gpu_report",
        "clio_gpu_validation",
        "d17_config",
        "d18_config",
        "d19_config",
        "d20_config",
        "d20_results",
        "d20_report",
        "clio_final_summary",
        "clio_final_validation",
        "clio_relation_protocol",
    ]:
        raise ValueError("D21 input identity or order changed")
    if tuple(payload["claim_terms"]) != CLAIM_TERMS:
        raise ValueError("D21 claim vocabulary changed")
    if payload["required_gaps"] != REQUIRED_GAPS:
        raise ValueError("D21 required gaps changed")
    if tuple(payload["result_ids"]) != RESULT_IDS:
        raise ValueError("D21 result inventory changed")


def _inputs(
    root: str | Path, manifest: Mapping[str, Any]
) -> dict[str, tuple[Path, str]]:
    project = Path(root).resolve()
    return {
        str(item["input_id"]): (
            project / str(item["path"]),
            str(item["sha256"]),
        )
        for item in manifest["inputs"]
    }


def _source(path: Path, digest: str, root: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": digest,
    }


def build_result_card(
    project_root: str | Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    validate_result_card_manifest(manifest, project_root=project_root)
    root = Path(project_root).resolve()
    inputs = _inputs(root, manifest)
    d16 = load_json(inputs["d16_report"][0])
    d16_acquisition = load_json(inputs["d16_acquisition"][0])
    clio_gpu = load_json(inputs["clio_gpu_report"][0])
    clio_gpu_validation = load_json(inputs["clio_gpu_validation"][0])
    d20 = load_json(inputs["d20_results"][0])
    d20_report = load_json(inputs["d20_report"][0])
    d18_config = load_json(inputs["d18_config"][0])
    d19_config = load_json(inputs["d19_config"][0])
    clio_final = load_json(inputs["clio_final_summary"][0])
    clio_final_validation = load_json(
        inputs["clio_final_validation"][0]
    )
    relation_rows = {
        row["metric"]: row["value"]
        for row in d20["relations"]["rows"]
    }
    d15 = d20["d15_5"]
    results = [
        {
            "result_id": "D15.5-scene-visualization",
            "description": "long-trajectory scene-memory visualization audit",
            "evidence": _source(
                inputs["d20_results"][0],
                inputs["d20_results"][1],
                root,
            ),
            "config": {
                "path": (
                    "evidence/week3/d20-reproduction/sources/"
                    "visualization_manifest.json"
                ),
                "sha256": d20["sources"]["d15_5_manifest"]["sha256"],
            },
            "sample_size": {
                "anchor_cameras": d15["anchor_cameras"],
                "selected_cameras": d15["selected_cameras"],
                "observations": d15["observations"],
                "predicted_objects": d15["predicted_objects"],
            },
            "budget": {"retrieval_top_k": 24},
            "validation_status": "PASS",
            "scope": "office_loop_engineering_visualization",
        },
        {
            "result_id": "D16-clio-readiness",
            "description": (
                "Clio feasibility plus local apartment RGB/task-metadata "
                "development subset acceptance"
            ),
            "evidence": _source(
                inputs["d16_acquisition"][0],
                inputs["d16_acquisition"][1],
                root,
            ),
            "config": _source(
                inputs["d16_config"][0],
                inputs["d16_config"][1],
                root,
            ),
            "sample_size": {
                "scene_roles": len(d16["scenes"]),
                "selected_rgb_frames": d16_acquisition[
                    "materialization_scope"
                ]["rgb_frame_count"],
                "task_metadata_files": d16_acquisition[
                    "materialization_scope"
                ]["task_metadata_file_count"],
            },
            "budget": {
                "available_bytes": d16["available_bytes"],
                "safety_reserve_bytes": d16["reserve_bytes"],
                "maximum_peak_bytes": d16["maximum_peak_bytes"],
            },
            "validation_status": "PASS",
            "scope": (
                "apartment_development_subset_only_"
                "full_modalities_and_cubicle_pending"
            ),
        },
        {
            "result_id": "Clio-apartment-GPU-acceptance",
            "description": (
                "real VGGT plus PE plus SAM3 development replay with "
                "query-specific multiview audit"
            ),
            "evidence": _source(
                inputs["clio_gpu_report"][0],
                inputs["clio_gpu_report"][1],
                root,
            ),
            "config": _source(
                inputs["d16_acquisition"][0],
                inputs["d16_acquisition"][1],
                root,
            ),
            "sample_size": {
                "geometry_frames": clio_gpu["geometry"]["frame_count"],
                "candidate_outcomes": clio_gpu["perception"][
                    "available_candidates"
                ],
                "lifted_observations": clio_gpu["perception"][
                    "lifted_instances"
                ],
                "evidence_frames": len(
                    clio_gpu["perception"][
                        "frames_with_lifted_observations"
                    ]
                ),
            },
            "budget": {"query": "pillow", "sam_calls": 24},
            "validation_status": clio_gpu_validation["status"],
            "scope": "real_gpu_development_replay_not_performance",
        },
        {
            "result_id": "D17-relation-reliability",
            "description": "label-separated selective-answer and abstention protocol",
            "evidence": _source(
                inputs["d20_results"][0],
                inputs["d20_results"][1],
                root,
            ),
            "config": _source(
                inputs["d17_config"][0],
                inputs["d17_config"][1],
                root,
            ),
            "sample_size": {
                "queries": relation_rows["query_count"],
                "positive": relation_rows["positive_count"],
                "negative": relation_rows["negative_count"],
            },
            "budget": {"engineering_threshold": 0.60},
            "validation_status": "PASS",
            "scope": "synthetic_selective_answer_correctness_real_calibration_pending",
        },
        {
            "result_id": "D18-QxA-protocol",
            "description": "frozen query-policy by association-policy matrix",
            "evidence": _source(
                inputs["d20_results"][0],
                inputs["d20_results"][1],
                root,
            ),
            "config": _source(
                inputs["d18_config"][0],
                inputs["d18_config"][1],
                root,
            ),
            "sample_size": {
                "frozen_queries": 1,
                "office_complete_cache_matrix_rows": len(
                    d20["qxa_development"]["rows"]
                ),
                "clio_development_matrix_rows": len(
                    d20["qxa_clio_development"]["rows"]
                ),
                "synthetic_matrix_rows": len(d20["qxa"]["rows"]),
            },
            "budget": d18_config["budgets"],
            "validation_status": "PASS",
            "scope": (
                "office_and_clio_apartment_complete_cache_development_"
                "replay_plus_synthetic_correctness_not_performance"
            ),
        },
        {
            "result_id": "D19-ablation-audit",
            "description": "one-factor ablations and six-class failure audit",
            "evidence": _source(
                inputs["d20_results"][0],
                inputs["d20_results"][1],
                root,
            ),
            "config": _source(
                inputs["d19_config"][0],
                inputs["d19_config"][1],
                root,
            ),
            "sample_size": {
                "frozen_queries": d20["failure_audit"][
                    "denominator_query_count"
                ],
                "q2_variants": len(d19_config["q2_variants"]),
                "a2_variants": len(d19_config["a2_variants"]),
                "clio_q2_variants": len(
                    d20["clio_ablations"]["q2_rows"]
                ),
                "clio_a2_variants": len(
                    d20["clio_ablations"]["a2_rows"]
                ),
            },
            "budget": {"association_input_q1_k": 5},
            "validation_status": "PASS",
            "scope": (
                "synthetic_correctness_plus_clio_unlabelled_engineering_"
                "ablation_real_metrics_pending"
            ),
        },
        {
            "result_id": "D20-reproduction-package",
            "description": "JSON-derived tables and moved-directory replay",
            "evidence": _source(
                inputs["d20_report"][0],
                inputs["d20_report"][1],
                root,
            ),
            "config": _source(
                inputs["d20_config"][0],
                inputs["d20_config"][1],
                root,
            ),
            "sample_size": {
                "canonical_inputs": len(d20["sources"]),
                "derived_tables": 3,
                "stage_validators": len(d20_report["validators"]),
            },
            "budget": {"cpu_only": True},
            "validation_status": d20_report["status"],
            "scope": "tracked_json_markdown_reproduction",
        },
        {
            "result_id": "Clio-final-object-grounding-association",
            "description": (
                "Apartment development and Cubicle fixed-confirmatory "
                "object grounding plus A1/A2 association"
            ),
            "evidence": _source(
                inputs["clio_final_summary"][0],
                inputs["clio_final_summary"][1],
                root,
            ),
            "config": _source(
                inputs["clio_relation_protocol"][0],
                inputs["clio_relation_protocol"][1],
                root,
            ),
            "sample_size": {
                "apartment_tasks": clio_final["scenes"]["apartment"]["task_count"],
                "cubicle_tasks": clio_final["scenes"]["cubicle"]["task_count"],
                "apartment_geometry_frames": clio_final["scenes"]["apartment"]["geometry_frames"],
                "cubicle_geometry_frames": clio_final["scenes"]["cubicle"]["geometry_frames"],
            },
            "budget": {"Q0_sam_calls": 1, "Q1F_sam_calls": 5},
            "validation_status": clio_final_validation["status"],
            "scope": "metric_specific_frozen_cubicle_object_grounding_not_general_superiority",
        },
        {
            "result_id": "Clio-final-relation-abstention",
            "description": (
                "real directional grounding and negative-query abstention"
            ),
            "evidence": _source(
                inputs["clio_final_summary"][0],
                inputs["clio_final_summary"][1],
                root,
            ),
            "config": _source(
                inputs["clio_relation_protocol"][0],
                inputs["clio_relation_protocol"][1],
                root,
            ),
            "sample_size": {
                "apartment_queries": clio_final["scenes"]["apartment"]["relation_and_abstention"]["query_count"],
                "cubicle_queries": clio_final["scenes"]["cubicle"]["relation_and_abstention"]["query_count"],
                "cubicle_positive": clio_final["scenes"]["cubicle"]["relation_and_abstention"]["positive_count"],
                "cubicle_negative": clio_final["scenes"]["cubicle"]["relation_and_abstention"]["negative_count"],
            },
            "budget": {"uncalibrated_answer_threshold": 0.60},
            "validation_status": clio_final_validation["status"],
            "scope": "fixed_confirmatory_target_reference_pair_relations_calibration_pending",
        },
    ]
    if tuple(item["result_id"] for item in results) != RESULT_IDS:
        raise AssertionError("D21 result inventory changed")
    return {
        "schema_version": D21_SCHEMA_VERSION,
        "status": "PASS",
        "stage": "D21-final-result-card",
        "project_positioning": PROJECT_POSITIONING,
        "final_conclusion": FINAL_CONCLUSION,
        "results": results,
        "gaps": REQUIRED_GAPS,
        "claim_boundary": {
            "official_found_it_reproduction": False,
            "closed_loop_navigation": False,
            "held_out_performance": False,
            "sota_or_superiority_claim": False,
            "performance_improvement": LIMITED_PERFORMANCE_CLAIM,
        },
        "source_status": D21_STATUS,
    }


def _qualified(line: str, term: str) -> tuple[bool, str]:
    boundary_markers = (
        "不声称",
        "不称",
        "不支持",
        "不能",
        "不得",
        "尚未",
        "未接入",
        "没有",
        "null",
        "PENDING",
        "缺口",
        "参照",
        "不是",
        "审计",
    )
    if any(marker in line for marker in boundary_markers):
        return True, "explicitly_qualified_or_negated"
    if term == "官方" and any(
        marker in line
        for marker in (
            "官方示例",
            "官方发布页",
            "官方几何管线",
            "官方 VGGT",
            "官方大小",
            "固定 commit",
        )
    ):
        return True, "upstream_attribution_or_source_fact"
    if term == "复现" and any(
        marker in line
        for marker in (
            "复现命令",
            "可复现",
            "验收",
            "证据",
            "已完成",
            "validator",
            "重建",
            "VGGT",
        )
    ):
        return True, "reproducibility_or_validated_result_context"
    if term == "导航" and any(
        marker in line
        for marker in ("感知前端", "可靠性层", "定位")
    ):
        return True, "perception_or_localization_scope"
    if term == "改进":
        return True, "descriptive_non_superiority_context"
    return False, "unqualified_high_risk_claim"


def audit_readme_claims(readme_text: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    lines = readme_text.splitlines()
    for line_number, line in enumerate(lines, start=1):
        for term in CLAIM_TERMS:
            if term.lower() not in line.lower():
                continue
            previous = lines[line_number - 2] if line_number > 1 else ""
            context = (
                f"{previous} {line}"
                if previous.strip()
                else line
            )
            accepted, reason = _qualified(context, term)
            records.append({
                "line": line_number,
                "term": term,
                "text": line.strip(),
                "disposition": (
                    "ACCEPTED_WITH_CONTEXT"
                    if accepted
                    else "REVIEW_REQUIRED"
                ),
                "reason": reason,
            })
    review_count = sum(
        item["disposition"] == "REVIEW_REQUIRED" for item in records
    )
    return {
        "schema_version": D21_SCHEMA_VERSION,
        "status": "PASS" if review_count == 0 else "FAIL",
        "stage": "D21-claim-audit",
        "terms": list(CLAIM_TERMS),
        "occurrence_count": len(records),
        "review_required_count": review_count,
        "records": records,
        "required_positioning_present": (
            PROJECT_POSITIONING in readme_text
        ),
        "required_conclusion_present": (
            FINAL_CONCLUSION in readme_text
        ),
    }


def render_result_card(card: Mapping[str, Any]) -> str:
    lines = [
        "# Final result card",
        "",
        f"**Status:** {card['source_status']}",
        "",
        f"**Positioning:** {card['project_positioning']}",
        "",
        f"**Conclusion:** {card['final_conclusion']}",
        "",
        "| result | sample size | budget | validation | scope |",
        "|---|---|---|---|---|",
    ]
    for row in card["results"]:
        sample = json.dumps(
            row["sample_size"], ensure_ascii=False, sort_keys=True
        )
        budget = json.dumps(
            row["budget"], ensure_ascii=False, sort_keys=True
        )
        lines.append(
            f"| {row['result_id']} | {sample} | {budget} | "
            f"{row['validation_status']} | {row['scope']} |"
        )
    lines.extend(["", "## Explicit gaps", ""])
    lines.extend(
        f"- {key}: {value}" for key, value in card["gaps"].items()
    )
    lines.extend([
        "",
        "No Clio held-out or superiority number is claimed.",
        "",
    ])
    return "\n".join(lines)
