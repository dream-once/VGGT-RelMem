"""D21 final result card and external-claim boundary audit."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
import json
import re


D21_SCHEMA_VERSION = "0.1"
D21_STATUS = "CPU_COMPLETE"
PROJECT_POSITIONING = "VGGT-SLAM 几何之上的可审计语义定位可靠性层"
FINAL_CONCLUSION = (
    "完成可复现基准、查询策略与关联策略隔离、"
    "可靠拒答协议和失败分析"
)
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
    "D17-relation-reliability",
    "D18-QxA-protocol",
    "D19-ablation-audit",
    "D20-reproduction-package",
)
REQUIRED_GAPS = {
    "clio_download": "DATA_DOWNLOAD_BLOCKED_SIZE_UNKNOWN",
    "data_license": "DATA_LICENSE_UNVERIFIED",
    "clio_held_out": "CLIO_HELD_OUT_PENDING",
    "real_calibration": "REAL_DATA_CALIBRATION_PENDING",
    "real_ablation": "REAL_ABLATION_PENDING",
    "new_gpu_inference": "PENDING",
    "optional_binary_release": "OPTIONAL_BINARY_RELEASE_PENDING",
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
        raise ValueError("D21 status must remain CPU_COMPLETE")
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
        "d16_config",
        "d16_report",
        "d17_config",
        "d18_config",
        "d19_config",
        "d20_config",
        "d20_results",
        "d20_report",
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
    d20 = load_json(inputs["d20_results"][0])
    d20_report = load_json(inputs["d20_report"][0])
    d18_config = load_json(inputs["d18_config"][0])
    d19_config = load_json(inputs["d19_config"][0])
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
            "description": "fail-closed Clio data and disk feasibility audit",
            "evidence": _source(
                inputs["d16_report"][0],
                inputs["d16_report"][1],
                root,
            ),
            "config": _source(
                inputs["d16_config"][0],
                inputs["d16_config"][1],
                root,
            ),
            "sample_size": {"scene_roles": len(d16["scenes"])},
            "budget": {
                "available_bytes": d16["available_bytes"],
                "safety_reserve_bytes": d16["reserve_bytes"],
                "maximum_peak_bytes": d16["maximum_peak_bytes"],
            },
            "validation_status": "PASS",
            "scope": "metadata_readiness_no_download",
        },
        {
            "result_id": "D17-relation-reliability",
            "description": "label-separated relation and abstention protocol",
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
            "scope": "synthetic_correctness_real_calibration_pending",
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
                "matrix_rows": len(d20["qxa"]["rows"]),
            },
            "budget": d18_config["budgets"],
            "validation_status": "PASS",
            "scope": "synthetic_correctness_office_development_replay",
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
            },
            "budget": {"association_input_q1_k": 5},
            "validation_status": "PASS",
            "scope": "synthetic_correctness_real_ablation_pending",
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
            "performance_improvement": None,
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
