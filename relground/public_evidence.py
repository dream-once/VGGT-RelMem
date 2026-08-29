"""Portable public evidence for GPU acceptance and D15/D15.5."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import copy
import json
import shutil

from .candidate_cache import CandidateOutcomeCache
from .observation_cache import sha256_file
from .q2_sequential import (
    Q2_LEGACY_OBSERVATION_FIELD,
    Q2_METHOD_NAME,
    Q2_OBSERVATION_METRIC,
    Q2_OBSERVATION_SEMANTICS,
    Q2_POLICY_ID,
    validate_trace_payload,
)


PUBLIC_EVIDENCE_SCHEMA_VERSION = "0.1"
PUBLIC_BUNDLE_STAGE = "D15-public-gpu-evidence"


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _inside(root: Path, path: str | Path, name: str) -> Path:
    resolved = Path(path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{name} escapes project root")
    return resolved


def _repo_ref(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _artifact(path: Path, reference: str) -> dict[str, Any]:
    return {
        "path": reference,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _absolute_strings(value: Any, path: str = "$") -> list[str]:
    failures: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            failures.extend(_absolute_strings(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            failures.extend(_absolute_strings(item, f"{path}[{index}]"))
    elif isinstance(value, str) and Path(value).is_absolute():
        failures.append(path)
    return failures


def _resolve_week3(bundle: Path, reference: str) -> Path:
    path = Path(reference)
    if path.is_absolute():
        raise ValueError("public evidence reference must be relative")
    boundary = bundle.resolve().parent
    resolved = (bundle / path).resolve()
    if resolved != boundary and boundary not in resolved.parents:
        raise ValueError("public evidence reference escapes Week 3")
    if not resolved.is_file():
        raise ValueError(f"missing public evidence artifact: {reference}")
    return resolved


def export_public_bundle(
    *,
    project_root: str | Path,
    gpu_bundle: str | Path,
    visualization_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    gpu = _inside(root, gpu_bundle, "GPU bundle")
    visual_dir = _inside(root, visualization_dir, "visualization directory")
    output = _inside(root, output_dir, "public output")
    output.mkdir(parents=True, exist_ok=True)

    source_report_path = gpu / "gpu_acceptance_report.json"
    source_report = load_json(source_report_path)
    if source_report["status"] != "PASS":
        raise ValueError("source GPU acceptance report is not PASS")
    checks = copy.deepcopy(source_report["checks"])
    d15_check = checks["D15_trace"]
    d15_check["new_observation_count"] = d15_check.pop("observed_gain")
    d15_check["legacy_source_field"] = Q2_LEGACY_OBSERVATION_FIELD
    d15_check["coverage_aware"] = False
    public_report = {
        "schema_version": PUBLIC_EVIDENCE_SCHEMA_VERSION,
        "status": "PASS",
        "stage": "GPU-acceptance-public-report",
        "gpu_acceptance": source_report["gpu_acceptance"],
        "evaluation_scope": source_report["evaluation_scope"],
        "source": {
            "local_report": {
                "path": _repo_ref(root, source_report_path),
                "sha256": sha256_file(source_report_path),
                "retained_in_git": False,
            },
            "bundle": _repo_ref(root, gpu),
            "retained_partial_cache": (
                "evidence/week2/d11-candidate-cache/"
                "candidate_cache.json"
            ),
        },
        "checks": checks,
    }
    public_report_path = output / "gpu_acceptance_report.json"
    write_json(public_report_path, public_report)

    cache_source = gpu / "d11-complete-cache/candidate_cache.json"
    cache = load_json(cache_source)
    CandidateOutcomeCache.from_dict(cache)
    cache_path = output / "candidate_cache.json"
    write_json(cache_path, cache)

    trace_source = gpu / "d15-complete-replay/real_trace.json"
    trace = load_json(trace_source)
    trace["source"]["candidate_cache"] = "candidate_cache.json"
    trace["source"]["candidate_cache_sha256"] = sha256_file(cache_path)
    validate_trace_payload(trace, cache)
    trace_path = output / "d15_complete_trace.json"
    write_json(trace_path, trace)

    from scripts.validate_d15_5_visualization import validate_output

    d15_5_validation = validate_output(visual_dir)
    if d15_5_validation["status"] != "PASS":
        raise ValueError("D15.5 local artifacts did not validate")
    d15_5_path = output / "d15_5_validation.json"
    write_json(d15_5_path, d15_5_validation)

    preview_source = visual_dir / "overview.png"
    preview_path = output / "overview.png"
    shutil.copy2(preview_source, preview_path)

    visual_manifest = (
        root
        / "evidence/week3/d20-reproduction/sources/"
        "visualization_manifest.json"
    )
    viewpoint_audit = (
        root
        / "evidence/week3/d20-reproduction/sources/"
        "viewpoint_audit.json"
    )
    result_tables = (
        root / "evidence/week3/d20-reproduction/result_tables.json"
    )
    result_card = (
        root / "evidence/week3/d21-final/result_card.json"
    )
    visual = load_json(visual_manifest)
    omitted = [
        {
            "kind": item["kind"],
            "path": item["path"],
            "sha256": item["sha256"],
            "bytes": item["bytes"],
            "retained_in_git": False,
            "distribution": "optional_GitHub_Release",
        }
        for item in visual["artifacts"]
        if item["kind"] in {
            "dynamic_object_parallax_video",
            "colored_scene_point_cloud",
        }
    ]
    manifest = {
        "schema_version": PUBLIC_EVIDENCE_SCHEMA_VERSION,
        "status": "PASS",
        "stage": PUBLIC_BUNDLE_STAGE,
        "scope": (
            "engineering_replay_and_visualization_evidence_"
            "not_held_out_performance"
        ),
        "q2_semantics": {
            "legacy_policy_id": Q2_POLICY_ID,
            "method_name": Q2_METHOD_NAME,
            "legacy_serialized_field": Q2_LEGACY_OBSERVATION_FIELD,
            "canonical_metric": Q2_OBSERVATION_METRIC,
            "metric_semantics": Q2_OBSERVATION_SEMANTICS,
            "coverage_aware": False,
            "performance_claim": None,
        },
        "artifacts": {
            "gpu_acceptance_report": _artifact(
                public_report_path, "gpu_acceptance_report.json"
            ),
            "candidate_cache": _artifact(
                cache_path, "candidate_cache.json"
            ),
            "d15_complete_trace": _artifact(
                trace_path, "d15_complete_trace.json"
            ),
            "d15_5_validation": _artifact(
                d15_5_path, "d15_5_validation.json"
            ),
            "d15_5_manifest": _artifact(
                visual_manifest,
                "../d20-reproduction/sources/"
                "visualization_manifest.json",
            ),
            "d15_5_viewpoint_audit": _artifact(
                viewpoint_audit,
                "../d20-reproduction/sources/viewpoint_audit.json",
            ),
            "key_result_tables": _artifact(
                result_tables,
                "../d20-reproduction/result_tables.json",
            ),
            "final_result_card": _artifact(
                result_card,
                "../d21-final/result_card.json",
            ),
            "overview_thumbnail": _artifact(
                preview_path, "overview.png"
            ),
        },
        "omitted_large_artifacts": omitted,
    }
    manifest_path = output / "artifact_manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def validate_public_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    bundle = Path(bundle_dir).resolve()
    failures: list[str] = []
    checks: dict[str, bool] = {}
    try:
        manifest = load_json(bundle / "artifact_manifest.json")
        artifacts = manifest["artifacts"]
        for item in artifacts.values():
            path = _resolve_week3(bundle, str(item["path"]))
            if sha256_file(path) != item["sha256"]:
                raise ValueError(f"artifact hash mismatch: {item['path']}")
            if path.stat().st_size != item["bytes"]:
                raise ValueError(f"artifact size mismatch: {item['path']}")
        checks["artifact_hashes"] = True

        cache = load_json(bundle / "candidate_cache.json")
        CandidateOutcomeCache.from_dict(cache)
        trace = load_json(bundle / "d15_complete_trace.json")
        if trace["source"]["candidate_cache"] != "candidate_cache.json":
            raise ValueError("D15 trace cache reference is not portable")
        validate_trace_payload(trace, cache)
        checks["d15_trace_replays"] = True

        report = load_json(bundle / "gpu_acceptance_report.json")
        if _absolute_strings(report):
            raise ValueError("public GPU report contains absolute paths")
        checks["gpu_report_portable"] = report["status"] == "PASS"

        d15_5 = load_json(bundle / "d15_5_validation.json")
        checks["d15_5_validator_pass"] = (
            d15_5["status"] == "PASS"
            and d15_5["artifact_status"] == "PASS"
        )
        semantics = manifest["q2_semantics"]
        checks["q2_boundary_honest"] = (
            semantics["canonical_metric"] == Q2_OBSERVATION_METRIC
            and semantics["coverage_aware"] is False
            and semantics["performance_claim"] is None
        )
        checks["thumbnail_is_bounded"] = (
            artifacts["overview_thumbnail"]["bytes"] <= 640 * 1024
        )
        checks["large_binaries_are_hash_only"] = all(
            not item["retained_in_git"]
            for item in manifest["omitted_large_artifacts"]
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        failures.append(str(error))
    status = (
        "PASS"
        if not failures and checks and all(checks.values())
        else "FAIL"
    )
    return {
        "schema_version": PUBLIC_EVIDENCE_SCHEMA_VERSION,
        "status": status,
        "stage": "D15-public-evidence-validation",
        "checks": checks,
        "failures": failures,
    }
