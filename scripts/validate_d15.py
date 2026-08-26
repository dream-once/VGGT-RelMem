"""Validate D15 real readiness, synthetic trace, and engineering comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from relground.candidate_cache import CandidateOutcomeCache
from relground.observation_cache import sha256_file
from relground.q2_sequential import (
    validate_comparison_payload,
    validate_trace_payload,
)
from scripts.run_fixed_topk_replay import write_json


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} root must be an object")
    return payload


def resolve_reference(bundle: Path, reference: str) -> Path:
    boundary = bundle.resolve().parent
    resolved = (bundle / reference).resolve()
    if resolved != boundary and boundary not in resolved.parents:
        raise ValueError("D15 reference escapes the Week 2 evidence boundary")
    if not resolved.is_file():
        raise ValueError(f"missing referenced artifact: {reference}")
    return resolved


def _trace_with_cache(
    bundle: Path,
    prefix: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    trace_path = bundle / f"{prefix}_trace.json"
    trace = read_json(trace_path)
    cache_path = resolve_reference(
        bundle,
        str(trace["source"]["candidate_cache"]),
    )
    if sha256_file(cache_path) != trace["source"]["candidate_cache_sha256"]:
        raise ValueError(f"{prefix} candidate cache hash changed")
    cache = CandidateOutcomeCache.load(cache_path).to_dict()
    validate_trace_payload(trace, cache)
    serialized = json.dumps(trace, sort_keys=True).lower()
    for forbidden in (
        "ground_truth",
        "instance_id",
        "pair_labels",
        '"metrics"',
        '"f1"',
        "frustum",
    ):
        if forbidden in serialized:
            raise ValueError(f"{prefix} trace leaks forbidden field {forbidden}")
    return trace, cache, trace_path


def validate_output(output_dir: Path) -> dict[str, Any]:
    bundle = output_dir.resolve()
    failures: list[str] = []
    results: dict[str, Any] = {}
    try:
        real, real_cache, _ = _trace_with_cache(bundle, "real")
        if real_cache["materialization_status"] != "partial":
            raise ValueError("real trace must use the retained partial cache")
        if real["status"] != "BLOCKED_MISSING_OUTCOME":
            raise ValueError("real partial trace must remain readiness/blocked evidence")
        if real["summary"]["performance_claim"] is not None:
            raise ValueError("real partial trace cannot publish a performance claim")
        results["real"] = {
            "status": real["status"],
            "selected_frames": real["summary"]["selected_frames"],
            "stop_reason": real["summary"]["stop_reason"],
            "budget1_matches_q0": real["summary"]["budget1_matches_q0"],
            "performance_claim": real["summary"]["performance_claim"],
        }
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        failures.append(f"real: {error}")
    try:
        synthetic, synthetic_cache, synthetic_path = _trace_with_cache(
            bundle,
            "synthetic",
        )
        if synthetic_cache["materialization_status"] != "complete":
            raise ValueError("synthetic trace must use a complete cache")
        if synthetic["status"] != "PASS":
            raise ValueError("synthetic trace must complete")
        comparison_path = bundle / "synthetic_comparison.json"
        comparison = read_json(comparison_path)
        q1_path = resolve_reference(
            bundle,
            str(comparison["source"]["q1_prediction"]),
        )
        if sha256_file(q1_path) != comparison["source"]["q1_prediction_sha256"]:
            raise ValueError("synthetic Q1 prediction hash changed")
        q2_path = resolve_reference(
            bundle,
            str(comparison["source"]["q2_trace"]),
        )
        if q2_path != synthetic_path.resolve():
            raise ValueError("comparison points to another Q2 trace")
        if sha256_file(q2_path) != comparison["source"]["q2_trace_sha256"]:
            raise ValueError("synthetic Q2 trace hash changed")
        validate_comparison_payload(
            comparison,
            read_json(q1_path),
            synthetic,
        )
        results["synthetic"] = {
            "status": synthetic["status"],
            "selected_frames": synthetic["summary"]["selected_frames"],
            "stop_reason": synthetic["summary"]["stop_reason"],
            "budget1_matches_q0": synthetic["summary"]["budget1_matches_q0"],
            "comparison_scope": comparison["comparison_scope"],
        }
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        failures.append(f"synthetic: {error}")
    files = [path for path in bundle.rglob("*") if path.is_file()]
    invalid_extensions = [
        path.name for path in files if path.suffix.lower() not in {".json", ".md"}
    ]
    size_bytes = sum(path.stat().st_size for path in files)
    if invalid_extensions:
        failures.append("non-lightweight evidence files: " + ", ".join(invalid_extensions))
    if size_bytes > 128 * 1024:
        failures.append(f"D15 evidence exceeds 128 KiB: {size_bytes}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "stage": "D15",
        "completion": "CPU_COMPLETE",
        "gpu_acceptance": "GPU_ACCEPTANCE_PENDING",
        "real_evidence_scope": "READINESS_BLOCKED_TRACE_ONLY",
        "results": results,
        "evidence_bytes": size_bytes,
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    parser.add_argument("--report")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = validate_output(Path(args.output_dir))
    if args.report:
        write_json(Path(args.report), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
