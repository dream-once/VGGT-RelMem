"""Validate real and synthetic D14 fixed-Top-K replay evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from relground.candidate_cache import CandidateOutcomeCache
from relground.d9_association import ManualInstanceLabels
from relground.observation_cache import sha256_file
from relground.q1_fixed_topk import (
    validate_evaluation_payload,
    validate_prediction_payload,
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
        raise ValueError("D14 reference escapes the Week 2 evidence boundary")
    if not resolved.is_file():
        raise ValueError(f"missing referenced artifact: {reference}")
    return resolved


def _validate_one(bundle: Path, prefix: str) -> dict[str, Any]:
    prediction_path = bundle / f"{prefix}_prediction.json"
    evaluation_path = bundle / f"{prefix}_evaluation.json"
    prediction = read_json(prediction_path)
    cache_path = resolve_reference(
        bundle,
        str(prediction["source"]["candidate_cache"]),
    )
    if sha256_file(cache_path) != prediction["source"]["candidate_cache_sha256"]:
        raise ValueError(f"{prefix} candidate cache hash changed")
    cache = CandidateOutcomeCache.load(cache_path).to_dict()
    validate_prediction_payload(prediction, cache)

    evaluation = read_json(evaluation_path)
    referenced_prediction = resolve_reference(
        bundle,
        str(evaluation["source"]["prediction"]),
    )
    if referenced_prediction != prediction_path.resolve():
        raise ValueError(f"{prefix} evaluation points to another prediction")
    if sha256_file(prediction_path) != evaluation["source"]["prediction_sha256"]:
        raise ValueError(f"{prefix} prediction hash changed")
    labels_path = resolve_reference(
        bundle,
        str(evaluation["source"]["labels"]),
    )
    if sha256_file(labels_path) != evaluation["source"]["labels_sha256"]:
        raise ValueError(f"{prefix} labels hash changed")
    labels = ManualInstanceLabels.load(labels_path)
    validate_evaluation_payload(evaluation, prediction, cache, labels)
    if prediction["status"] != "PASS" or evaluation["status"] != "PASS":
        raise ValueError(f"{prefix} replay is not complete")
    return {
        "materialization_status": cache["materialization_status"],
        "selected_frames": [
            row["frame_id"]
            for row in prediction["curves"][-1]["selected_frames"]
        ],
        "requested_budget": prediction["curves"][-1]["requested_budget"],
        "selected_count": prediction["curves"][-1]["selected_count"],
        "exhaustion_reason": prediction["curves"][-1]["exhaustion_reason"],
        "k1_matches_q0": prediction["acceptance"]["k1_matches_q0"],
        "metrics": evaluation["budget_details"],
    }


def validate_output(output_dir: Path) -> dict[str, Any]:
    bundle = output_dir.resolve()
    failures: list[str] = []
    results: dict[str, Any] = {}
    try:
        results["real"] = _validate_one(bundle, "real")
        if results["real"]["materialization_status"] != "partial":
            raise ValueError("real replay must document the partial D11 cache")
        if results["real"]["selected_count"] != 4:
            raise ValueError("real K=5 replay must retain four nonredundant frames")
        if results["real"]["exhaustion_reason"] != "nonredundant_candidates_exhausted":
            raise ValueError("real K=5 exhaustion reason changed")
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        failures.append(f"real: {error}")
    try:
        results["synthetic"] = _validate_one(bundle, "synthetic")
        if results["synthetic"]["materialization_status"] != "complete":
            raise ValueError("synthetic replay must use a complete cache")
        if results["synthetic"]["selected_count"] != 5:
            raise ValueError("synthetic K=5 replay must select five frames")
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
        failures.append(f"D14 evidence exceeds 128 KiB: {size_bytes}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "stage": "D14",
        "completion": "CPU_COMPLETE",
        "gpu_acceptance": "GPU_ACCEPTANCE_PENDING",
        "development_replay": True,
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
