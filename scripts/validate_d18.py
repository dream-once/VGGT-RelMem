"""Validate frozen D18 manifest and retained CPU replay evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relground.d9_association import ManualInstanceLabels
from relground.experiment_protocol import (
    deterministic_prediction_replay,
    evaluate_synthetic_prediction,
    load_json,
    sha256_file,
    source_by_id,
    validate_experiment_manifest,
    validate_prediction_payload,
)


def run(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.project_root).resolve()
    manifest = load_json(root / args.manifest)
    validate_experiment_manifest(manifest, project_root=root)
    checks: dict[str, bool] = {}

    source_dirs = (
        ("office-loop-development", root / args.office_evidence),
        ("synthetic-correctness", root / args.synthetic_evidence),
    )
    for source_id, directory in source_dirs:
        source = source_by_id(manifest, source_id)
        cache = load_json(root / source["cache_ref"])
        prediction = load_json(directory / "prediction.json")
        validate_prediction_payload(prediction, manifest, cache)
        checks[f"{source_id}_deterministic"] = (
            prediction
            == deterministic_prediction_replay(
                prediction, manifest, cache
            )
        )
        checks[f"{source_id}_same_cache"] = all(
            row["candidate_count"] == len(cache["candidates"])
            for row in prediction["matrix_rows"]
        )
        if source["labels_ref"] is not None:
            evaluation = load_json(directory / "evaluation.json")
            labels = ManualInstanceLabels.load(root / source["labels_ref"])
            expected = evaluate_synthetic_prediction(
                prediction,
                cache,
                labels,
                prediction_ref="prediction.json",
                prediction_sha256=sha256_file(
                    directory / "prediction.json"
                ),
                labels_ref=source["labels_ref"],
                labels_sha256=source["labels_sha256"],
                created_at=evaluation["created_at"],
            )
            checks["synthetic_evaluation_deterministic"] = (
                evaluation == expected
            )
            checks["synthetic_not_performance"] = (
                evaluation["performance_claim"] is None
            )

    office = load_json(root / args.office_evidence / "prediction.json")
    checks["office_development_only"] = (
        office["result_scope"]
        == "development_engineering_replay_not_performance"
    )
    checks["held_out_pending"] = (
        office["held_out"]["status"] == "CLIO_HELD_OUT_PENDING"
        and office["held_out"]["metric_values"] is None
    )
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "schema_version": "0.1",
        "status": status,
        "stage": "D18-validation",
        "checks": checks,
        "source_status": "CPU_COMPLETE",
        "held_out_status": "CLIO_HELD_OUT_PENDING",
    }
    if args.output:
        Path(args.output).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument(
        "--manifest", default="configs/d18_experiment_manifest.json"
    )
    parser.add_argument(
        "--office-evidence",
        default="evidence/week3/d18-qxa/office-loop",
    )
    parser.add_argument(
        "--synthetic-evidence",
        default="evidence/week3/d18-qxa/synthetic",
    )
    parser.add_argument("--output")
    return parser


if __name__ == "__main__":
    raise SystemExit(
        0 if run(build_parser().parse_args())["status"] == "PASS" else 2
    )
