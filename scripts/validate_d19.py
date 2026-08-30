"""Validate retained D19 ablation and failure-audit evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relground.ablation_protocol import (
    FAILURE_CATEGORIES,
    evaluate_synthetic_ablations,
    validate_ablation_manifest,
    validate_prediction_replay,
)
from relground.d9_association import ManualInstanceLabels
from relground.experiment_protocol import (
    load_json,
    sha256_file,
    source_by_id,
    validate_experiment_manifest,
)


def run(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.project_root).resolve()
    manifest = load_json(root / args.manifest)
    validate_ablation_manifest(manifest, project_root=root)
    d18 = load_json(root / manifest["d18_manifest"])
    validate_experiment_manifest(d18, project_root=root)
    checks: dict[str, bool] = {}

    for source_id, evidence in (
        ("office-loop-development", root / args.office_evidence),
        ("synthetic-correctness", root / args.synthetic_evidence),
        ("clio-apartment-development", root / args.clio_evidence),
    ):
        source = source_by_id(d18, source_id)
        cache = load_json(root / source["cache_ref"])
        prediction = load_json(evidence / "prediction.json")
        validate_prediction_replay(prediction, manifest, cache)
        checks[f"{source_id}_deterministic"] = True
        checks[f"{source_id}_one_cache"] = (
            prediction["source"]["candidate_cache_sha256"]
            == source["cache_sha256"]
        )
        if source["labels_ref"] is not None:
            evaluation = load_json(evidence / "evaluation.json")
            labels = ManualInstanceLabels.load(root / source["labels_ref"])
            expected = evaluate_synthetic_ablations(
                prediction,
                cache,
                labels,
                prediction_ref="prediction.json",
                prediction_sha256=sha256_file(
                    evidence / "prediction.json"
                ),
                labels_ref=source["labels_ref"],
                labels_sha256=source["labels_sha256"],
                created_at=evaluation["created_at"],
            )
            checks["synthetic_evaluation_deterministic"] = (
                evaluation == expected
            )
            audit = evaluation["failure_audit"]
            checks["failure_denominator_complete"] = (
                audit["denominator_query_count"]
                == len(audit["frozen_query_ids"])
                and audit["failed_query_count"]
                == sum(audit["category_counts"].values())
                + audit["uncategorized_failed_queries"]
            )
            checks["failure_taxonomy_complete"] = (
                tuple(audit["category_counts"]) == FAILURE_CATEGORIES
            )

    synthetic = load_json(
        root / args.synthetic_evidence / "prediction.json"
    )
    checks["historical_success_honest"] = (
        synthetic["historical_success_ablation"]["status"]
        == "NOT_IMPLEMENTED"
    )
    checks["no_performance_claim"] = (
        synthetic["performance_claim"] is None
    )
    clio = load_json(root / args.clio_evidence / "prediction.json")
    checks["clio_development_structure_only"] = (
        clio["source_id"] == "clio-apartment-development"
        and clio["performance_claim"] is None
        and clio["failure_audit_readiness"]["status"]
        == "UNLABELLED_ENGINEERING_ONLY"
        and clio["failure_audit_readiness"]["category_counts"] is None
    )
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "schema_version": "0.1",
        "status": status,
        "stage": "D19-validation",
        "checks": checks,
        "source_status": "CPU_COMPLETE",
        "real_ablation_status": "REAL_ABLATION_PENDING",
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
        "--manifest", default="configs/d19_ablation_manifest.json"
    )
    parser.add_argument(
        "--office-evidence",
        default="evidence/week3/d19-ablations/office-loop",
    )
    parser.add_argument(
        "--synthetic-evidence",
        default="evidence/week3/d19-ablations/synthetic",
    )
    parser.add_argument(
        "--clio-evidence",
        default="evidence/week4/clio-apartment-gpu/d19-ablations",
    )
    parser.add_argument("--output")
    return parser


if __name__ == "__main__":
    raise SystemExit(
        0 if run(build_parser().parse_args())["status"] == "PASS" else 2
    )
