"""Independently recompute a self-contained D17 relation bundle."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
from typing import Any
import json

from relground.association import ObjectMemory
from relground.relation_protocol import (
    evaluate_relation_prediction,
    run_relation_prediction,
)


REQUIRED = (
    "object_memory.json", "anchor_poses.json", "queries.json",
    "calibration_manifest.json", "prediction.json", "labels.json",
    "evaluation.json", "README.md",
)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def validate(bundle: str | Path) -> dict[str, Any]:
    root = Path(bundle)
    errors = [f"missing {name}" for name in REQUIRED if not (root / name).is_file()]
    if errors:
        return {"status": "FAIL", "stage": "D17", "errors": errors}
    try:
        paths = {name: root / name for name in REQUIRED}
        anchors = json.loads(paths["anchor_poses.json"].read_text())
        queries = json.loads(paths["queries.json"].read_text())
        calibration = json.loads(paths["calibration_manifest.json"].read_text())
        prediction = json.loads(paths["prediction.json"].read_text())
        labels = json.loads(paths["labels.json"].read_text())
        evaluation = json.loads(paths["evaluation.json"].read_text())
        prediction_source = {
            "object_memory": "object_memory.json",
            "object_memory_sha256": _sha(paths["object_memory.json"]),
            "anchor_poses": "anchor_poses.json",
            "anchor_poses_sha256": _sha(paths["anchor_poses.json"]),
            "queries": "queries.json",
            "queries_sha256": _sha(paths["queries.json"]),
            "calibration": "calibration_manifest.json",
            "calibration_sha256": _sha(paths["calibration_manifest.json"]),
        }
        expected_prediction = run_relation_prediction(
            ObjectMemory.load(paths["object_memory.json"]),
            anchors,
            queries,
            calibration,
            source=prediction_source,
            created_at=str(prediction["created_at"]),
        )
        if prediction != expected_prediction:
            errors.append("prediction does not recompute exactly")
        evaluation_source = {
            "prediction": "prediction.json",
            "prediction_sha256": _sha(paths["prediction.json"]),
            "labels": "labels.json",
            "labels_sha256": _sha(paths["labels.json"]),
        }
        expected_evaluation = evaluate_relation_prediction(
            prediction,
            labels,
            source=evaluation_source,
            created_at=str(evaluation["created_at"]),
        )
        if evaluation != expected_evaluation:
            errors.append("evaluation does not recompute exactly")
        serialized_prediction = json.dumps(prediction, sort_keys=True).lower()
        for forbidden in (
            '"answerable"', '"answer_object_id"',
            '"expected_abstain_reason"', '"labels"', '"metrics"',
        ):
            if forbidden in serialized_prediction:
                errors.append(f"prediction leakage: {forbidden}")
        acceptance = evaluation.get("acceptance", {})
        if acceptance.get(
            "negative_rejection_counted_as_task_success"
        ) is not True:
            errors.append("negative rejection accounting is not explicit")
        if acceptance.get(
            "correct_rejections_excluded_from_answer_coverage"
        ) is not True:
            errors.append("correct rejections leaked into answer coverage")
        if acceptance.get("post_decision_confidence_inversion") is not False:
            errors.append("post-decision confidence inversion is forbidden")
        curve = evaluation.get("selective_answer_risk_coverage", [])
        answered = int(evaluation.get("metrics", {}).get("answered_count", -1))
        if len(curve) != answered:
            errors.append("answer coverage row count does not match answers")
        if curve and curve[-1]["coverage"] != (
            answered / evaluation["metrics"]["query_count"]
        ):
            errors.append("answer coverage denominator is not the frozen query set")
        if any(row.get("covered_by_current_policy") for row in evaluation["rows"] if row["abstain"]):
            errors.append("abstained query was marked covered")
        if any("decision_confidence" in row for row in evaluation["rows"]):
            errors.append("legacy decision confidence remains in evaluation")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(str(exc))
    return {
        "status": "PASS" if not errors else "FAIL",
        "stage": "D17",
        "completion": "CPU_COMPLETE",
        "real_data_calibration": "REAL_DATA_CALIBRATION_PENDING",
        "checks": {
            "prediction_label_free": not errors,
            "evaluation_recomputed": not errors,
            "negative_rejection_correct": not errors,
            "selective_answer_risk_recomputed": not errors,
            "raw_confidence_calibration_recomputed": not errors,
        },
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle")
    parser.add_argument("--write-report")
    args = parser.parse_args()
    result = validate(args.bundle)
    if args.write_report:
        Path(args.write_report).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
