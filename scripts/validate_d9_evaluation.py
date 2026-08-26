"""Independently validate a labelled D9 evaluation bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from relground.association import ObjectMemory
from relground.d9_association import (
    D9_BASELINE_ID,
    D9_EVALUATION_ACCEPTANCE_FIELDS,
    D9_EVALUATION_ARTIFACT_FIELDS,
    D9_EVALUATION_RESULT_FIELDS,
    D9_EVALUATION_SCHEMA_VERSION,
    D9_EVALUATION_SOURCE_FIELDS,
    D9_METRIC_FIELDS,
    ManualInstanceLabels,
    evaluate_predictions,
)
from relground.observation_cache import sha256_file
from scripts.validate_d9_association import (
    read_json,
    resolve_bundle_reference,
    validate_output as validate_prediction_output,
)


def resolve_evaluation_reference(root: Path, reference: str) -> Path:
    relative = Path(reference)
    if relative.is_absolute():
        raise ValueError("D9 evaluation source paths must be relative")
    boundary = root.resolve().parent
    candidate = (root / relative).resolve()
    if candidate != boundary and boundary not in candidate.parents:
        raise ValueError(
            "D9 evaluation source path escapes the artifact bundle"
        )
    return candidate


def validate_output(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    failures: list[str] = []
    result_path = root / "d9_evaluation.json"
    labels_path = root / "pair_labels.json"
    run_manifest_path = root / "run_manifest.json"
    for artifact in (result_path, labels_path, run_manifest_path):
        if not artifact.is_file() or artifact.stat().st_size == 0:
            failures.append(
                f"missing or empty artifact: {artifact.name}"
            )
    if failures:
        return {"status": "FAIL", "failures": failures}

    try:
        result = read_json(result_path)
        raw_labels = read_json(labels_path)
        run_manifest = read_json(run_manifest_path)
        labels = ManualInstanceLabels.from_dict(raw_labels)
        if set(result) != set(D9_EVALUATION_RESULT_FIELDS):
            raise ValueError("D9 evaluation result fields are not frozen")
        source = result["source"]
        artifacts = result["artifacts"]
        metrics = result["metrics"]
        acceptance = result["acceptance"]
        for payload, fields, name in (
            (source, D9_EVALUATION_SOURCE_FIELDS, "source"),
            (
                artifacts,
                D9_EVALUATION_ARTIFACT_FIELDS,
                "artifacts",
            ),
            (
                acceptance,
                D9_EVALUATION_ACCEPTANCE_FIELDS,
                "acceptance",
            ),
        ):
            if not isinstance(payload, Mapping) or set(payload) != set(fields):
                raise ValueError(
                    f"D9 evaluation {name} fields are not frozen"
                )
        if (
            not isinstance(metrics, Mapping)
            or set(metrics) != set(D9_METRIC_FIELDS)
        ):
            raise ValueError("D9 evaluation metric fields are not frozen")
        prediction_path = resolve_evaluation_reference(
            root,
            str(source["prediction_result"]),
        )
        referenced_labels_path = resolve_evaluation_reference(
            root,
            str(source["pair_labels"]),
        )
        artifact_labels_path = resolve_evaluation_reference(
            root,
            str(artifacts["pair_labels"]),
        )
        prediction_hash = sha256_file(prediction_path)
        labels_hash = sha256_file(referenced_labels_path)
        prediction = read_json(prediction_path)
        prediction_dir = prediction_path.parent
        prediction_source_path = resolve_bundle_reference(
            prediction_dir,
            str(prediction["artifacts"]["source_memory"]),
        )
        source_memory = ObjectMemory.load(prediction_source_path)
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        return {
            "status": "FAIL",
            "failures": [f"invalid D9 evaluation artifact: {error}"],
        }

    if result["schema_version"] != D9_EVALUATION_SCHEMA_VERSION:
        failures.append("unsupported D9 evaluation schema")
    if result["stage"] != "D9-evaluation":
        failures.append("result stage is not D9-evaluation")
    if result["baseline_id"] != D9_BASELINE_ID:
        failures.append("unexpected D9 baseline id")
    if prediction_hash != source["prediction_result_sha256"]:
        failures.append("D9 prediction result hash changed")
    if labels_hash != source["pair_labels_sha256"]:
        failures.append("D9 pair label hash changed")
    if referenced_labels_path != labels_path.resolve():
        failures.append("D9 pair label source reference is inconsistent")
    if artifact_labels_path != labels_path.resolve():
        failures.append("D9 pair label artifact reference is inconsistent")
    if artifacts != {"pair_labels": labels_path.name}:
        failures.append("D9 evaluation artifact references are inconsistent")
    if raw_labels != labels.to_dict():
        failures.append("D9 pair labels are not canonical")
    if result["scene_id"] != labels.scene_id:
        failures.append("result scene_id differs from pair labels")
    if result["query"] != labels.query:
        failures.append("result query differs from pair labels")
    if prediction.get("scene_id") != labels.scene_id:
        failures.append("prediction scene_id differs from pair labels")
    if prediction.get("query") != labels.query:
        failures.append("prediction query differs from pair labels")

    prediction_report = validate_prediction_output(prediction_dir)
    if prediction_report["status"] != "PASS":
        failures.append(
            "referenced D9 prediction failed validation: "
            + "; ".join(prediction_report["failures"])
        )

    observations = list(source_memory.pending_observations.values())
    try:
        expected = evaluate_predictions(
            observations,
            prediction["pairs"],
            labels,
        )
    except (KeyError, TypeError, ValueError) as error:
        failures.append(f"cannot recompute D9 evaluation: {error}")
        expected = None

    if expected is not None:
        if result["pairs"] != expected["pairs"]:
            failures.append(
                "saved pairs differs from recomputed D9 evaluation"
            )
        if metrics != expected["metrics"]:
            failures.append(
                "saved metrics differs from recomputed D9 evaluation"
            )
        if result["failure_cases"] != expected["failure_cases"]:
            failures.append(
                "saved failure_cases differs from recomputed D9 evaluation"
            )
        try:
            min_pairwise_f1 = float(acceptance["min_pairwise_f1"])
        except (TypeError, ValueError):
            min_pairwise_f1 = float("nan")
        pairwise_f1_pass = (
            0.0 <= min_pairwise_f1 <= 1.0
            and expected["metrics"]["f1"] >= min_pairwise_f1
        )
        expected_acceptance = {
            "min_pairwise_f1": min_pairwise_f1,
            "pairwise_f1_pass": pairwise_f1_pass,
        }
        if acceptance != expected_acceptance:
            failures.append(
                "D9 evaluation acceptance flags are inconsistent"
            )
        expected_status = "PASS" if pairwise_f1_pass else "FAIL"
        if result["status"] != expected_status:
            failures.append(
                "D9 evaluation status is inconsistent with acceptance"
            )

    manifest_config = run_manifest.get("config", {})
    if not isinstance(manifest_config, Mapping):
        failures.append("run manifest config is not an object")
    else:
        expected_manifest_values = {
            "stage": "D9-evaluation",
            "pipeline": D9_BASELINE_ID,
            "prediction_result": source["prediction_result"],
            "prediction_result_sha256": prediction_hash,
            "pair_labels": source["pair_labels"],
            "pair_labels_sha256": labels_hash,
            "min_pairwise_f1": acceptance["min_pairwise_f1"],
        }
        for key, expected_value in expected_manifest_values.items():
            if manifest_config.get(key) != expected_value:
                failures.append(
                    f"run manifest {key} is inconsistent"
                )
    if run_manifest.get("dataset_split") != result["scene_id"]:
        failures.append("run manifest dataset split is inconsistent")
    if run_manifest.get("peak_vram_mb") is not None:
        failures.append(
            "model-free D9 evaluation unexpectedly records GPU memory"
        )

    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "scene_id": result.get("scene_id"),
        "query": result.get("query"),
        "pair_count": metrics.get("pair_count"),
        "positive_pairs": metrics.get("positive_pairs"),
        "negative_pairs": metrics.get("negative_pairs"),
        "pairwise_precision": metrics.get("precision"),
        "pairwise_recall": metrics.get("recall"),
        "pairwise_f1": metrics.get("f1"),
        "failure_case_count": len(result.get("failure_cases", [])),
        "prediction_status": prediction_report["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    parser.add_argument("--report")
    args = parser.parse_args()
    report = validate_output(args.output_dir)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
