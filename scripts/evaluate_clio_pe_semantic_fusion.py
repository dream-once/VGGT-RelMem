"""Evaluate frozen PE crop-fusion predictions against Clio task OBBs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from relground.clio_grounding_benchmark import (
    _evaluate_center,
    _transform_center,
)
from relground.clio_task_evaluation import _parse_gt_boxes
from relground.pe_semantic_fusion import (
    EVALUATION_STAGE,
    PREDICTION_STAGE,
    SCHEMA_VERSION,
    aggregate_task_results,
    paired_transitions,
    select_medoid,
    select_quality_representative,
    select_semantic_representative,
)


METHOD_KEYS = (
    "current_q1f",
    "quality_representative",
    "medoid_representative",
    "pe_semantic_representative",
    "raw_observation_oracle",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("evaluation source escapes project root") from error


def _rounded(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _rounded(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), 12)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _fallback_evaluation(
    baseline_task: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(baseline_task["q0_top1"])
    result["source"] = "q0_single_view_fallback"
    return result


def _center_evaluation(
    record: Mapping[str, Any],
    alignment: Mapping[str, Any],
    boxes: list[dict[str, Any]],
    margin_m: float,
    *,
    source: str,
) -> dict[str, Any]:
    center_world = _transform_center(record["center_vggt"], alignment)
    return {
        "source": source,
        "observation_id": record["observation_id"],
        "center_world_m": center_world,
        **_evaluate_center(
            center_world,
            boxes,
            alignment_margin_m=margin_m,
        ),
    }


def _oracle_evaluation(
    records: list[Mapping[str, Any]],
    alignment: Mapping[str, Any],
    boxes: list[dict[str, Any]],
    margin_m: float,
    fallback: Mapping[str, Any],
) -> dict[str, Any]:
    if not records:
        result = dict(fallback)
        result["source"] = "q0_single_view_fallback"
        return result
    evaluations = [
        _center_evaluation(
            record,
            alignment,
            boxes,
            margin_m,
            source="raw_observation_oracle",
        )
        for record in records
    ]
    strict = next(
        (item for item in evaluations if item["correct"]),
        None,
    )
    padded = next(
        (
            item for item in evaluations
            if item["correct_with_alignment_rmse_margin"]
        ),
        None,
    )
    representative = strict or padded or evaluations[0]
    result = dict(representative)
    result["answered"] = True
    result["correct"] = strict is not None
    result["correct_with_alignment_rmse_margin"] = padded is not None
    result["source"] = "raw_observation_oracle_evaluator_only"
    return result


def build_evaluation(
    *,
    project_root: Path,
    prediction_path: Path,
    grounding_benchmark_path: Path,
) -> dict[str, Any]:
    root = project_root.resolve()
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    grounding = json.loads(
        grounding_benchmark_path.read_text(encoding="utf-8")
    )
    if prediction.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported semantic-fusion prediction schema")
    if prediction.get("stage") != PREDICTION_STAGE:
        raise ValueError("input is not a semantic-fusion prediction")
    if prediction.get("status") != "PASS":
        raise ValueError("semantic-fusion prediction did not pass")
    contract = prediction.get("contract", {})
    if (
        contract.get("ground_truth_used") is not False
        or contract.get("world_alignment_used") is not False
        or contract.get("learned_parameters") is not False
    ):
        raise ValueError("prediction is not label-free and training-free")
    if grounding.get("status") != "PASS":
        raise ValueError("grounding benchmark did not pass")
    if prediction["scene_id"] != grounding["scene_id"]:
        raise ValueError("prediction and benchmark scene mismatch")
    if prediction["split_role"] != grounding["split_role"]:
        raise ValueError("prediction and benchmark role mismatch")
    baseline_by_task = {
        str(item["task"]): item for item in grounding["tasks"]
    }
    prediction_by_task = {
        str(item["task"]): item for item in prediction["tasks"]
    }
    if set(prediction_by_task) != set(baseline_by_task):
        raise ValueError("prediction task denominator changed")
    alignment_path = root / grounding["sources"]["world_alignment"]
    task_gt_path = root / grounding["sources"]["task_gt"]
    if _sha256(alignment_path) != grounding["sources"]["world_alignment_sha256"]:
        raise ValueError("world alignment hash changed")
    if _sha256(task_gt_path) != grounding["sources"]["task_gt_sha256"]:
        raise ValueError("task GT hash changed")
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    margin_m = float(grounding["alignment_rmse_m"])
    task_rows: list[dict[str, Any]] = []
    for task in sorted(baseline_by_task):
        baseline_task = baseline_by_task[task]
        prediction_task = prediction_by_task[task]
        selected_records = [
            item for item in prediction_task["observations"]
            if item["in_selected_object"]
        ]
        choices: dict[str, Mapping[str, Any] | None] = {
            "quality_representative": (
                select_quality_representative(selected_records)
                if selected_records else None
            ),
            "medoid_representative": (
                select_medoid(selected_records)
                if selected_records else None
            ),
            "pe_semantic_representative": (
                select_semantic_representative(selected_records)
                if selected_records else None
            ),
        }
        expected_selected = choices["pe_semantic_representative"]
        expected_id = (
            expected_selected["observation_id"]
            if expected_selected is not None else None
        )
        if expected_id != prediction_task["selected_observation_id"]:
            raise ValueError(
                f"semantic representative changed for task: {task}"
            )
        # GT is opened only after every non-oracle representative is fixed.
        boxes = _parse_gt_boxes(task_gt_path, task)
        row: dict[str, Any] = {
            "task": task,
            "sam_query": prediction_task["sam_query"],
            "selected_object_id": prediction_task["selected_object_id"],
            "current_q1f": dict(
                baseline_task["q1f_top5_a2_with_q0_fallback"]
            ),
        }
        for key, record in choices.items():
            row[key] = (
                _center_evaluation(
                    record,
                    alignment,
                    boxes,
                    margin_m,
                    source=key,
                )
                if record is not None
                else _fallback_evaluation(baseline_task)
            )
        row["raw_observation_oracle"] = _oracle_evaluation(
            list(prediction_task["observations"]),
            alignment,
            boxes,
            margin_m,
            baseline_task["q0_top1"],
        )
        task_rows.append(row)
    metrics = {
        key: aggregate_task_results(task_rows, key)
        for key in METHOD_KEYS
    }
    comparisons = {}
    for key in (
        "quality_representative",
        "medoid_representative",
        "pe_semantic_representative",
    ):
        comparisons[f"{key}_minus_current_q1f"] = {
            "strict_percentage_points": 100.0 * (
                metrics[key]["grounding_acc_at_1"]
                - metrics["current_q1f"]["grounding_acc_at_1"]
            ),
            "padded_percentage_points": 100.0 * (
                metrics[key][
                    "grounding_acc_at_1_with_alignment_rmse_margin"
                ]
                - metrics["current_q1f"][
                    "grounding_acc_at_1_with_alignment_rmse_margin"
                ]
            ),
            "paired_strict_transitions": paired_transitions(
                task_rows,
                "current_q1f",
                key,
            ),
        }
    comparisons["pe_minus_quality_representative"] = {
        "strict_percentage_points": 100.0 * (
            metrics["pe_semantic_representative"]["grounding_acc_at_1"]
            - metrics["quality_representative"]["grounding_acc_at_1"]
        ),
        "paired_strict_transitions": paired_transitions(
            task_rows,
            "quality_representative",
            "pe_semantic_representative",
        ),
    }
    role = str(prediction["split_role"])
    pe_transition = comparisons[
        "pe_semantic_representative_minus_current_q1f"
    ]["paired_strict_transitions"]
    development_gate = None
    if role == "development":
        development_gate = {
            "decision": (
                "PROCEED_TO_FIXED_CONFIRMATORY_DIAGNOSTIC"
                if (
                    comparisons[
                        "pe_semantic_representative_minus_current_q1f"
                    ]["strict_percentage_points"] > 0.0
                    and pe_transition["regressions"] == 0
                )
                else "STOP"
            ),
            "semantic_specific_gain_over_quality_baseline": (
                comparisons["pe_minus_quality_representative"][
                    "strict_percentage_points"
                ] > 0.0
            ),
            "training_claim": False,
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "stage": EVALUATION_STAGE,
        "experiment_id": prediction["experiment_id"],
        "scene_id": prediction["scene_id"],
        "split_role": role,
        "contract": {
            "official_clio_metric_claim": False,
            "metric": (
                "predicted representative center inside any official "
                "task GT OBB; nearest GT diagnostic only"
            ),
            "prediction_gt_free": True,
            "prediction_world_alignment_free": True,
            "learned_ranker": False,
            "base_object_selection_changed": False,
            "cubicle_scope": "fixed-confirmatory-not-untouched-held-out",
        },
        "sources": {
            "prediction": _relative(root, prediction_path),
            "prediction_sha256": _sha256(prediction_path),
            "grounding_benchmark": _relative(
                root,
                grounding_benchmark_path,
            ),
            "grounding_benchmark_sha256": _sha256(
                grounding_benchmark_path
            ),
            "task_gt": _relative(root, task_gt_path),
            "task_gt_sha256": _sha256(task_gt_path),
            "world_alignment": _relative(root, alignment_path),
            "world_alignment_sha256": _sha256(alignment_path),
        },
        "metrics": metrics,
        "comparisons": comparisons,
        "development_gate": development_gate,
        "tasks": task_rows,
    }
    return _rounded(payload)


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--grounding-benchmark", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.project_root).resolve()

    def resolve(value: str) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    payload = build_evaluation(
        project_root=root,
        prediction_path=resolve(args.prediction),
        grounding_benchmark_path=resolve(args.grounding_benchmark),
    )
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"],
        "scene_id": payload["scene_id"],
        "split_role": payload["split_role"],
        "metrics": payload["metrics"],
        "comparisons": payload["comparisons"],
        "development_gate": payload["development_gate"],
        "output": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
