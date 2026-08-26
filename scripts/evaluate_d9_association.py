"""Evaluate a frozen D9 prediction bundle against separate manual labels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any

from relground.association import ObjectMemory
from relground.d9_association import (
    D9_BASELINE_ID,
    D9_EVALUATION_ACCEPTANCE_FIELDS,
    D9_EVALUATION_ARTIFACT_FIELDS,
    D9_EVALUATION_RESULT_FIELDS,
    D9_EVALUATION_SCHEMA_VERSION,
    D9_EVALUATION_SOURCE_FIELDS,
    ManualInstanceLabels,
    evaluate_predictions,
)
from relground.observation_cache import sha256_file
from relground.schemas import RunManifest
from scripts.run_d9_association import (
    git_commit,
    prepare_output_dir,
    write_json,
)
from scripts.validate_d9_association import (
    resolve_bundle_reference,
    validate_output as validate_prediction_output,
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} root must be an object")
    return payload


def relative_evaluation_reference(
    target: Path,
    output_dir: Path,
) -> str:
    """Return a relative path contained by the prediction/evaluation parent."""

    boundary = output_dir.resolve().parent
    resolved = target.resolve()
    if resolved != boundary and boundary not in resolved.parents:
        raise ValueError(
            "prediction and evaluation bundles must share one parent"
        )
    return Path(os.path.relpath(resolved, output_dir)).as_posix()


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    project_root = Path(args.project_root).resolve()
    prediction_dir = Path(args.prediction_dir).resolve()
    input_labels_path = Path(args.labels).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not 0.0 <= args.min_pairwise_f1 <= 1.0:
        raise ValueError("min_pairwise_f1 must be in [0, 1]")

    prediction_report = validate_prediction_output(prediction_dir)
    if prediction_report["status"] != "PASS":
        raise ValueError(
            "D9 prediction bundle failed validation: "
            + "; ".join(prediction_report["failures"])
        )
    prepare_output_dir(output_dir)

    prediction_path = prediction_dir / "d9_result.json"
    prediction = read_json(prediction_path)
    prediction_hash = sha256_file(prediction_path)
    prediction_reference = relative_evaluation_reference(
        prediction_path,
        output_dir,
    )
    source_memory_path = resolve_bundle_reference(
        prediction_dir,
        str(prediction["artifacts"]["source_memory"]),
    )
    source_memory = ObjectMemory.load(source_memory_path)

    labels = ManualInstanceLabels.load(input_labels_path)
    if labels.scene_id != prediction["scene_id"]:
        raise ValueError("D9 labels scene_id differs from prediction")
    if labels.query != prediction["query"]:
        raise ValueError("D9 labels query differs from prediction")
    labels_path = output_dir / "pair_labels.json"
    write_json(labels_path, labels.to_dict())
    labels_hash = sha256_file(labels_path)

    observations = list(source_memory.pending_observations.values())
    evaluation = evaluate_predictions(
        observations,
        prediction["pairs"],
        labels,
    )
    pairwise_f1_pass = (
        evaluation["metrics"]["f1"] >= args.min_pairwise_f1
    )
    acceptance = {
        "min_pairwise_f1": args.min_pairwise_f1,
        "pairwise_f1_pass": pairwise_f1_pass,
    }
    if tuple(acceptance) != D9_EVALUATION_ACCEPTANCE_FIELDS:
        raise AssertionError("D9 evaluation acceptance fields changed")
    source = {
        "prediction_result": prediction_reference,
        "prediction_result_sha256": prediction_hash,
        "pair_labels": labels_path.name,
        "pair_labels_sha256": labels_hash,
    }
    if tuple(source) != D9_EVALUATION_SOURCE_FIELDS:
        raise AssertionError("D9 evaluation source fields changed")
    artifacts = {"pair_labels": labels_path.name}
    if tuple(artifacts) != D9_EVALUATION_ARTIFACT_FIELDS:
        raise AssertionError("D9 evaluation artifact fields changed")
    status = "PASS" if pairwise_f1_pass else "FAIL"

    result = {
        "schema_version": D9_EVALUATION_SCHEMA_VERSION,
        "status": status,
        "stage": "D9-evaluation",
        "baseline_id": D9_BASELINE_ID,
        "scene_id": labels.scene_id,
        "query": labels.query,
        "source": source,
        "metrics": evaluation["metrics"],
        "pairs": evaluation["pairs"],
        "failure_cases": evaluation["failure_cases"],
        "acceptance": acceptance,
        "artifacts": artifacts,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if tuple(result) != D9_EVALUATION_RESULT_FIELDS:
        raise AssertionError("D9 evaluation result fields changed")
    write_json(output_dir / "d9_evaluation.json", result)

    RunManifest(
        git_sha=git_commit(project_root),
        env_lock="D9 evaluation is deterministic and model-free",
        dataset_split=labels.scene_id,
        seed=0,
        config={
            "stage": "D9-evaluation",
            "pipeline": D9_BASELINE_ID,
            "prediction_result": prediction_reference,
            "prediction_result_sha256": prediction_hash,
            "pair_labels": labels_path.name,
            "pair_labels_sha256": labels_hash,
            "min_pairwise_f1": args.min_pairwise_f1,
        },
        command=shlex.join(sys.argv),
        runtime_seconds=time.perf_counter() - started,
        peak_vram_mb=None,
    ).save(output_dir / "run_manifest.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 2


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-dir",
        default="runs/office-loop-mv-d9-trash-can/prediction",
    )
    parser.add_argument(
        "--labels",
        default="configs/d9_office_loop_trash_can_labels.json",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/office-loop-mv-d9-trash-can/evaluation",
    )
    parser.add_argument("--min-pairwise-f1", type=float, default=0.95)
    parser.add_argument("--project-root", default=str(root))
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
