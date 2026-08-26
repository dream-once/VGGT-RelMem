"""Evaluate frozen D12 A2 predictions with separate manual labels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
import time

from relground.a2_association import (
    A2_ASSOCIATION_ID,
    A2_EVALUATION_ACCEPTANCE_FIELDS,
    A2_EVALUATION_ARTIFACT_FIELDS,
    A2_EVALUATION_RESULT_FIELDS,
    A2_EVALUATION_SCHEMA_VERSION,
    A2_EVALUATION_SOURCE_FIELDS,
    EvidenceAssociationConfig,
    evaluate_a2_predictions,
)
from relground.association import ObjectMemory
from relground.d9_association import ManualInstanceLabels
from relground.observation_cache import sha256_file
from relground.schemas import RunManifest
from scripts.evaluate_d9_association import relative_evaluation_reference
from scripts.run_d9_association import (
    git_commit,
    prepare_output_dir,
    write_json,
)
from scripts.validate_a2_association import (
    read_json,
    resolve_bundle_reference,
    validate_output as validate_prediction_output,
)


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    project_root = Path(args.project_root).resolve()
    prediction_dir = Path(args.prediction_dir).resolve()
    input_labels_path = Path(args.labels).resolve()
    output_dir = Path(args.output_dir).resolve()

    prediction_report = validate_prediction_output(prediction_dir)
    if prediction_report["status"] != "PASS":
        raise ValueError(
            "A2 prediction bundle failed validation: "
            + "; ".join(prediction_report["failures"])
        )
    prepare_output_dir(output_dir)
    prediction_path = prediction_dir / "a2_result.json"
    prediction = read_json(prediction_path)
    prediction_hash = sha256_file(prediction_path)
    prediction_reference = relative_evaluation_reference(
        prediction_path, output_dir
    )
    source_memory_path = resolve_bundle_reference(
        prediction_dir,
        str(prediction["artifacts"]["source_memory"]),
    )
    source_memory = ObjectMemory.load(source_memory_path)

    labels = ManualInstanceLabels.load(input_labels_path)
    if labels.scene_id != prediction["scene_id"]:
        raise ValueError("A2 labels scene_id differs from prediction")
    if labels.query != prediction["query"]:
        raise ValueError("A2 labels query differs from prediction")
    labels_path = output_dir / "pair_labels.json"
    write_json(labels_path, labels.to_dict())
    labels_hash = sha256_file(labels_path)

    evaluation = evaluate_a2_predictions(
        list(source_memory.pending_observations.values()),
        prediction["pairs"],
        labels,
    )
    frozen_defaults = (
        EvidenceAssociationConfig.from_dict(prediction["config"]).to_dict()
        == EvidenceAssociationConfig().to_dict()
    )
    acceptance = {
        "prediction_valid": True,
        "evaluation_recomputed": True,
        "thresholds_frozen_before_evaluation": frozen_defaults,
    }
    if tuple(acceptance) != A2_EVALUATION_ACCEPTANCE_FIELDS:
        raise AssertionError("A2 evaluation acceptance fields changed")
    status = "PASS" if all(acceptance.values()) else "FAIL"
    source = {
        "prediction_result": prediction_reference,
        "prediction_result_sha256": prediction_hash,
        "pair_labels": labels_path.name,
        "pair_labels_sha256": labels_hash,
    }
    artifacts = {"pair_labels": labels_path.name}
    if tuple(source) != A2_EVALUATION_SOURCE_FIELDS:
        raise AssertionError("A2 evaluation source fields changed")
    if tuple(artifacts) != A2_EVALUATION_ARTIFACT_FIELDS:
        raise AssertionError("A2 evaluation artifact fields changed")
    result = {
        "schema_version": A2_EVALUATION_SCHEMA_VERSION,
        "status": status,
        "stage": "D12-A2-evaluation",
        "association_id": A2_ASSOCIATION_ID,
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
    if tuple(result) != A2_EVALUATION_RESULT_FIELDS:
        raise AssertionError("A2 evaluation fields changed")
    write_json(output_dir / "a2_evaluation.json", result)

    RunManifest(
        git_sha=git_commit(project_root),
        env_lock="D12 A2 evaluation is deterministic and model-free",
        dataset_split=labels.scene_id,
        seed=0,
        config={
            "stage": "D12-A2-evaluation",
            "status": "CPU_COMPLETE",
            "gpu_acceptance": "PENDING",
            "association_id": A2_ASSOCIATION_ID,
            "prediction_result": prediction_reference,
            "prediction_result_sha256": prediction_hash,
            "pair_labels": labels_path.name,
            "pair_labels_sha256": labels_hash,
            "thresholds_frozen_before_evaluation": frozen_defaults,
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
        default="runs/office-loop-mv-d12-a2-trash-can/prediction",
    )
    parser.add_argument(
        "--labels",
        default="configs/d9_office_loop_trash_can_labels.json",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/office-loop-mv-d12-a2-trash-can/evaluation",
    )
    parser.add_argument("--project-root", default=str(root))
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
