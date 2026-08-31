"""Evaluator-only Clio instance-association benchmark.

The association predictors run on D8 observations before task GT is read.  GT
OBBs are used only to label observation pairs for evaluation.  Because Clio
task annotations do not identify background instances, background/background
pairs are explicitly unknown rather than forced to be negative.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .a21_scale_association import (
    ScaleAwareAssociationConfig,
    predict_scale_aware_pairs,
)
from .a2_association import (
    EvidenceAssociationConfig,
    complete_link_clusters,
    predict_all_pairs as predict_a2_pairs,
)
from .association import ObjectMemory
from .clio_retrieval_evaluation import slugify_task
from .clio_task_evaluation import _parse_gt_boxes, point_in_obb
from .d9_association import SpatialGateConfig, predict_all_pairs as predict_a1_pairs


SCHEMA_VERSION = "0.1"
STAGE = "clio-instance-association-benchmark"
POLICY_IDS = ("A1", "A2", "A2.1-development")


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
        raise ValueError("association benchmark source escapes project root") from error


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


def _task_paths(run_root: Path, scene_id: str, task: str) -> tuple[Path, Path]:
    slug = slugify_task(task)
    if scene_id == "apartment":
        if task == "bring me a pillow":
            return (
                run_root / "d6-bring-me-a-pillow-k5/d6_result.json",
                run_root / "d8-bring-me-a-pillow-k5/object_memory.json",
            )
        return (
            run_root / f"dev-d6-{slug}-k5/d6_result.json",
            run_root / f"dev-d8-{slug}-k5/object_memory.json",
        )
    return (
        run_root / f"d6-{slug}-k5/d6_result.json",
        run_root / f"d8-{slug}-k5/object_memory.json",
    )


def _prediction_maps(
    observations: Sequence[Any],
    policy_ids: Sequence[str] = POLICY_IDS,
) -> dict[str, dict[tuple[str, str], bool]]:
    """Run requested label-free predictors before evaluator GT is read."""

    a1 = predict_a1_pairs(observations, SpatialGateConfig())
    raw_a2 = predict_a2_pairs(observations, EvidenceAssociationConfig())
    _, a2, _ = complete_link_clusters(observations, raw_a2)
    predictions = {
        "A1": {
            (item.obs_id_a, item.obs_id_b): bool(item.predicted_same)
            for item in a1
        },
        "A2": {
            (item.obs_id_a, item.obs_id_b): bool(item.predicted_same)
            for item in a2
        },
    }
    if "A2.1-development" in policy_ids:
        raw_a21, _ = predict_scale_aware_pairs(
            observations, ScaleAwareAssociationConfig()
        )
        _, a21, _ = complete_link_clusters(observations, raw_a21)
        predictions["A2.1-development"] = {
            (item.obs_id_a, item.obs_id_b): bool(item.predicted_same)
            for item in a21
        }
    unexpected = set(policy_ids) - set(predictions)
    if unexpected:
        raise ValueError(f"unsupported association policies: {sorted(unexpected)}")
    return {policy: predictions[policy] for policy in policy_ids}


def _world_center(observation: Any, alignment: Mapping[str, Any]) -> np.ndarray:
    scale = float(alignment["sim3"]["scale"])
    rotation = np.asarray(alignment["sim3"]["rotation"], dtype=np.float64)
    translation = np.asarray(alignment["sim3"]["translation"], dtype=np.float64)
    return scale * (rotation @ np.asarray(observation.center, dtype=np.float64)) + translation


def assign_observations_to_gt(
    observations: Sequence[Any],
    gt_boxes: Sequence[Mapping[str, Any]],
    alignment: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Assign target observations using the frozen alignment-RMSE margin."""

    padding = float(alignment["error_m"]["rmse"])
    assignments: list[dict[str, Any]] = []
    for observation in sorted(observations, key=lambda item: item.obs_id):
        center = _world_center(observation, alignment)
        options: list[dict[str, Any]] = []
        for box in gt_boxes:
            distance = float(np.linalg.norm(center - np.asarray(box["center"])))
            strict = point_in_obb(
                center,
                center=box["center"],
                extent=box["extent"],
                rotation=box["rotation"],
            )
            padded = point_in_obb(
                center,
                center=box["center"],
                extent=box["extent"],
                rotation=box["rotation"],
                padding_m=padding,
            )
            if padded:
                options.append({
                    "gt_id": str(box["gt_id"]),
                    "center_distance_m": distance,
                    "strict_containment": bool(strict),
                })
        options.sort(key=lambda item: (item["center_distance_m"], item["gt_id"]))
        selected = options[0] if options else None
        assignments.append({
            "observation_id": observation.obs_id,
            "frame_id": observation.frame_id,
            "assigned_gt_id": selected["gt_id"] if selected else None,
            "center_distance_m": selected["center_distance_m"] if selected else None,
            "strict_containment": selected["strict_containment"] if selected else False,
            "assignment_rule": (
                "nearest_of_alignment_rmse_padded_containing_gt_obbs"
                if selected else "unmatched_background_or_failed_localization"
            ),
            "ambiguous_padded_gt_count": len(options),
        })
    return assignments


def label_pair(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool | None:
    """Return same-instance label, or None for unlabelled background pairs."""

    first_gt = first["assigned_gt_id"]
    second_gt = second["assigned_gt_id"]
    if first_gt is None and second_gt is None:
        return None
    return bool(first_gt is not None and first_gt == second_gt)


def _empty_counts() -> dict[str, int]:
    return {
        "pair_count": 0,
        "positive_pairs": 0,
        "negative_pairs": 0,
        "true_positive": 0,
        "false_positive": 0,
        "true_negative": 0,
        "false_negative": 0,
    }


def _metrics(counts: Mapping[str, int]) -> dict[str, Any]:
    tp = int(counts["true_positive"])
    fp = int(counts["false_positive"])
    tn = int(counts["true_negative"])
    fn = int(counts["false_negative"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    pair_count = int(counts["pair_count"])
    return {
        **{key: int(value) for key, value in counts.items()},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / pair_count if pair_count else 0.0,
    }


def _accumulate(counts: dict[str, int], expected: bool, predicted: bool) -> None:
    counts["pair_count"] += 1
    counts["positive_pairs" if expected else "negative_pairs"] += 1
    if expected and predicted:
        counts["true_positive"] += 1
    elif expected:
        counts["false_negative"] += 1
    elif predicted:
        counts["false_positive"] += 1
    else:
        counts["true_negative"] += 1


def build_clio_association_benchmark(
    *,
    project_root: Path,
    query_manifest_path: Path,
    task_yaml_path: Path,
    world_alignment_path: Path,
    run_root: Path,
    created_at: str | None = None,
) -> dict[str, Any]:
    root = project_root.resolve()
    query_manifest = json.loads(query_manifest_path.read_text(encoding="utf-8"))
    alignment = json.loads(world_alignment_path.read_text(encoding="utf-8"))
    if alignment.get("status") != "PASS" or alignment.get("contract", {}).get("use") != "evaluator_only":
        raise ValueError("association benchmark requires passing evaluator-only alignment")
    scene_id = str(query_manifest["scene_id"])
    split_role = str(query_manifest["role"])
    policy_ids = POLICY_IDS if split_role == "development" else POLICY_IDS[:2]
    task_rows: list[dict[str, Any]] = []
    totals = {policy: _empty_counts() for policy in policy_ids}
    task_recovered = {policy: 0 for policy in policy_ids}
    source_rows: list[dict[str, Any]] = []

    for query in query_manifest["queries"]:
        if query["split"] != split_role:
            continue
        task = str(query["task"])
        d6_path, d8_path = _task_paths(run_root, scene_id, task)
        d6 = json.loads(d6_path.read_text(encoding="utf-8"))
        source_row = {
            "task": task,
            "d6_result": _relative(root, d6_path),
            "d6_result_sha256": _sha256(d6_path),
            "d8_memory": None,
            "d8_memory_sha256": None,
        }
        if not d8_path.is_file():
            task_rows.append({
                "task": task,
                "d6_status": str(d6["status"]),
                "status": "PIPELINE_NO_D8_MEMORY",
                "observation_count": 0,
                "matched_target_observations": 0,
                "unknown_background_pairs": 0,
                "evaluable_pairs": 0,
                "associable_target": False,
                "assignments": [],
                "pairs": [],
                "policy_target_recovered": {policy: False for policy in policy_ids},
            })
            source_rows.append(source_row)
            continue

        memory = ObjectMemory.load(d8_path)
        observations = sorted(memory.pending_observations.values(), key=lambda item: item.obs_id)
        predictions = _prediction_maps(observations, policy_ids)
        # GT is intentionally opened only after every policy prediction exists.
        gt_boxes = _parse_gt_boxes(task_yaml_path, task)
        assignments = assign_observations_to_gt(observations, gt_boxes, alignment)
        assignment_by_id = {item["observation_id"]: item for item in assignments}
        pairs: list[dict[str, Any]] = []
        unknown = 0
        associable = False
        recovered = {policy: False for policy in policy_ids}
        observation_by_id = {item.obs_id: item for item in observations}
        keys = sorted(next(iter(predictions.values()))) if predictions else []
        for first_id, second_id in keys:
            expected = label_pair(assignment_by_id[first_id], assignment_by_id[second_id])
            if expected is None:
                unknown += 1
            else:
                if expected and observation_by_id[first_id].frame_id != observation_by_id[second_id].frame_id:
                    associable = True
                for policy in policy_ids:
                    predicted = predictions[policy][(first_id, second_id)]
                    _accumulate(totals[policy], expected, predicted)
                    if expected and predicted and observation_by_id[first_id].frame_id != observation_by_id[second_id].frame_id:
                        recovered[policy] = True
            pairs.append({
                "observation_id_a": first_id,
                "observation_id_b": second_id,
                "expected_same": expected,
                "label_status": "UNKNOWN_BACKGROUND_PAIR" if expected is None else "EVALUABLE",
                "predictions": {
                    policy: predictions[policy][(first_id, second_id)]
                    for policy in policy_ids
                },
            })
        for policy in policy_ids:
            task_recovered[policy] += int(recovered[policy])
        task_rows.append({
            "task": task,
            "d6_status": str(d6["status"]),
            "status": "EVALUATED",
            "observation_count": len(observations),
            "matched_target_observations": sum(item["assigned_gt_id"] is not None for item in assignments),
            "unknown_background_pairs": unknown,
            "evaluable_pairs": len(pairs) - unknown,
            "associable_target": associable,
            "assignments": assignments,
            "pairs": pairs,
            "policy_target_recovered": recovered,
        })
        source_row["d8_memory"] = _relative(root, d8_path)
        source_row["d8_memory_sha256"] = _sha256(d8_path)
        source_rows.append(source_row)

    task_count = len(task_rows)
    associable_tasks = sum(bool(row["associable_target"]) for row in task_rows)
    metrics: dict[str, Any] = {}
    for policy in policy_ids:
        metrics[policy] = {
            **_metrics(totals[policy]),
            "associable_task_count": associable_tasks,
            "recovered_associable_tasks": task_recovered[policy],
            "conditional_associable_task_recall": (
                task_recovered[policy] / associable_tasks if associable_tasks else 0.0
            ),
            "full_pipeline_association_task_recall": (
                task_recovered[policy] / task_count if task_count else 0.0
            ),
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "stage": STAGE,
        "scene_id": scene_id,
        "split_role": split_role,
        "contract": {
            "prediction_runs_before_gt_read": True,
            "gt_and_world_alignment_are_evaluator_only": True,
            "target_assignment": "nearest official GT OBB containing center after alignment-RMSE padding",
            "background_background_pair_label": "UNKNOWN_EXCLUDED",
            "failed_tasks_remain_in_full_pipeline_denominator": True,
            "official_clio_metric_claim": False,
            "a21_status": "DEVELOPMENT_CANDIDATE_NOT_FROZEN_FOR_CONFIRMATORY_DATA",
        },
        "sources": {
            "query_manifest": _relative(root, query_manifest_path),
            "query_manifest_sha256": _sha256(query_manifest_path),
            "task_gt": _relative(root, task_yaml_path),
            "task_gt_sha256": _sha256(task_yaml_path),
            "world_alignment": _relative(root, world_alignment_path),
            "world_alignment_sha256": _sha256(world_alignment_path),
            "run_root": _relative(root, run_root),
            "artifacts": source_rows,
        },
        "config": {
            policy: {
                "A1": SpatialGateConfig().to_dict(),
                "A2": EvidenceAssociationConfig().to_dict(),
                "A2.1-development": ScaleAwareAssociationConfig().to_dict(),
            }[policy]
            for policy in policy_ids
        },
        "counts": {
            "frozen_tasks": task_count,
            "tasks_with_d8_memory": sum(row["status"] == "EVALUATED" for row in task_rows),
            "tasks_with_any_matched_target_observation": sum(row["matched_target_observations"] > 0 for row in task_rows),
            "associable_tasks": associable_tasks,
            "unknown_background_pairs_excluded": sum(row["unknown_background_pairs"] for row in task_rows),
        },
        "metrics": metrics,
        "tasks": task_rows,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }
    return _rounded(payload)


def validate_clio_association_benchmark(
    payload: Mapping[str, Any], *, project_root: Path
) -> dict[str, Any]:
    failures: list[str] = []
    try:
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported Clio association benchmark schema")
        sources = payload["sources"]
        root = project_root.resolve()

        def resolve(reference: str) -> Path:
            path = Path(reference)
            if path.is_absolute():
                raise ValueError("association benchmark references must be relative")
            resolved = (root / path).resolve()
            try:
                resolved.relative_to(root)
            except ValueError as error:
                raise ValueError("association benchmark reference escapes project root") from error
            return resolved

        recomputed = build_clio_association_benchmark(
            project_root=root,
            query_manifest_path=resolve(str(sources["query_manifest"])),
            task_yaml_path=resolve(str(sources["task_gt"])),
            world_alignment_path=resolve(str(sources["world_alignment"])),
            run_root=resolve(str(sources["run_root"])),
            created_at=str(payload["created_at"]),
        )
        if recomputed != dict(payload):
            raise ValueError("Clio association benchmark differs from deterministic replay")
        if payload["contract"]["background_background_pair_label"] != "UNKNOWN_EXCLUDED":
            raise ValueError("unknown background pairs were assigned fabricated labels")
        if payload.get("status") != "PASS":
            raise ValueError("Clio association benchmark did not pass")
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        failures.append(str(error))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not failures else "FAIL",
        "stage": f"{STAGE}-validation",
        "checks": {
            "portable_sources": not failures,
            "deterministic_replay": not failures,
            "gt_is_evaluator_only": not failures,
            "unknown_background_pairs_excluded": not failures,
            "full_denominator_retained": not failures,
        },
        "failures": failures,
    }
