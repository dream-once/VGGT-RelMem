"""Evaluator-only Clio task grounding metrics in the official world frame."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


CLIO_TASK_EVALUATION_SCHEMA_VERSION = "0.1"
CLIO_TASK_EVALUATION_STAGE = "clio-task-grounding-evaluation"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("evaluation source escapes project root") from error


def _round(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _round(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _round(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), 12)
    if isinstance(value, np.integer):
        return int(value)
    return value


def quaternion_wxyz_to_rotation(rotation: Mapping[str, Any]) -> np.ndarray:
    quaternion = np.asarray(
        [rotation["w"], rotation["x"], rotation["y"], rotation["z"]],
        dtype=np.float64,
    )
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("GT quaternion must be finite w/x/y/z")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0:
        raise ValueError("GT quaternion has zero norm")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def point_in_obb(
    point: Sequence[float],
    *,
    center: Sequence[float],
    extent: Sequence[float],
    rotation: Sequence[Sequence[float]],
    padding_m: float = 0.0,
) -> bool:
    point_array = np.asarray(point, dtype=np.float64)
    center_array = np.asarray(center, dtype=np.float64)
    extent_array = np.asarray(extent, dtype=np.float64)
    rotation_array = np.asarray(rotation, dtype=np.float64)
    if point_array.shape != (3,) or center_array.shape != (3,):
        raise ValueError("OBB point and center must have shape (3,)")
    if extent_array.shape != (3,) or np.any(extent_array <= 0):
        raise ValueError("OBB extent must contain three positive full lengths")
    if rotation_array.shape != (3, 3):
        raise ValueError("OBB rotation must have shape (3, 3)")
    if padding_m < 0:
        raise ValueError("OBB padding must be non-negative")
    local = rotation_array.T @ (point_array - center_array)
    return bool(np.all(np.abs(local) <= extent_array / 2.0 + padding_m + 1e-12))


def _transform_object(obj: Mapping[str, Any], alignment: Mapping[str, Any]) -> dict[str, Any]:
    scale = float(alignment["sim3"]["scale"])
    rotation = np.asarray(alignment["sim3"]["rotation"], dtype=np.float64)
    translation = np.asarray(alignment["sim3"]["translation"], dtype=np.float64)
    center = np.asarray(obj["fused_center"], dtype=np.float64)
    obb = obj["fused_obb"]
    obb_center = np.asarray(obb["center"], dtype=np.float64)
    obb_rotation = np.asarray(obb["rotation"], dtype=np.float64)
    obb_extent = np.asarray(obb["extent"], dtype=np.float64)
    if scale <= 0 or rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("invalid evaluator world Sim(3)")
    return {
        "object_id": str(obj["object_id"]),
        "confidence": float(obj["confidence"]),
        "observation_count": len(obj["observations"]),
        "frame_ids": sorted({str(item["frame_id"]) for item in obj["observations"]}),
        "center_world_m": scale * (rotation @ center) + translation,
        "obb_world": {
            "center": scale * (rotation @ obb_center) + translation,
            "extent": scale * obb_extent,
            "rotation": rotation @ obb_rotation,
        },
    }


def _parse_gt_boxes(task_yaml_path: Path, task_query: str) -> list[dict[str, Any]]:
    payload = yaml.safe_load(task_yaml_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or task_query not in payload:
        raise ValueError(f"task query is absent from Clio GT: {task_query}")
    boxes: list[dict[str, Any]] = []
    for index, item in enumerate(payload[task_query]):
        center = np.asarray(item["center"], dtype=np.float64)
        extent = np.asarray(item["extents"], dtype=np.float64)
        rotation = quaternion_wxyz_to_rotation(item["rotation"])
        if center.shape != (3,) or extent.shape != (3,) or np.any(extent <= 0):
            raise ValueError("invalid Clio GT OBB")
        boxes.append({
            "gt_id": f"gt_{index:04d}",
            "center": center,
            "extent": extent,
            "rotation": rotation,
        })
    if not boxes:
        raise ValueError("Clio task has no GT objects")
    return boxes


def build_clio_task_evaluation(
    *,
    project_root: Path,
    object_memory_path: Path,
    world_alignment_path: Path,
    task_yaml_path: Path,
    task_query: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Evaluate frozen predictions; GT and alignment remain evaluator-only inputs."""

    root = project_root.resolve()
    memory = json.loads(object_memory_path.read_text(encoding="utf-8"))
    alignment = json.loads(world_alignment_path.read_text(encoding="utf-8"))
    if alignment.get("status") != "PASS":
        raise ValueError("world alignment must pass before task evaluation")
    contract = alignment.get("contract", {})
    if contract.get("use") != "evaluator_only" or contract.get("main_inference_may_read_alignment") is not False:
        raise ValueError("world alignment is not evaluator-only")
    segmentation_query = str(memory.get("metadata", {}).get("query", "")).strip()
    if not segmentation_query:
        raise ValueError("prediction memory has no segmentation query")

    gt_boxes = _parse_gt_boxes(task_yaml_path, task_query)
    predictions = [_transform_object(obj, alignment) for obj in memory.get("objects", [])]
    predictions.sort(key=lambda item: (-item["confidence"], item["object_id"]))
    alignment_padding_m = float(alignment["error_m"]["rmse"])

    comparisons: list[dict[str, Any]] = []
    for prediction in predictions:
        for gt_box in gt_boxes:
            distance = float(np.linalg.norm(prediction["center_world_m"] - gt_box["center"]))
            pred_center_in_gt = point_in_obb(
                prediction["center_world_m"],
                center=gt_box["center"],
                extent=gt_box["extent"],
                rotation=gt_box["rotation"],
            )
            pred_center_in_gt_with_alignment_margin = point_in_obb(
                prediction["center_world_m"],
                center=gt_box["center"],
                extent=gt_box["extent"],
                rotation=gt_box["rotation"],
                padding_m=alignment_padding_m,
            )
            gt_center_in_pred = point_in_obb(
                gt_box["center"],
                center=prediction["obb_world"]["center"],
                extent=prediction["obb_world"]["extent"],
                rotation=prediction["obb_world"]["rotation"],
            )
            comparisons.append({
                "object_id": prediction["object_id"],
                "gt_id": gt_box["gt_id"],
                "center_distance_m": distance,
                "predicted_center_in_gt_obb": pred_center_in_gt,
                "predicted_center_in_gt_obb_with_alignment_rmse_margin": pred_center_in_gt_with_alignment_margin,
                "gt_center_in_predicted_obb": gt_center_in_pred,
                "mutual_center_containment": pred_center_in_gt and gt_center_in_pred,
            })

    by_object: dict[str, list[dict[str, Any]]] = {}
    for comparison in comparisons:
        by_object.setdefault(comparison["object_id"], []).append(comparison)
    object_results: list[dict[str, Any]] = []
    for rank, prediction in enumerate(predictions, start=1):
        options = sorted(
            by_object[prediction["object_id"]],
            key=lambda item: (item["center_distance_m"], item["gt_id"]),
        )
        best = options[0]
        object_results.append({
            "rank": rank,
            **prediction,
            "nearest_gt": best,
        })

    top1 = object_results[0] if object_results else None
    top1_center_correct = bool(top1 and top1["nearest_gt"]["predicted_center_in_gt_obb"])
    top1_margin_correct = bool(
        top1 and top1["nearest_gt"]["predicted_center_in_gt_obb_with_alignment_rmse_margin"]
    )
    covered_gt = {
        item["gt_id"]
        for item in comparisons
        if item["predicted_center_in_gt_obb"]
    }
    covered_gt_margin = {
        item["gt_id"]
        for item in comparisons
        if item["predicted_center_in_gt_obb_with_alignment_rmse_margin"]
    }
    correct_predictions = {
        item["object_id"]
        for item in comparisons
        if item["predicted_center_in_gt_obb"]
    }

    gt_count = len(gt_boxes)
    prediction_count = len(predictions)
    result = {
        "schema_version": CLIO_TASK_EVALUATION_SCHEMA_VERSION,
        "status": "PASS",
        "stage": CLIO_TASK_EVALUATION_STAGE,
        "scope": "APARTMENT_DEVELOPMENT_EVALUATOR_ONLY",
        "scene_id": str(alignment["scene_id"]),
        "task_query": task_query,
        "segmentation_query": segmentation_query,
        "contract": {
            "prediction_reads_gt": False,
            "prediction_reads_world_alignment": False,
            "evaluation_reads_gt_after_prediction": True,
            "metric_family": "oriented_center_containment",
            "official_clio_metric_claim": False,
            "note": "Clio official weak/strict metrics also use IoU-based greedy matching; this report intentionally does not claim byte-for-byte official evaluation.",
        },
        "sources": {
            "object_memory": _relative(root, object_memory_path),
            "object_memory_sha256": _sha256_file(object_memory_path),
            "world_alignment": _relative(root, world_alignment_path),
            "world_alignment_sha256": _sha256_file(world_alignment_path),
            "task_gt": _relative(root, task_yaml_path),
            "task_gt_sha256": _sha256_file(task_yaml_path),
        },
        "alignment_error_m": alignment["error_m"],
        "counts": {
            "predicted_permanent_objects": prediction_count,
            "pending_observations_excluded": len(memory.get("pending_observations", [])),
            "gt_objects": gt_count,
        },
        "metrics": {
            "center_grounding_acc_at_1": 1.0 if top1_center_correct else 0.0,
            "center_grounding_acc_at_1_with_alignment_rmse_margin": 1.0 if top1_margin_correct else 0.0,
            "gt_center_coverage": len(covered_gt) / gt_count,
            "gt_center_coverage_with_alignment_rmse_margin": len(covered_gt_margin) / gt_count,
            "prediction_center_precision": len(correct_predictions) / prediction_count if prediction_count else 0.0,
        },
        "objects": object_results,
        "gt_boxes": gt_boxes,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }
    return _round(result)


def validate_clio_task_evaluation(payload: Mapping[str, Any], *, project_root: Path) -> dict[str, Any]:
    failures: list[str] = []
    try:
        if payload.get("schema_version") != CLIO_TASK_EVALUATION_SCHEMA_VERSION:
            raise ValueError("unsupported Clio task-evaluation schema")
        sources = payload["sources"]
        root = project_root.resolve()

        def resolve(reference: str) -> Path:
            path = Path(reference)
            if path.is_absolute():
                raise ValueError("evaluation references must be relative")
            resolved = (root / path).resolve()
            try:
                resolved.relative_to(root)
            except ValueError as error:
                raise ValueError("evaluation reference escapes project root") from error
            return resolved

        recomputed = build_clio_task_evaluation(
            project_root=root,
            object_memory_path=resolve(str(sources["object_memory"])),
            world_alignment_path=resolve(str(sources["world_alignment"])),
            task_yaml_path=resolve(str(sources["task_gt"])),
            task_query=str(payload["task_query"]),
            created_at=str(payload["created_at"]),
        )
        if recomputed != dict(payload):
            raise ValueError("Clio task evaluation differs from deterministic replay")
        if payload["contract"]["prediction_reads_gt"] is not False:
            raise ValueError("GT leaked into prediction")
        if payload["contract"]["prediction_reads_world_alignment"] is not False:
            raise ValueError("world alignment leaked into prediction")
        if payload.get("status") != "PASS":
            raise ValueError("Clio task evaluation did not pass")
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        failures.append(str(error))
    return {
        "schema_version": CLIO_TASK_EVALUATION_SCHEMA_VERSION,
        "status": "PASS" if not failures else "FAIL",
        "stage": f"{CLIO_TASK_EVALUATION_STAGE}-validation",
        "checks": {
            "portable_sources": not failures,
            "deterministic_replay": not failures,
            "gt_is_evaluator_only": not failures,
        },
        "failures": failures,
    }
