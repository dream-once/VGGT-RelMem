"""Compare Q0 single-view observations with Q1 Top-K A2 objects on Clio."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .clio_retrieval_evaluation import slugify_task
from .clio_task_evaluation import _parse_gt_boxes, _transform_object, point_in_obb


SCHEMA_VERSION = "0.2"
STAGE = "clio-grounding-benchmark"


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
        raise ValueError("benchmark source escapes project root") from error


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


def _quality(observation: Mapping[str, Any]) -> float:
    values = np.asarray([
        max(float(observation["retrieval_score"]), 0.0),
        max(float(observation["sam_score"]), 0.0),
        max(float(observation["valid_point_ratio"]), 0.0),
    ], dtype=np.float64)
    return float(np.prod(values) ** (1.0 / 3.0))


def _transform_center(center: Sequence[float], alignment: Mapping[str, Any]) -> np.ndarray:
    scale = float(alignment["sim3"]["scale"])
    rotation = np.asarray(alignment["sim3"]["rotation"], dtype=np.float64)
    translation = np.asarray(alignment["sim3"]["translation"], dtype=np.float64)
    return scale * (rotation @ np.asarray(center, dtype=np.float64)) + translation


def _evaluate_center(
    center_world: np.ndarray | None,
    gt_boxes: Sequence[Mapping[str, Any]],
    *,
    alignment_margin_m: float,
) -> dict[str, Any]:
    if center_world is None:
        return {
            "answered": False,
            "correct": False,
            "correct_with_alignment_rmse_margin": False,
            "nearest_gt_id": None,
            "center_distance_m": None,
        }
    comparisons: list[tuple[float, Mapping[str, Any]]] = []
    for box in gt_boxes:
        comparisons.append((float(np.linalg.norm(center_world - box["center"])), box))
    distance, nearest = min(comparisons, key=lambda item: (item[0], item[1]["gt_id"]))
    correct = point_in_obb(
        center_world,
        center=nearest["center"],
        extent=nearest["extent"],
        rotation=nearest["rotation"],
    )
    margin_correct = point_in_obb(
        center_world,
        center=nearest["center"],
        extent=nearest["extent"],
        rotation=nearest["rotation"],
        padding_m=alignment_margin_m,
    )
    return {
        "answered": True,
        "correct": correct,
        "correct_with_alignment_rmse_margin": margin_correct,
        "nearest_gt_id": nearest["gt_id"],
        "center_distance_m": distance,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    total = len(rows)
    answered = sum(bool(row[key]["answered"]) for row in rows)
    strict = sum(bool(row[key]["correct"]) for row in rows)
    margin = sum(bool(row[key]["correct_with_alignment_rmse_margin"]) for row in rows)
    return {
        "task_count": total,
        "answered_tasks": answered,
        "coverage": answered / total,
        "grounding_acc_at_1": strict / total,
        "grounding_acc_at_1_with_alignment_rmse_margin": margin / total,
        "conditional_acc_at_1": strict / answered if answered else 0.0,
        "conditional_acc_at_1_with_alignment_rmse_margin": margin / answered if answered else 0.0,
    }


def build_clio_grounding_benchmark(
    *,
    project_root: Path,
    query_manifest_path: Path,
    task_yaml_path: Path,
    world_alignment_path: Path,
    run_root: Path,
    frozen_policy_path: Path | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    root = project_root.resolve()
    query_manifest = json.loads(query_manifest_path.read_text(encoding="utf-8"))
    scene_id = str(query_manifest["scene_id"])
    split_role = str(query_manifest["role"])
    if split_role == "held-out" and frozen_policy_path is None:
        raise ValueError("held-out benchmark requires a frozen policy manifest")
    alignment = json.loads(world_alignment_path.read_text(encoding="utf-8"))
    if alignment.get("status") != "PASS" or alignment.get("contract", {}).get("use") != "evaluator_only":
        raise ValueError("benchmark requires a passing evaluator-only world alignment")
    margin_m = float(alignment["error_m"]["rmse"])
    task_rows: list[dict[str, Any]] = []
    artifact_sources: list[dict[str, Any]] = []

    for query in query_manifest["queries"]:
        if query["split"] != split_role:
            continue
        task = str(query["task"])
        slug = slugify_task(task)
        if scene_id == "apartment":
            d6_dir = (
                run_root / "d6-bring-me-a-pillow-k5"
                if task == "bring me a pillow"
                else run_root / f"dev-d6-{slug}-k5"
            )
        else:
            d6_dir = run_root / f"d6-{slug}-k5"
        result_path = d6_dir / "d6_result.json"
        observations_path = d6_dir / "observations.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        observations_payload = json.loads(observations_path.read_text(encoding="utf-8"))
        observations = observations_payload.get("observations", observations_payload)
        if not isinstance(observations, list):
            raise ValueError(f"invalid D6 observations for {task}")
        top1_frame = str(result["selected_frames"][0]["frame_id"])
        top1_observations = [item for item in observations if str(item["frame_id"]) == top1_frame]
        top1_observations.sort(key=lambda item: (-_quality(item), str(item["obs_id"])))
        q0_center = (
            _transform_center(top1_observations[0]["center"], alignment)
            if top1_observations else None
        )

        if scene_id == "apartment":
            a2_memory_path = (
                run_root / "a2-bring-me-a-pillow-k5/prediction/object_memory.json"
                if task == "bring me a pillow"
                else run_root / f"dev-a2-{slug}-k5/prediction/object_memory.json"
            )
        else:
            a2_memory_path = run_root / f"a2-{slug}-k5/prediction/object_memory.json"
        q1_center: np.ndarray | None = None
        q1_object_id: str | None = None
        if a2_memory_path.is_file():
            memory = json.loads(a2_memory_path.read_text(encoding="utf-8"))
            objects = [_transform_object(item, alignment) for item in memory.get("objects", [])]
            objects.sort(key=lambda item: (-item["confidence"], item["object_id"]))
            if objects:
                q1_center = np.asarray(objects[0]["center_world_m"], dtype=np.float64)
                q1_object_id = str(objects[0]["object_id"])

        gt_boxes = _parse_gt_boxes(task_yaml_path, task)
        task_rows.append({
            "task": task,
            "sam_query": str(query["sam_query"]),
            "d6_status": str(result["status"]),
            "q0_top1": {
                "sam_calls": 1,
                "frame_id": top1_frame,
                "observation_id": str(top1_observations[0]["obs_id"]) if top1_observations else None,
                "center_world_m": q0_center,
                **_evaluate_center(q0_center, gt_boxes, alignment_margin_m=margin_m),
            },
            "q1_top5_a2": {
                "sam_calls": 5,
                "object_id": q1_object_id,
                "center_world_m": q1_center,
                **_evaluate_center(q1_center, gt_boxes, alignment_margin_m=margin_m),
            },
        })
        q1_row = task_rows[-1]["q1_top5_a2"]
        q0_row = task_rows[-1]["q0_top1"]
        fallback_source = (
            "a2_permanent_object"
            if q1_row["answered"]
            else "q0_single_view_fallback"
        )
        fallback_row = dict(q1_row if q1_row["answered"] else q0_row)
        fallback_row["source"] = fallback_source
        fallback_row["sam_calls"] = 5
        task_rows[-1]["q1f_top5_a2_with_q0_fallback"] = fallback_row
        source_row = {
            "task": task,
            "d6_result": _relative(root, result_path),
            "d6_result_sha256": _sha256(result_path),
            "d6_observations": _relative(root, observations_path),
            "d6_observations_sha256": _sha256(observations_path),
            "a2_memory": None,
            "a2_memory_sha256": None,
        }
        if a2_memory_path.is_file():
            source_row["a2_memory"] = _relative(root, a2_memory_path)
            source_row["a2_memory_sha256"] = _sha256(a2_memory_path)
        artifact_sources.append(source_row)

    q0 = _aggregate(task_rows, "q0_top1")
    q1 = _aggregate(task_rows, "q1_top5_a2")
    q1f = _aggregate(task_rows, "q1f_top5_a2_with_q0_fallback")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "stage": STAGE,
        "scope": f"{scene_id.upper()}_{split_role.upper().replace('-', '_')}",
        "scene_id": scene_id,
        "split_role": split_role,
        "contract": {
            "gt_is_evaluator_only": True,
            "q0": "Top-1 frame; highest-quality robust single-view observation",
            "q1": "fixed Top-5; frozen A2 permanent object ranked by confidence",
            "q1f": "development-derived deterministic fallback: Q1 object when available, otherwise Q0 observation",
            "strict_metric": "predicted center inside official oriented GT OBB",
            "uncertainty_metric": "same containment with measured alignment RMSE padding",
            "failed_or_abstained_tasks_remain_in_denominator": True,
            "official_clio_metric_claim": False,
            "frozen_before_heldout": (
                split_role != "held-out" or frozen_policy_path is not None
            ),
        },
        "sources": {
            "query_manifest": _relative(root, query_manifest_path),
            "query_manifest_sha256": _sha256(query_manifest_path),
            "task_gt": _relative(root, task_yaml_path),
            "task_gt_sha256": _sha256(task_yaml_path),
            "world_alignment": _relative(root, world_alignment_path),
            "world_alignment_sha256": _sha256(world_alignment_path),
            "run_root": _relative(root, run_root),
            "frozen_policy": (
                _relative(root, frozen_policy_path)
                if frozen_policy_path is not None else None
            ),
            "frozen_policy_sha256": (
                _sha256(frozen_policy_path)
                if frozen_policy_path is not None else None
            ),
            "artifacts": artifact_sources,
        },
        "alignment_rmse_m": margin_m,
        "metrics": {
            "q0_top1": q0,
            "q1_top5_a2": q1,
            "q1f_top5_a2_with_q0_fallback": q1f,
            "delta_q1_minus_q0": {
                key: q1[key] - q0[key]
                for key in (
                    "coverage", "grounding_acc_at_1",
                    "grounding_acc_at_1_with_alignment_rmse_margin",
                )
            },
            "delta_q1f_minus_q0": {
                key: q1f[key] - q0[key]
                for key in (
                    "coverage", "grounding_acc_at_1",
                    "grounding_acc_at_1_with_alignment_rmse_margin",
                )
            },
        },
        "tasks": task_rows,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }
    return _rounded(payload)


def validate_clio_grounding_benchmark(payload: Mapping[str, Any], *, project_root: Path) -> dict[str, Any]:
    failures: list[str] = []
    try:
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported Clio grounding benchmark schema")
        root = project_root.resolve()
        sources = payload["sources"]

        def resolve(reference: str) -> Path:
            path = Path(reference)
            if path.is_absolute():
                raise ValueError("benchmark references must be relative")
            resolved = (root / path).resolve()
            try:
                resolved.relative_to(root)
            except ValueError as error:
                raise ValueError("benchmark reference escapes project root") from error
            return resolved

        recomputed = build_clio_grounding_benchmark(
            project_root=root,
            query_manifest_path=resolve(str(sources["query_manifest"])),
            task_yaml_path=resolve(str(sources["task_gt"])),
            world_alignment_path=resolve(str(sources["world_alignment"])),
            run_root=resolve(str(sources["run_root"])),
            frozen_policy_path=(
                resolve(str(sources["frozen_policy"]))
                if sources.get("frozen_policy") is not None else None
            ),
            created_at=str(payload["created_at"]),
        )
        if recomputed != dict(payload):
            raise ValueError("Clio grounding benchmark differs from deterministic replay")
        if payload["contract"]["gt_is_evaluator_only"] is not True:
            raise ValueError("GT evaluator-only guard changed")
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        failures.append(str(error))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not failures else "FAIL",
        "stage": f"{STAGE}-validation",
        "checks": {
            "portable_sources": not failures,
            "deterministic_replay": not failures,
            "complete_split_denominator": not failures,
        },
        "failures": failures,
    }
