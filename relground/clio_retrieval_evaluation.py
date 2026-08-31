"""Evaluator-only frustum coverage for Clio task retrieval."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import struct
from typing import Any, BinaryIO, Mapping, Sequence

import numpy as np
import yaml

from .clio_task_evaluation import quaternion_wxyz_to_rotation
from .clio_pose_alignment import _qvec_to_rotation


SCHEMA_VERSION = "0.1"
STAGE = "clio-retrieval-frustum-evaluation"
CAMERA_MODEL_PARAM_COUNTS = {
    0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4,
    9: 5, 10: 12,
}


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise ValueError("truncated COLMAP model")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("retrieval evaluation source escapes project root") from error


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


def slugify_task(task: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", task.lower())).strip("-")


def read_colmap_cameras(path: Path) -> dict[int, dict[str, Any]]:
    cameras: dict[int, dict[str, Any]] = {}
    with path.open("rb") as stream:
        count = struct.unpack("<Q", _read_exact(stream, 8))[0]
        for _ in range(count):
            camera_id, model_id = struct.unpack("<ii", _read_exact(stream, 8))
            width, height = struct.unpack("<QQ", _read_exact(stream, 16))
            if model_id not in CAMERA_MODEL_PARAM_COUNTS:
                raise ValueError(f"unsupported COLMAP camera model id: {model_id}")
            param_count = CAMERA_MODEL_PARAM_COUNTS[model_id]
            params = np.asarray(struct.unpack(f"<{param_count}d", _read_exact(stream, 8 * param_count)))
            cameras[camera_id] = {
                "model_id": model_id,
                "width": int(width),
                "height": int(height),
                "params": params,
            }
    if not cameras:
        raise ValueError("COLMAP cameras.bin contains no cameras")
    return cameras


def read_colmap_image_records(path: Path) -> dict[str, dict[str, Any]]:
    images: dict[str, dict[str, Any]] = {}
    with path.open("rb") as stream:
        count = struct.unpack("<Q", _read_exact(stream, 8))[0]
        for _ in range(count):
            _image_id = struct.unpack("<i", _read_exact(stream, 4))[0]
            qvec = np.asarray(struct.unpack("<4d", _read_exact(stream, 32)))
            tvec = np.asarray(struct.unpack("<3d", _read_exact(stream, 24)))
            camera_id = struct.unpack("<i", _read_exact(stream, 4))[0]
            name_bytes = bytearray()
            while True:
                byte = _read_exact(stream, 1)
                if byte == b"\0":
                    break
                name_bytes.extend(byte)
            point_count = struct.unpack("<Q", _read_exact(stream, 8))[0]
            stream.seek(point_count * 24, 1)
            frame_id = Path(name_bytes.decode("utf-8")).stem
            if frame_id in images:
                raise ValueError(f"duplicate COLMAP frame id: {frame_id}")
            images[frame_id] = {
                "camera_id": camera_id,
                "camera_from_colmap_rotation": _qvec_to_rotation(qvec),
                "camera_from_colmap_translation": tvec,
            }
    if not images:
        raise ValueError("COLMAP images.bin contains no images")
    return images


def _obb_points(box: Mapping[str, Any]) -> np.ndarray:
    center = np.asarray(box["center"], dtype=np.float64)
    extent = np.asarray(box["extents"], dtype=np.float64)
    rotation = quaternion_wxyz_to_rotation(box["rotation"])
    signs = np.asarray([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ], dtype=np.float64)
    corners = center + (rotation @ (signs * extent / 2.0).T).T
    return np.vstack([center, corners])


def _project(camera: Mapping[str, Any], points_camera: np.ndarray) -> np.ndarray:
    model_id = int(camera["model_id"])
    params = np.asarray(camera["params"], dtype=np.float64)
    normalized = points_camera[:, :2] / points_camera[:, 2, None]
    x, y = normalized[:, 0], normalized[:, 1]
    if model_id == 0:  # SIMPLE_PINHOLE
        f, cx, cy = params
        return np.column_stack([f * x + cx, f * y + cy])
    if model_id == 1:  # PINHOLE
        fx, fy, cx, cy = params
        return np.column_stack([fx * x + cx, fy * y + cy])
    if model_id == 2:  # SIMPLE_RADIAL
        f, cx, cy, k = params
        factor = 1.0 + k * (x * x + y * y)
        return np.column_stack([f * factor * x + cx, f * factor * y + cy])
    raise ValueError(f"projection is not implemented for camera model id {model_id}")


def box_intersects_frame(
    box: Mapping[str, Any],
    *,
    image: Mapping[str, Any],
    camera: Mapping[str, Any],
    world_from_colmap_scale: float,
    world_from_colmap_rotation: np.ndarray,
    world_from_colmap_translation: np.ndarray,
) -> bool:
    points_world = _obb_points(box)
    points_colmap = (
        world_from_colmap_rotation.T
        @ (points_world - world_from_colmap_translation).T
    ).T / world_from_colmap_scale
    rotation = np.asarray(image["camera_from_colmap_rotation"], dtype=np.float64)
    translation = np.asarray(image["camera_from_colmap_translation"], dtype=np.float64)
    points_camera = (rotation @ points_colmap.T).T + translation
    positive = points_camera[:, 2] > 1e-6
    if not positive.any():
        return False
    pixels = _project(camera, points_camera[positive])
    width, height = int(camera["width"]), int(camera["height"])
    return bool(
        pixels[:, 0].min() <= width - 1
        and pixels[:, 0].max() >= 0
        and pixels[:, 1].min() <= height - 1
        and pixels[:, 1].max() >= 0
    )


def build_clio_retrieval_evaluation(
    *,
    project_root: Path,
    query_manifest_path: Path,
    task_yaml_path: Path,
    scene_transform_path: Path,
    cameras_path: Path,
    images_path: Path,
    geometry_manifest_path: Path,
    retrieval_root: Path,
    created_at: str | None = None,
) -> dict[str, Any]:
    root = project_root.resolve()
    query_manifest = json.loads(query_manifest_path.read_text(encoding="utf-8"))
    task_gt = yaml.safe_load(task_yaml_path.read_text(encoding="utf-8"))
    scene_transforms = json.loads(scene_transform_path.read_text(encoding="utf-8"))
    geometry_manifest = json.loads(geometry_manifest_path.read_text(encoding="utf-8"))
    cameras = read_colmap_cameras(cameras_path)
    images = read_colmap_image_records(images_path)
    scene = scene_transforms["scenes"][query_manifest["scene_id"]]
    world_from_colmap_scale = float(scene["scale"])
    matrix = np.asarray(scene["T_world_from_scaled_colmap"], dtype=np.float64)
    world_from_colmap_rotation = matrix[:3, :3]
    world_from_colmap_translation = matrix[:3, 3]
    geometry_records = geometry_manifest.get("frames", geometry_manifest.get("records"))
    if not isinstance(geometry_records, list) or not geometry_records:
        raise ValueError("geometry manifest has no frame records")
    geometry_frames = [str(item["frame_id"]) for item in geometry_records]

    def visible_gt_ids(task: str, frame_ids: Sequence[str]) -> set[str]:
        visible: set[str] = set()
        for frame_id in frame_ids:
            if frame_id not in images:
                continue
            image = images[frame_id]
            camera = cameras[int(image["camera_id"])]
            for index, box in enumerate(task_gt[task]):
                if box_intersects_frame(
                    box,
                    image=image,
                    camera=camera,
                    world_from_colmap_scale=world_from_colmap_scale,
                    world_from_colmap_rotation=world_from_colmap_rotation,
                    world_from_colmap_translation=world_from_colmap_translation,
                ):
                    visible.add(f"gt_{index:04d}")
        return visible

    task_results: list[dict[str, Any]] = []
    for record in query_manifest["queries"]:
        task = str(record["task"])
        if task not in task_gt:
            raise ValueError(f"query manifest task is absent from GT: {task}")
        retrieval_dir = retrieval_root / f"retrieval-{slugify_task(task)}"
        eligible_gt = visible_gt_ids(task, geometry_frames)
        budgets: dict[str, Any] = {}
        source_files: dict[str, Any] = {}
        for budget in (1, 3, 5):
            selection_path = retrieval_dir / f"topk_{budget}.json"
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            if selection["query"] != task:
                raise ValueError(f"retrieval query mismatch for {task}")
            selected = [str(item["frame_id"]) for item in selection["frames"]]
            covered = visible_gt_ids(task, selected)
            frame_visibility = {
                frame_id: (
                    sorted(visible_gt_ids(task, [frame_id]))
                    if frame_id in images else None
                )
                for frame_id in selected
            }
            budgets[f"k{budget}"] = {
                "selected_frames": selected,
                "selected_frames_without_colmap_pose": [
                    frame_id for frame_id in selected if frame_id not in images
                ],
                "frame_visibility": frame_visibility,
                "visible_gt_ids": sorted(covered),
                "task_hit": bool(covered),
                "gt_recall_all": len(covered) / len(task_gt[task]),
                "gt_recall_sampled": len(covered) / len(eligible_gt) if eligible_gt else None,
            }
            source_files[f"topk_{budget}"] = {
                "path": _relative(root, selection_path),
                "sha256": _sha256_file(selection_path),
            }
        task_results.append({
            "task": task,
            "sam_query": str(record["sam_query"]),
            "split": str(record["split"]),
            "gt_count": len(task_gt[task]),
            "sampled_sequence_visible_gt_ids": sorted(eligible_gt),
            "sampled_sequence_visible_gt_count": len(eligible_gt),
            "budgets": budgets,
            "sources": source_files,
        })

    aggregates: dict[str, Any] = {}
    split_names = sorted({str(item["split"]) for item in task_results})
    for split in (*split_names, "all"):
        rows = task_results if split == "all" else [item for item in task_results if item["split"] == split]
        gt_count = sum(int(item["gt_count"]) for item in rows)
        aggregates[split] = {
            "task_count": len(rows),
            "gt_object_count": gt_count,
            "sampled_sequence_task_coverage": sum(bool(item["sampled_sequence_visible_gt_ids"]) for item in rows) / len(rows),
            **{
                f"task_hit_at_{budget}": sum(item["budgets"][f"k{budget}"]["task_hit"] for item in rows) / len(rows)
                for budget in (1, 3, 5)
            },
            **{
                f"gt_object_recall_at_{budget}": sum(
                    len(item["budgets"][f"k{budget}"]["visible_gt_ids"])
                    for item in rows
                ) / gt_count
                for budget in (1, 3, 5)
            },
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "stage": STAGE,
        "scope": "FRUSTUM_COVERAGE_UPPER_BOUND_NO_OCCLUSION",
        "contract": {
            "retrieval_reads_gt": False,
            "evaluation_reads_gt_after_retrieval": True,
            "visibility_uses_official_colmap_poses_and_intrinsics": True,
            "unregistered_frames_have_unknown_visibility": True,
            "occlusion_checked": False,
            "performance_claim": None,
        },
        "sources": {
            "query_manifest": _relative(root, query_manifest_path),
            "query_manifest_sha256": _sha256_file(query_manifest_path),
            "task_gt": _relative(root, task_yaml_path),
            "task_gt_sha256": _sha256_file(task_yaml_path),
            "scene_transform": _relative(root, scene_transform_path),
            "scene_transform_sha256": _sha256_file(scene_transform_path),
            "cameras": _relative(root, cameras_path),
            "cameras_sha256": _sha256_file(cameras_path),
            "images": _relative(root, images_path),
            "images_sha256": _sha256_file(images_path),
            "geometry_manifest": _relative(root, geometry_manifest_path),
            "geometry_manifest_sha256": _sha256_file(geometry_manifest_path),
            "retrieval_root": _relative(root, retrieval_root),
        },
        "geometry_frame_count": len(geometry_frames),
        "geometry_frames_without_colmap_pose": [
            frame_id for frame_id in geometry_frames if frame_id not in images
        ],
        "aggregates": aggregates,
        "tasks": task_results,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }
    return _rounded(payload)


def validate_clio_retrieval_evaluation(payload: Mapping[str, Any], *, project_root: Path) -> dict[str, Any]:
    failures: list[str] = []
    try:
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported Clio retrieval-evaluation schema")
        root = project_root.resolve()
        sources = payload["sources"]

        def resolve(reference: str) -> Path:
            path = Path(reference)
            if path.is_absolute():
                raise ValueError("retrieval-evaluation paths must be relative")
            resolved = (root / path).resolve()
            try:
                resolved.relative_to(root)
            except ValueError as error:
                raise ValueError("retrieval-evaluation path escapes project root") from error
            return resolved

        recomputed = build_clio_retrieval_evaluation(
            project_root=root,
            query_manifest_path=resolve(str(sources["query_manifest"])),
            task_yaml_path=resolve(str(sources["task_gt"])),
            scene_transform_path=resolve(str(sources["scene_transform"])),
            cameras_path=resolve(str(sources["cameras"])),
            images_path=resolve(str(sources["images"])),
            geometry_manifest_path=resolve(str(sources["geometry_manifest"])),
            retrieval_root=resolve(str(sources["retrieval_root"])),
            created_at=str(payload["created_at"]),
        )
        if recomputed != dict(payload):
            raise ValueError("Clio retrieval evaluation differs from deterministic replay")
        if payload["contract"]["retrieval_reads_gt"] is not False:
            raise ValueError("GT leaked into retrieval")
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
        },
        "failures": failures,
    }
