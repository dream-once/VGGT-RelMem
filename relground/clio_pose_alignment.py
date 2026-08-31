"""Evaluator-only alignment between VGGT anchors and Clio COLMAP poses."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Mapping

import numpy as np


ALIGNMENT_SCHEMA_VERSION = "0.2"
ALIGNMENT_STAGE = "D16.3-vggt-to-colmap-sim3"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"alignment input is outside project root: {path}") from error


def _read_exact(handle: Any, size: int) -> bytes:
    value = handle.read(size)
    if len(value) != size:
        raise ValueError("truncated COLMAP images.bin")
    return value


def _qvec_to_rotation(qvec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(qvec))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("invalid COLMAP quaternion")
    qw, qx, qy, qz = qvec / norm
    return np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def read_colmap_world_from_camera(path: Path) -> dict[str, np.ndarray]:
    """Read registered COLMAP image poses without requiring pycolmap.

    COLMAP stores ``camera_from_world`` as quaternion plus translation.  This
    function returns rigid ``world_from_camera`` matrices keyed by image stem.
    """

    poses: dict[str, np.ndarray] = {}
    with path.open("rb") as handle:
        image_count = struct.unpack("<Q", _read_exact(handle, 8))[0]
        for _ in range(image_count):
            _image_id = struct.unpack("<i", _read_exact(handle, 4))[0]
            qvec = np.asarray(struct.unpack("<4d", _read_exact(handle, 32)))
            tvec = np.asarray(struct.unpack("<3d", _read_exact(handle, 24)))
            _camera_id = struct.unpack("<i", _read_exact(handle, 4))[0]
            name_bytes = bytearray()
            while True:
                value = _read_exact(handle, 1)
                if value == b"\0":
                    break
                name_bytes.extend(value)
                if len(name_bytes) > 4096:
                    raise ValueError("invalid COLMAP image name")
            name = name_bytes.decode("utf-8")
            point_count = struct.unpack("<Q", _read_exact(handle, 8))[0]
            handle.seek(point_count * 24, 1)
            rotation = _qvec_to_rotation(qvec)
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = rotation.T
            pose[:3, 3] = -rotation.T @ tvec
            key = Path(name).stem
            if key in poses:
                raise ValueError(f"duplicate COLMAP image stem: {key}")
            poses[key] = pose
    if not poses:
        raise ValueError("COLMAP images.bin contains no registered images")
    return poses


def estimate_sim3(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate ``target = scale * rotation @ source + translation``."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Sim(3) inputs must have matching Nx3 shape")
    if len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Sim(3) requires at least three finite correspondences")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    if variance < 1e-12 or np.linalg.matrix_rank(source_centered) < 2:
        raise ValueError("Sim(3) source centers are degenerate")
    covariance = target_centered.T @ source_centered / len(source)
    left, singular, right_t = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(left @ right_t) < 0:
        sign[-1] = -1.0
    rotation = left @ np.diag(sign) @ right_t
    scale = float(np.sum(singular * sign) / variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Sim(3) scale must be positive")
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def _rounded(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return round(float(value), 12)
    if isinstance(value, np.ndarray):
        return _rounded(value.tolist())
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    return value


def build_vggt_to_colmap_alignment(
    *,
    project_root: Path,
    anchor_poses_path: Path,
    colmap_images_path: Path,
    created_at: str | None = None,
    min_matches: int = 6,
    max_rmse_colmap_units: float,
    scene_id: str = "apartment",
    split_role: str = "development",
) -> dict[str, Any]:
    project_root = project_root.resolve()
    anchors_raw = json.loads(anchor_poses_path.read_text(encoding="utf-8"))
    if not isinstance(anchors_raw, Mapping):
        raise ValueError("anchor pose JSON must be an object")
    anchors: dict[str, np.ndarray] = {}
    for frame_id, value in anchors_raw.items():
        pose = np.asarray(value, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(f"invalid anchor pose for {frame_id}")
        anchors[str(frame_id)] = pose
    colmap = read_colmap_world_from_camera(colmap_images_path)
    matched = sorted(set(anchors) & set(colmap), key=lambda item: int(item.rsplit("_", 1)[-1]))
    if len(matched) < min_matches:
        raise ValueError(f"only {len(matched)} matched poses; need at least {min_matches}")
    source = np.asarray([anchors[key][:3, 3] for key in matched])
    target = np.asarray([colmap[key][:3, 3] for key in matched])
    scale, rotation, translation = estimate_sim3(source, target)
    aligned = (scale * (rotation @ source.T)).T + translation
    errors = np.linalg.norm(aligned - target, axis=1)
    rmse = float(np.sqrt(np.mean(errors * errors)))
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    status = "PASS" if rmse <= max_rmse_colmap_units else "FAIL_ALIGNMENT_RMSE"
    payload = {
        "schema_version": ALIGNMENT_SCHEMA_VERSION,
        "status": status,
        "stage": ALIGNMENT_STAGE,
        "scene_id": scene_id,
        "split_role": split_role,
        "source": {
            "anchor_poses": _relative(project_root, anchor_poses_path),
            "anchor_poses_sha256": _sha256_file(anchor_poses_path),
            "colmap_images": _relative(project_root, colmap_images_path),
            "colmap_images_sha256": _sha256_file(colmap_images_path),
        },
        "contract": {
            "mapping": "colmap_from_vggt",
            "use": "evaluator_only",
            "main_inference_may_read_alignment": False,
            "task_gt_coordinate_alignment": "ROS_BAG_ALIGNMENT_PENDING",
        },
        "matches": {
            "count": len(matched),
            "frame_ids": matched,
            "anchor_count": len(anchors),
            "colmap_registered_count": len(colmap),
        },
        "sim3": {
            "scale": scale,
            "rotation": rotation,
            "translation": translation,
            "matrix": matrix,
        },
        "error_colmap_units": {
            "rmse": rmse,
            "median": float(np.median(errors)),
            "max": float(np.max(errors)),
            "threshold_rmse": float(max_rmse_colmap_units),
            "per_frame": [
                {"frame_id": key, "position_error": float(error)}
                for key, error in zip(matched, errors)
            ],
        },
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }
    return _rounded(payload)


def validate_vggt_to_colmap_alignment(
    payload: Mapping[str, Any], *, project_root: Path
) -> dict[str, Any]:
    failures: list[str] = []
    try:
        if payload.get("schema_version") != ALIGNMENT_SCHEMA_VERSION:
            raise ValueError("unsupported VGGT-to-COLMAP alignment schema")
        source = payload["source"]
        root = project_root.resolve()

        def resolve(reference: str) -> Path:
            path = Path(reference)
            if path.is_absolute():
                raise ValueError("alignment references must be relative")
            resolved = (root / path).resolve()
            try:
                resolved.relative_to(root)
            except ValueError as error:
                raise ValueError("alignment reference escapes project root") from error
            return resolved

        recomputed = build_vggt_to_colmap_alignment(
            project_root=root,
            anchor_poses_path=resolve(str(source["anchor_poses"])),
            colmap_images_path=resolve(str(source["colmap_images"])),
            created_at=str(payload["created_at"]),
            min_matches=6,
            max_rmse_colmap_units=float(payload["error_colmap_units"]["threshold_rmse"]),
            scene_id=str(payload["scene_id"]),
            split_role=str(payload["split_role"]),
        )
        if recomputed != dict(payload):
            raise ValueError("VGGT-to-COLMAP alignment differs from deterministic replay")
        if payload["contract"]["main_inference_may_read_alignment"] is not False:
            raise ValueError("alignment leaked into main inference")
        if payload.get("status") != "PASS":
            raise ValueError("alignment RMSE exceeds frozen threshold")
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        failures.append(str(error))
    return {
        "schema_version": "0.1",
        "status": "PASS" if not failures else "FAIL",
        "stage": "D16.3-vggt-to-colmap-validation",
        "checks": {
            "portable_sources": not failures,
            "colmap_pose_replay": not failures,
            "sim3_replay": not failures,
            "evaluator_only_guard": not failures,
        },
        "failures": failures,
    }
