"""Compose evaluator-only VGGT poses into the official Clio world frame."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .clio_pose_alignment import _relative, _rounded, _sha256_file


WORLD_ALIGNMENT_SCHEMA_VERSION = "0.1"
WORLD_ALIGNMENT_STAGE = "D16.4-vggt-to-clio-world-sim3"


def _validate_rigid_transform(matrix: np.ndarray) -> None:
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Clio scene transform must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("Clio scene transform has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("Clio scene transform rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError("Clio scene transform rotation is not proper")


def build_vggt_to_clio_world_alignment(
    *,
    project_root: Path,
    colmap_alignment_path: Path,
    scene_transform_path: Path,
    scene_id: str = "apartment",
    created_at: str | None = None,
    max_rmse_m: float = 0.15,
) -> dict[str, Any]:
    """Compose estimated VGGT-to-COLMAP with Clio's frozen evaluator transform."""

    root = project_root.resolve()
    colmap_payload = json.loads(colmap_alignment_path.read_text(encoding="utf-8"))
    if colmap_payload.get("status") != "PASS":
        raise ValueError("VGGT-to-COLMAP alignment must pass before composition")
    if colmap_payload.get("contract", {}).get("main_inference_may_read_alignment") is not False:
        raise ValueError("VGGT-to-COLMAP alignment is not evaluator-only")
    config = json.loads(scene_transform_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "0.1":
        raise ValueError("unsupported Clio scene-transform schema")
    try:
        scene = config["scenes"][scene_id]
        official_scale = float(scene["scale"])
        official_matrix = np.asarray(scene["T_world_from_scaled_colmap"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid Clio scene-transform entry for {scene_id}") from error
    if not np.isfinite(official_scale) or official_scale <= 0:
        raise ValueError("Clio scene scale must be positive")
    _validate_rigid_transform(official_matrix)

    colmap_scale = float(colmap_payload["sim3"]["scale"])
    colmap_rotation = np.asarray(colmap_payload["sim3"]["rotation"], dtype=np.float64)
    colmap_translation = np.asarray(colmap_payload["sim3"]["translation"], dtype=np.float64)
    official_rotation = official_matrix[:3, :3]
    official_translation = official_matrix[:3, 3]
    world_scale = official_scale * colmap_scale
    world_rotation = official_rotation @ colmap_rotation
    world_translation = official_scale * (official_rotation @ colmap_translation) + official_translation
    world_matrix = np.eye(4, dtype=np.float64)
    world_matrix[:3, :3] = world_scale * world_rotation
    world_matrix[:3, 3] = world_translation
    colmap_error = colmap_payload["error_colmap_units"]
    world_rmse = official_scale * float(colmap_error["rmse"])
    status = "PASS" if world_rmse <= max_rmse_m else "FAIL_ALIGNMENT_RMSE"

    payload = {
        "schema_version": WORLD_ALIGNMENT_SCHEMA_VERSION,
        "status": status,
        "stage": WORLD_ALIGNMENT_STAGE,
        "scene_id": scene_id,
        "split_role": scene["split_role"],
        "source": {
            "colmap_alignment": _relative(root, colmap_alignment_path),
            "colmap_alignment_sha256": _sha256_file(colmap_alignment_path),
            "scene_transform": _relative(root, scene_transform_path),
            "scene_transform_sha256": _sha256_file(scene_transform_path),
            "official_repository": config["source"]["repository"],
            "official_commit": config["source"]["commit"],
            "official_path": config["source"]["path"],
            "official_file_sha256": config["source"]["sha256"],
        },
        "contract": {
            "mapping": "clio_world_from_vggt",
            "use": "evaluator_only",
            "main_inference_may_read_alignment": False,
            "task_gt_coordinate_alignment": "OFFICIAL_CLIO_SCENE_TRANSFORM_FROZEN",
            "application": "world = T_rotation @ (scene_scale * colmap) + T_translation",
        },
        "components": {
            "colmap_from_vggt": {
                "scale": colmap_scale,
                "rotation": colmap_rotation,
                "translation": colmap_translation,
            },
            "world_from_colmap": {
                "scale": official_scale,
                "rotation": official_rotation,
                "translation": official_translation,
                "matrix_T": official_matrix,
            },
        },
        "sim3": {
            "scale": world_scale,
            "rotation": world_rotation,
            "translation": world_translation,
            "matrix": world_matrix,
        },
        "error_m": {
            "rmse": world_rmse,
            "median": official_scale * float(colmap_error["median"]),
            "max": official_scale * float(colmap_error["max"]),
            "threshold_rmse": float(max_rmse_m),
            "source": "VGGT-to-COLMAP camera-center residual scaled to Clio metric world",
        },
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }
    return _rounded(payload)


def validate_vggt_to_clio_world_alignment(
    payload: Mapping[str, Any], *, project_root: Path
) -> dict[str, Any]:
    failures: list[str] = []
    try:
        if payload.get("schema_version") != WORLD_ALIGNMENT_SCHEMA_VERSION:
            raise ValueError("unsupported VGGT-to-Clio-world alignment schema")
        root = project_root.resolve()
        source = payload["source"]

        def resolve(reference: str) -> Path:
            path = Path(reference)
            if path.is_absolute():
                raise ValueError("world-alignment references must be relative")
            resolved = (root / path).resolve()
            try:
                resolved.relative_to(root)
            except ValueError as error:
                raise ValueError("world-alignment reference escapes project root") from error
            return resolved

        recomputed = build_vggt_to_clio_world_alignment(
            project_root=root,
            colmap_alignment_path=resolve(str(source["colmap_alignment"])),
            scene_transform_path=resolve(str(source["scene_transform"])),
            scene_id=str(payload["scene_id"]),
            created_at=str(payload["created_at"]),
            max_rmse_m=float(payload["error_m"]["threshold_rmse"]),
        )
        if recomputed != dict(payload):
            raise ValueError("VGGT-to-Clio-world alignment differs from deterministic replay")
        if payload["contract"]["main_inference_may_read_alignment"] is not False:
            raise ValueError("Clio world alignment leaked into main inference")
        if payload["contract"]["task_gt_coordinate_alignment"] != "OFFICIAL_CLIO_SCENE_TRANSFORM_FROZEN":
            raise ValueError("Clio task-world alignment is not frozen")
        if payload.get("status") != "PASS":
            raise ValueError("Clio world alignment input failed")
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        failures.append(str(error))
    return {
        "schema_version": "0.1",
        "status": "PASS" if not failures else "FAIL",
        "stage": "D16.4-vggt-to-clio-world-validation",
        "checks": {
            "portable_sources": not failures,
            "official_transform_frozen": not failures,
            "sim3_composition_replayed": not failures,
            "evaluator_only_guard": not failures,
        },
        "failures": failures,
    }
