"""NPZ contract exported by the isolated vggt_geom environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import json

import numpy as np


GEOMETRY_SCHEMA_VERSION = "0.2"
LEGACY_GEOMETRY_SCHEMA_VERSION = "0.1"


@dataclass(frozen=True)
class GeometryFrame:
    frame_id: str
    point_map: np.ndarray
    confidence_map: np.ndarray
    world_from_camera: np.ndarray
    raw_confidence_map: np.ndarray | None = None
    valid_mask: np.ndarray | None = None


@dataclass
class GeometryBundle:
    frame_ids: list[str]
    point_maps: np.ndarray
    confidence_maps: np.ndarray
    world_from_camera: np.ndarray
    raw_confidence_maps: np.ndarray | None = None
    valid_masks: np.ndarray | None = None
    schema_version: str = GEOMETRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        count = len(self.frame_ids)
        self.point_maps = np.asarray(self.point_maps)
        self.confidence_maps = np.asarray(self.confidence_maps)
        self.world_from_camera = np.asarray(self.world_from_camera)
        if (
            self.point_maps.ndim != 4
            or self.point_maps.shape[0] != count
            or self.point_maps.shape[-1] != 3
        ):
            raise ValueError("point_maps must have shape (N, H, W, 3)")
        expected_shape = self.point_maps.shape[:-1]
        if self.confidence_maps.shape != expected_shape:
            raise ValueError("confidence_maps must have shape (N, H, W)")
        if self.world_from_camera.shape != (count, 4, 4):
            raise ValueError(
                "world_from_camera must have shape (N, 4, 4)"
            )
        if len(set(self.frame_ids)) != count:
            raise ValueError("frame_ids must be unique")

        if self.raw_confidence_maps is not None:
            self.raw_confidence_maps = np.asarray(
                self.raw_confidence_maps
            )
            if self.raw_confidence_maps.shape != expected_shape:
                raise ValueError(
                    "raw_confidence_maps must have shape (N, H, W)"
                )
        if self.valid_masks is None:
            self.valid_masks = self.confidence_maps > 0
        else:
            self.valid_masks = np.asarray(self.valid_masks)
            if self.valid_masks.shape != expected_shape:
                raise ValueError(
                    "valid_masks must have shape (N, H, W)"
                )
            if self.valid_masks.dtype != np.bool_:
                raise ValueError("valid_masks must be boolean")

    def get(self, frame_id: str) -> GeometryFrame:
        try:
            index = self.frame_ids.index(frame_id)
        except ValueError as error:
            raise KeyError(
                f"unknown geometry frame: {frame_id}"
            ) from error
        raw_confidence = (
            None
            if self.raw_confidence_maps is None
            else self.raw_confidence_maps[index]
        )
        return GeometryFrame(
            frame_id=frame_id,
            point_map=self.point_maps[index],
            confidence_map=self.confidence_maps[index],
            world_from_camera=self.world_from_camera[index],
            raw_confidence_map=raw_confidence,
            valid_mask=self.valid_masks[index],
        )


def save_geometry_npz(
    path: str | Path,
    *,
    frame_ids: Sequence[str],
    point_maps: np.ndarray,
    confidence_maps: np.ndarray,
    world_from_camera: np.ndarray,
    raw_confidence_maps: np.ndarray | None = None,
    valid_masks: np.ndarray | None = None,
) -> None:
    extended_fields = (
        raw_confidence_maps is not None,
        valid_masks is not None,
    )
    if any(extended_fields) and not all(extended_fields):
        raise ValueError(
            "raw_confidence_maps and valid_masks must be saved together"
        )
    schema_version = (
        GEOMETRY_SCHEMA_VERSION
        if all(extended_fields)
        else LEGACY_GEOMETRY_SCHEMA_VERSION
    )
    bundle = GeometryBundle(
        frame_ids=list(frame_ids),
        point_maps=point_maps,
        confidence_maps=confidence_maps,
        world_from_camera=world_from_camera,
        raw_confidence_maps=raw_confidence_maps,
        valid_masks=valid_masks,
        schema_version=schema_version,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(bundle.schema_version),
        "frame_ids": np.asarray(bundle.frame_ids, dtype=np.str_),
        "point_maps": bundle.point_maps,
        "confidence_maps": bundle.confidence_maps,
        "world_from_camera": bundle.world_from_camera,
    }
    if bundle.raw_confidence_maps is not None:
        arrays["raw_confidence_maps"] = bundle.raw_confidence_maps
        arrays["valid_masks"] = bundle.valid_masks
    np.savez_compressed(output, **arrays)


def load_geometry_npz(path: str | Path) -> GeometryBundle:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "frame_ids",
            "point_maps",
            "confidence_maps",
            "world_from_camera",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(
                f"geometry archive missing fields: {sorted(missing)}"
            )
        version = (
            str(archive["schema_version"].item())
            if "schema_version" in archive
            else "unknown"
        )
        if version == GEOMETRY_SCHEMA_VERSION:
            extended = {"raw_confidence_maps", "valid_masks"}
            missing_extended = extended - set(archive.files)
            if missing_extended:
                raise ValueError(
                    "geometry 0.2 archive missing fields: "
                    f"{sorted(missing_extended)}"
                )
        return GeometryBundle(
            frame_ids=[
                str(item)
                for item in archive["frame_ids"].tolist()
            ],
            point_maps=archive["point_maps"],
            confidence_maps=archive["confidence_maps"],
            world_from_camera=archive["world_from_camera"],
            raw_confidence_maps=(
                archive["raw_confidence_maps"]
                if "raw_confidence_maps" in archive
                else None
            ),
            valid_masks=(
                archive["valid_masks"]
                if "valid_masks" in archive
                else None
            ),
            schema_version=version,
        )


def load_anchor_poses(
    path: str | Path,
    required_frame_ids: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Load rigid world_from_anchor poses used for viewpoint suppression."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(
            "anchor pose file must contain a non-empty object"
        )
    poses: dict[str, np.ndarray] = {}
    for frame_id, value in payload.items():
        pose = np.asarray(value, dtype=np.float64)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError(
                f"anchor pose for {frame_id} must be a finite 4x4 matrix"
            )
        if not np.allclose(
            pose[3],
            [0.0, 0.0, 0.0, 1.0],
            atol=1e-6,
        ):
            raise ValueError(
                f"anchor pose for {frame_id} is not affine"
            )
        rotation = pose[:3, :3]
        if not np.allclose(
            rotation.T @ rotation,
            np.eye(3),
            atol=1e-4,
        ):
            raise ValueError(
                f"anchor pose for {frame_id} is not rigid"
            )
        if not np.isclose(
            np.linalg.det(rotation),
            1.0,
            atol=1e-4,
        ):
            raise ValueError(
                f"anchor pose for {frame_id} has invalid handedness"
            )
        poses[str(frame_id)] = pose
    if required_frame_ids is not None:
        missing = [
            frame_id
            for frame_id in required_frame_ids
            if frame_id not in poses
        ]
        if missing:
            raise ValueError(
                f"anchor pose file is missing frames: {missing}"
            )
    return poses
