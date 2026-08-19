"""NPZ contract exported by the isolated ``vggt_geom`` environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


GEOMETRY_SCHEMA_VERSION = "0.1"


@dataclass(frozen=True)
class GeometryFrame:
    frame_id: str
    point_map: np.ndarray
    confidence_map: np.ndarray
    world_from_camera: np.ndarray


@dataclass
class GeometryBundle:
    frame_ids: list[str]
    point_maps: np.ndarray
    confidence_maps: np.ndarray
    world_from_camera: np.ndarray
    schema_version: str = GEOMETRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        count = len(self.frame_ids)
        self.point_maps = np.asarray(self.point_maps)
        self.confidence_maps = np.asarray(self.confidence_maps)
        self.world_from_camera = np.asarray(self.world_from_camera)
        if self.point_maps.ndim != 4 or self.point_maps.shape[0] != count or self.point_maps.shape[-1] != 3:
            raise ValueError("point_maps must have shape (N, H, W, 3)")
        if self.confidence_maps.shape != self.point_maps.shape[:-1]:
            raise ValueError("confidence_maps must have shape (N, H, W)")
        if self.world_from_camera.shape != (count, 4, 4):
            raise ValueError("world_from_camera must have shape (N, 4, 4)")
        if len(set(self.frame_ids)) != count:
            raise ValueError("frame_ids must be unique")

    def get(self, frame_id: str) -> GeometryFrame:
        try:
            index = self.frame_ids.index(frame_id)
        except ValueError as error:
            raise KeyError(f"unknown geometry frame: {frame_id}") from error
        return GeometryFrame(
            frame_id=frame_id,
            point_map=self.point_maps[index],
            confidence_map=self.confidence_maps[index],
            world_from_camera=self.world_from_camera[index],
        )


def save_geometry_npz(
    path: str | Path,
    *,
    frame_ids: Sequence[str],
    point_maps: np.ndarray,
    confidence_maps: np.ndarray,
    world_from_camera: np.ndarray,
) -> None:
    bundle = GeometryBundle(
        frame_ids=list(frame_ids),
        point_maps=point_maps,
        confidence_maps=confidence_maps,
        world_from_camera=world_from_camera,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema_version=np.asarray(bundle.schema_version),
        frame_ids=np.asarray(bundle.frame_ids, dtype=np.str_),
        point_maps=bundle.point_maps,
        confidence_maps=bundle.confidence_maps,
        world_from_camera=bundle.world_from_camera,
    )


def load_geometry_npz(path: str | Path) -> GeometryBundle:
    with np.load(path, allow_pickle=False) as archive:
        required = {"frame_ids", "point_maps", "confidence_maps", "world_from_camera"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"geometry archive missing fields: {sorted(missing)}")
        version = str(archive["schema_version"].item()) if "schema_version" in archive else "unknown"
        return GeometryBundle(
            frame_ids=[str(item) for item in archive["frame_ids"].tolist()],
            point_maps=archive["point_maps"],
            confidence_maps=archive["confidence_maps"],
            world_from_camera=archive["world_from_camera"],
            schema_version=version,
        )
