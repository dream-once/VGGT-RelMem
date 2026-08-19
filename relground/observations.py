"""Mask-to-3D lifting and ObjectObservation construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schemas import ObjectObservation, OrientedBoundingBox


class LiftingError(ValueError):
    """Raised when a mask does not contain enough trustworthy 3D evidence."""


@dataclass(frozen=True)
class LifterConfig:
    confidence_threshold: float = 0.25
    min_points: int = 30
    outlier_mad_scale: float = 3.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if self.min_points < 3:
            raise ValueError("min_points must be at least 3")
        if self.outlier_mad_scale < 0.0:
            raise ValueError("outlier_mad_scale must be non-negative")


@dataclass
class LiftedGeometry:
    points: np.ndarray
    center: np.ndarray
    obb: OrientedBoundingBox
    valid_point_ratio: float


class Robust3DLifter:
    def __init__(self, config: LifterConfig | None = None) -> None:
        self.config = config or LifterConfig()

    def lift(
        self,
        mask: np.ndarray,
        point_map: np.ndarray,
        confidence_map: np.ndarray | None = None,
        world_from_camera: np.ndarray | None = None,
    ) -> LiftedGeometry:
        mask_array = np.asarray(mask, dtype=bool)
        points_array = np.asarray(point_map, dtype=np.float64)
        if points_array.shape != mask_array.shape + (3,):
            raise LiftingError("point_map must have shape mask.shape + (3,)")
        masked_count = int(mask_array.sum())
        if masked_count == 0:
            raise LiftingError("empty mask")

        valid = mask_array & np.all(np.isfinite(points_array), axis=-1)
        if confidence_map is not None:
            confidence = np.asarray(confidence_map, dtype=np.float64)
            if confidence.shape != mask_array.shape:
                raise LiftingError("confidence_map must match mask shape")
            valid &= np.isfinite(confidence) & (confidence >= self.config.confidence_threshold)

        points = points_array[valid]
        if world_from_camera is not None:
            transform = np.asarray(world_from_camera, dtype=np.float64)
            if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
                raise LiftingError("world_from_camera must be a finite 4x4 matrix")
            homogeneous = np.column_stack([points, np.ones(len(points))])
            transformed = homogeneous @ transform.T
            safe_w = transformed[:, 3]
            if np.any(np.abs(safe_w) < 1e-12):
                raise LiftingError("world transform produced points at infinity")
            points = transformed[:, :3] / safe_w[:, None]

        points = self._remove_outliers(points)
        valid_ratio = float(len(points) / masked_count)
        if len(points) < self.config.min_points:
            raise LiftingError(
                f"too few valid points: {len(points)} < {self.config.min_points} "
                f"(ratio={valid_ratio:.3f})"
            )
        obb = self._fit_obb(points)
        return LiftedGeometry(points=points, center=obb.center.copy(), obb=obb, valid_point_ratio=valid_ratio)

    def make_observation(
        self,
        *,
        obs_id: str,
        class_text: str,
        frame_id: str,
        mask: np.ndarray,
        point_map: np.ndarray,
        retrieval_score: float,
        sam_score: float,
        confidence_map: np.ndarray | None = None,
        world_from_camera: np.ndarray | None = None,
        mask_ref: str | None = None,
        points_ref: str | None = None,
        semantic_embedding: np.ndarray | None = None,
    ) -> tuple[ObjectObservation, np.ndarray]:
        lifted = self.lift(mask, point_map, confidence_map, world_from_camera)
        observation = ObjectObservation(
            obs_id=obs_id,
            class_text=class_text,
            frame_id=frame_id,
            mask_ref=mask_ref,
            retrieval_score=retrieval_score,
            sam_score=sam_score,
            valid_point_ratio=lifted.valid_point_ratio,
            points_ref=points_ref,
            center=lifted.center,
            obb=lifted.obb,
            semantic_embedding=semantic_embedding,
        )
        return observation, lifted.points

    def _remove_outliers(self, points: np.ndarray) -> np.ndarray:
        if len(points) < self.config.min_points or self.config.outlier_mad_scale == 0.0:
            return points
        median_point = np.median(points, axis=0)
        distances = np.linalg.norm(points - median_point, axis=1)
        median_distance = float(np.median(distances))
        mad = float(np.median(np.abs(distances - median_distance)))
        if mad < 1e-12:
            return points
        limit = median_distance + self.config.outlier_mad_scale * 1.4826 * mad
        return points[distances <= limit]

    @staticmethod
    def _fit_obb(points: np.ndarray) -> OrientedBoundingBox:
        mean = points.mean(axis=0)
        centered = points - mean
        covariance = centered.T @ centered / max(len(points) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        rotation = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
        for axis in range(3):
            dominant = int(np.argmax(np.abs(rotation[:, axis])))
            if rotation[dominant, axis] < 0.0:
                rotation[:, axis] *= -1.0
        if np.linalg.det(rotation) < 0.0:
            rotation[:, -1] *= -1.0
        local = points @ rotation
        lower = local.min(axis=0)
        upper = local.max(axis=0)
        center = ((lower + upper) / 2.0) @ rotation.T
        return OrientedBoundingBox(center=center, extent=upper - lower, rotation=rotation)
