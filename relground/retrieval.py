"""Top-K frame selection with temporal and viewpoint redundancy suppression."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


@dataclass
class FrameCandidate:
    frame_id: str
    score: float
    index: int | None = None
    camera_center: np.ndarray | None = None
    view_direction: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = float(self.score)
        if not self.frame_id or not np.isfinite(self.score):
            raise ValueError("frame_id and a finite score are required")
        if self.camera_center is not None:
            self.camera_center = np.asarray(self.camera_center, dtype=np.float64)
            if self.camera_center.shape != (3,):
                raise ValueError("camera_center must have shape (3,)")
        if self.view_direction is not None:
            direction = np.asarray(self.view_direction, dtype=np.float64)
            if direction.shape != (3,) or np.linalg.norm(direction) < 1e-12:
                raise ValueError("view_direction must be a non-zero length-3 vector")
            self.view_direction = direction / np.linalg.norm(direction)


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 5
    redundancy: str = "hybrid"
    min_frame_gap: int = 3
    min_camera_distance: float = 0.15
    min_view_angle_deg: float = 12.0

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.redundancy not in {"none", "temporal", "viewpoint", "hybrid"}:
            raise ValueError("unsupported redundancy mode")


class TopKFrameRetriever:
    """Ranks precomputed text-frame scores and removes near-duplicate views.

    Text/image encoding remains an upstream adapter concern. Keeping this class
    score-only makes K-selection deterministic and easy to evaluate in isolation.
    """

    def __init__(self, config: RetrievalConfig | None = None) -> None:
        self.config = config or RetrievalConfig()

    def retrieve(self, candidates: Iterable[FrameCandidate]) -> list[FrameCandidate]:
        ordered = sorted(candidates, key=lambda item: (-item.score, item.frame_id))
        selected: list[FrameCandidate] = []
        seen: set[str] = set()
        for candidate in ordered:
            if candidate.frame_id in seen:
                continue
            seen.add(candidate.frame_id)
            if any(self._is_redundant(candidate, kept) for kept in selected):
                continue
            selected.append(candidate)
            if len(selected) == self.config.top_k:
                break
        return selected

    def _is_redundant(self, first: FrameCandidate, second: FrameCandidate) -> bool:
        mode = self.config.redundancy
        if mode == "none":
            return False

        temporal = (
            first.index is not None
            and second.index is not None
            and abs(first.index - second.index) < self.config.min_frame_gap
        )
        viewpoint = False
        if (
            first.camera_center is not None
            and second.camera_center is not None
            and first.view_direction is not None
            and second.view_direction is not None
        ):
            distance = float(np.linalg.norm(first.camera_center - second.camera_center))
            cosine = float(np.clip(np.dot(first.view_direction, second.view_direction), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cosine)))
            viewpoint = (
                distance < self.config.min_camera_distance
                and angle < self.config.min_view_angle_deg
            )

        if mode == "temporal":
            return temporal
        if mode == "viewpoint":
            return viewpoint
        return temporal or viewpoint
