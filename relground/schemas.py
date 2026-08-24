"""Serializable data contracts shared by all project environments.

The contracts deliberately contain only JSON-compatible metadata. Dense point
maps, masks and per-observation point clouds are referenced by path and remain
in NPZ/NPY files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

import numpy as np


SCHEMA_VERSION = "0.1"
OBJECT_OBSERVATION_SCHEMA_VERSION = "1.0"
OBJECT_OBSERVATION_FIELDS = (
    "schema_version",
    "obs_id",
    "class_text",
    "frame_id",
    "mask_ref",
    "retrieval_score",
    "sam_score",
    "valid_point_ratio",
    "points_ref",
    "center",
    "obb",
    "semantic_embedding",
    "metadata",
)
OBJECT_MEMORY_SCHEMA_VERSION = "1.0"
MEMORY_OBJECT_SCHEMA_VERSION = "1.0"
MEMORY_EVIDENCE_FIELDS = (
    "obs_id",
    "frame_id",
    "quality",
)
MEMORY_OBJECT_FIELDS = (
    "schema_version",
    "object_id",
    "class_text",
    "observations",
    "evidence",
    "fused_center",
    "fused_obb",
    "semantic_proto",
    "confidence",
)



def _vector3(value: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite length-3 vector")
    return array


def _matrix3(value: Sequence[Sequence[float]], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    return array


def observation_quality(observation: "ObjectObservation") -> float:
    """Conservative quality used for fusion, bounded to [0, 1]."""

    terms = np.clip(
        [
            observation.retrieval_score,
            observation.sam_score,
            observation.valid_point_ratio,
        ],
        0.0,
        1.0,
    )
    return float(np.prod(terms) ** (1.0 / 3.0))


@dataclass
class OrientedBoundingBox:
    center: np.ndarray
    extent: np.ndarray
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self) -> None:
        self.center = _vector3(self.center, "obb.center")
        self.extent = _vector3(self.extent, "obb.extent")
        self.rotation = _matrix3(self.rotation, "obb.rotation")
        if np.any(self.extent < 0.0):
            raise ValueError("obb.extent must be non-negative")

    def corners(self) -> np.ndarray:
        signs = np.array(
            [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
        )
        local = signs * (self.extent / 2.0)
        return local @ self.rotation.T + self.center

    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        corners = self.corners()
        return corners.min(axis=0), corners.max(axis=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "center": self.center.tolist(),
            "extent": self.extent.tolist(),
            "rotation": self.rotation.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OrientedBoundingBox":
        return cls(data["center"], data["extent"], data.get("rotation", np.eye(3)))


@dataclass
class ObjectObservation:
    obs_id: str
    class_text: str
    frame_id: str
    mask_ref: str | None
    retrieval_score: float
    sam_score: float
    valid_point_ratio: float
    points_ref: str | None
    center: np.ndarray
    obb: OrientedBoundingBox
    semantic_embedding: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = OBJECT_OBSERVATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OBJECT_OBSERVATION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported ObjectObservation schema: {self.schema_version}"
            )
        if not self.obs_id or not self.frame_id or not self.class_text.strip():
            raise ValueError("obs_id, frame_id and class_text are required")
        self.class_text = self.class_text.strip()
        self.center = _vector3(self.center, "observation.center")
        for name in ("retrieval_score", "sam_score", "valid_point_ratio"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            setattr(self, name, value)
        if self.semantic_embedding is not None:
            embedding = np.asarray(self.semantic_embedding, dtype=np.float64)
            if embedding.ndim != 1 or embedding.size == 0 or not np.all(np.isfinite(embedding)):
                raise ValueError("semantic_embedding must be a finite vector")
            self.semantic_embedding = embedding

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "obs_id": self.obs_id,
            "class_text": self.class_text,
            "frame_id": self.frame_id,
            "mask_ref": self.mask_ref,
            "retrieval_score": self.retrieval_score,
            "sam_score": self.sam_score,
            "valid_point_ratio": self.valid_point_ratio,
            "points_ref": self.points_ref,
            "center": self.center.tolist(),
            "obb": self.obb.to_dict(),
            "semantic_embedding": None
            if self.semantic_embedding is None
            else self.semantic_embedding.tolist(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObjectObservation":
        return cls(
            obs_id=str(data["obs_id"]),
            class_text=str(data["class_text"]),
            frame_id=str(data["frame_id"]),
            mask_ref=data.get("mask_ref"),
            retrieval_score=float(data["retrieval_score"]),
            sam_score=float(data["sam_score"]),
            valid_point_ratio=float(data["valid_point_ratio"]),
            points_ref=data.get("points_ref"),
            center=data["center"],
            obb=OrientedBoundingBox.from_dict(data["obb"]),
            semantic_embedding=data.get("semantic_embedding"),
            metadata=dict(data.get("metadata", {})),
            schema_version=str(
                data.get("schema_version", OBJECT_OBSERVATION_SCHEMA_VERSION)
            ),
        )


@dataclass
class MemoryObject:
    object_id: str
    class_text: str
    observations: list[ObjectObservation]
    fused_center: np.ndarray
    fused_obb: OrientedBoundingBox
    semantic_proto: np.ndarray | None
    confidence: float
    schema_version: str = MEMORY_OBJECT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MEMORY_OBJECT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported MemoryObject schema: {self.schema_version}"
            )
        self.object_id = str(self.object_id).strip()
        self.class_text = str(self.class_text).strip()
        if not self.object_id or not self.class_text:
            raise ValueError("object_id and class_text are required")
        if not self.observations:
            raise ValueError("MemoryObject requires observation evidence")
        observation_ids = [item.obs_id for item in self.observations]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("MemoryObject observation ids must be unique")
        self.fused_center = _vector3(
            self.fused_center, "memory.fused_center"
        )
        self.confidence = float(self.confidence)
        if not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("memory confidence must be finite in [0, 1]")
        if self.semantic_proto is not None:
            prototype = np.asarray(self.semantic_proto, dtype=np.float64)
            if (
                prototype.ndim != 1
                or prototype.size == 0
                or not np.all(np.isfinite(prototype))
            ):
                raise ValueError("semantic_proto must be a finite vector")
            self.semantic_proto = prototype

    @property
    def evidence_frames(self) -> list[str]:
        return list(
            dict.fromkeys(item.frame_id for item in self.observations)
        )

    @property
    def evidence(self) -> list[dict[str, Any]]:
        return [
            {
                "obs_id": item.obs_id,
                "frame_id": item.frame_id,
                "quality": observation_quality(item),
            }
            for item in self.observations
        ]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "object_id": self.object_id,
            "class_text": self.class_text,
            "observations": [item.to_dict() for item in self.observations],
            "evidence": self.evidence,
            "fused_center": self.fused_center.tolist(),
            "fused_obb": self.fused_obb.to_dict(),
            "semantic_proto": (
                None
                if self.semantic_proto is None
                else self.semantic_proto.tolist()
            ),
            "confidence": self.confidence,
        }
        if tuple(payload) != MEMORY_OBJECT_FIELDS:
            raise AssertionError("MemoryObject serialization fields changed")
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryObject":
        actual, expected = set(data), set(MEMORY_OBJECT_FIELDS)
        if actual != expected:
            raise ValueError(
                "MemoryObject fields differ from frozen schema: "
                f"missing={sorted(expected - actual)} "
                f"unexpected={sorted(actual - expected)}"
            )
        raw_observations = data["observations"]
        if not isinstance(raw_observations, list):
            raise ValueError("MemoryObject observations must be a list")
        for index, raw in enumerate(raw_observations):
            if (
                not isinstance(raw, Mapping)
                or set(raw) != set(OBJECT_OBSERVATION_FIELDS)
            ):
                raise ValueError(
                    f"MemoryObject observation {index} fields are not frozen"
                )
        observations = [
            ObjectObservation.from_dict(item) for item in raw_observations
        ]
        memory_object = cls(
            object_id=str(data["object_id"]),
            class_text=str(data["class_text"]),
            observations=observations,
            fused_center=data["fused_center"],
            fused_obb=OrientedBoundingBox.from_dict(data["fused_obb"]),
            semantic_proto=data["semantic_proto"],
            confidence=float(data["confidence"]),
            schema_version=str(data["schema_version"]),
        )
        raw_evidence = data["evidence"]
        expected_evidence = memory_object.evidence
        if (
            not isinstance(raw_evidence, list)
            or len(raw_evidence) != len(expected_evidence)
        ):
            raise ValueError("MemoryObject evidence count is inconsistent")
        for index, (actual_row, expected_row) in enumerate(
            zip(raw_evidence, expected_evidence)
        ):
            if (
                not isinstance(actual_row, Mapping)
                or set(actual_row) != set(MEMORY_EVIDENCE_FIELDS)
                or actual_row["obs_id"] != expected_row["obs_id"]
                or actual_row["frame_id"] != expected_row["frame_id"]
                or not np.isclose(
                    float(actual_row["quality"]),
                    expected_row["quality"],
                    atol=1e-12,
                )
            ):
                raise ValueError(
                    f"MemoryObject evidence {index} is inconsistent"
                )
        return memory_object



@dataclass(frozen=True)
class GroundingQuery:
    query_id: str
    target: str
    relation: str | None = None
    reference: str | None = None
    anchor_frame: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GroundingQuery":
        return cls(
            query_id=str(data.get("query_id", "query")),
            target=str(data["target"]),
            relation=data.get("relation"),
            reference=data.get("reference"),
            anchor_frame=data.get("anchor_frame"),
        )


@dataclass
class GroundingResult:
    query_id: str
    ranked_ids: list[str]
    relation_scores: dict[str, float]
    confidence: float
    evidence_frames: list[str]
    abstain: bool
    reason: str | None
    explanation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "ranked_ids": self.ranked_ids,
            "relation_scores": self.relation_scores,
            "confidence": self.confidence,
            "evidence_frames": self.evidence_frames,
            "abstain": self.abstain,
            "reason": self.reason,
            "explanation": self.explanation,
        }


@dataclass
class RunManifest:
    git_sha: str
    env_lock: str
    dataset_split: str
    seed: int
    config: dict[str, Any]
    command: str
    runtime_seconds: float | None = None
    peak_vram_mb: float | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n")
