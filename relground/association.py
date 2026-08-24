"""Explainable cross-frame association and persistent object memory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator
import json
import re

import numpy as np

from .schemas import (
    MEMORY_OBJECT_SCHEMA_VERSION,
    OBJECT_MEMORY_SCHEMA_VERSION,
    OBJECT_OBSERVATION_FIELDS,
    OBJECT_OBSERVATION_SCHEMA_VERSION,
    MemoryObject,
    ObjectObservation,
    OrientedBoundingBox,
    observation_quality,
)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower().replace("_", " ")))


def text_similarity(first: str, second: str) -> float:
    first_tokens, second_tokens = _tokens(first), _tokens(second)
    if not first_tokens or not second_tokens:
        return 0.0
    return len(first_tokens & second_tokens) / len(first_tokens | second_tokens)


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        return 0.0
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator < 1e-12:
        return 0.0
    return float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))


def aabb_iou(first: OrientedBoundingBox, second: OrientedBoundingBox) -> float:
    first_min, first_max = first.aabb()
    second_min, second_max = second.aabb()
    overlap = np.maximum(0.0, np.minimum(first_max, second_max) - np.maximum(first_min, second_min))
    intersection = float(np.prod(overlap))
    first_volume = float(np.prod(np.maximum(0.0, first_max - first_min)))
    second_volume = float(np.prod(np.maximum(0.0, second_max - second_min)))
    union = first_volume + second_volume - intersection
    return 0.0 if union <= 1e-12 else intersection / union


@dataclass(frozen=True)
class AssociationConfig:
    distance_threshold: float = 0.75
    semantic_threshold: float = 0.70
    min_match_score: float = 0.45
    distance_weight: float = 0.55
    semantic_weight: float = 0.30
    overlap_weight: float = 0.15

    def __post_init__(self) -> None:
        if self.distance_threshold <= 0.0:
            raise ValueError("distance_threshold must be positive")
        if not 0.0 <= self.semantic_threshold <= 1.0:
            raise ValueError("semantic_threshold must be in [0, 1]")
        weight_sum = self.distance_weight + self.semantic_weight + self.overlap_weight
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError("association weights must sum to 1")


@dataclass(frozen=True)
class AssociationDecision:
    obs_id: str
    object_id: str
    created: bool
    score: float
    margin: float
    distance: float | None
    semantic_similarity: float
    overlap_iou: float


OBJECT_MEMORY_FIELDS = (
    "schema_version",
    "observation_schema_version",
    "memory_object_schema_version",
    "association_config",
    "pending_observations",
    "objects",
    "decisions",
    "evidence",
    "metadata",
)
OBJECT_MEMORY_EVIDENCE_FIELDS = (
    "pending_observation_ids",
    "associated_observation_ids",
    "frame_ids",
    "object_observation_ids",
)


class ObjectMemory:
    def __init__(
        self,
        config: AssociationConfig | None = None,
        schema_version: str = OBJECT_MEMORY_SCHEMA_VERSION,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if schema_version != OBJECT_MEMORY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported ObjectMemory schema: {schema_version}"
            )
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("ObjectMemory metadata must be an object")
        self.config = config or AssociationConfig()
        self.schema_version = schema_version
        self.metadata = dict(metadata or {})
        self.pending_observations: dict[str, ObjectObservation] = {}
        self.objects: dict[str, MemoryObject] = {}
        self.decisions: list[AssociationDecision] = []
        self._next_id = 1

    @property
    def version(self) -> str:
        """Compatibility alias for the frozen schema version."""

        return self.schema_version

    def __len__(self) -> int:
        return len(self.objects)

    def __iter__(self) -> Iterator[MemoryObject]:
        return iter(self.objects.values())

    def get(self, object_id: str) -> MemoryObject:
        return self.objects[object_id]

    def _associated_observation_ids(self) -> set[str]:
        return {
            observation.obs_id
            for memory_object in self.objects.values()
            for observation in memory_object.observations
        }

    def stage_many(
        self,
        observations: Iterable[ObjectObservation],
    ) -> list[str]:
        staged: list[str] = []
        associated_ids = self._associated_observation_ids()
        for observation in observations:
            if (
                observation.obs_id in self.pending_observations
                or observation.obs_id in associated_ids
            ):
                raise ValueError(
                    f"duplicate observation id: {observation.obs_id}"
                )
            self.pending_observations[observation.obs_id] = observation
            staged.append(observation.obs_id)
        return staged

    def add_many(
        self,
        observations: Iterable[ObjectObservation],
    ) -> list[AssociationDecision]:
        return [self.add_observation(item) for item in observations]

    def add_observation(self, observation: ObjectObservation) -> AssociationDecision:
        associated_ids = self._associated_observation_ids()
        if observation.obs_id in associated_ids:
            raise ValueError(
                f"duplicate observation id: {observation.obs_id}"
            )
        staged = self.pending_observations.get(observation.obs_id)
        if staged is not None and staged.to_dict() != observation.to_dict():
            raise ValueError(
                f"staged observation changed: {observation.obs_id}"
            )

        matches: list[tuple[float, float, float, float, MemoryObject]] = []
        for memory_object in self.objects.values():
            semantic = self._semantic_similarity(observation, memory_object)
            distance = float(np.linalg.norm(observation.center - memory_object.fused_center))
            overlap = aabb_iou(observation.obb, memory_object.fused_obb)
            if semantic < self.config.semantic_threshold:
                continue
            if distance > self.config.distance_threshold and overlap <= 0.0:
                continue
            distance_score = max(0.0, 1.0 - distance / self.config.distance_threshold)
            score = (
                self.config.distance_weight * distance_score
                + self.config.semantic_weight * max(0.0, semantic)
                + self.config.overlap_weight * overlap
            )
            if score >= self.config.min_match_score:
                matches.append((score, distance, semantic, overlap, memory_object))

        matches.sort(key=lambda item: (-item[0], item[4].object_id))
        if not matches:
            object_id = f"obj_{self._next_id:04d}"
            self._next_id += 1
            self.objects[object_id] = self._fuse(object_id, [observation])
            decision = AssociationDecision(
                obs_id=observation.obs_id,
                object_id=object_id,
                created=True,
                score=0.0,
                margin=1.0,
                distance=None,
                semantic_similarity=1.0,
                overlap_iou=0.0,
            )
        else:
            score, distance, semantic, overlap, memory_object = matches[0]
            runner_up = matches[1][0] if len(matches) > 1 else 0.0
            observations = [*memory_object.observations, observation]
            self.objects[memory_object.object_id] = self._fuse(memory_object.object_id, observations)
            decision = AssociationDecision(
                obs_id=observation.obs_id,
                object_id=memory_object.object_id,
                created=False,
                score=score,
                margin=score - runner_up,
                distance=distance,
                semantic_similarity=semantic,
                overlap_iou=overlap,
            )
        self.pending_observations.pop(observation.obs_id, None)
        self.decisions.append(decision)
        return decision

    @staticmethod
    def _semantic_similarity(observation: ObjectObservation, memory_object: MemoryObject) -> float:
        if observation.semantic_embedding is not None and memory_object.semantic_proto is not None:
            return max(0.0, cosine_similarity(observation.semantic_embedding, memory_object.semantic_proto))
        return text_similarity(observation.class_text, memory_object.class_text)

    @staticmethod
    def _fuse(object_id: str, observations: list[ObjectObservation]) -> MemoryObject:
        qualities = np.array([max(observation_quality(item), 1e-3) for item in observations])
        centers = np.stack([item.center for item in observations])
        fused_center = np.average(centers, axis=0, weights=qualities)

        corners = np.concatenate([item.obb.corners() for item in observations], axis=0)
        lower, upper = corners.min(axis=0), corners.max(axis=0)
        fused_obb = OrientedBoundingBox(
            center=(lower + upper) / 2.0,
            extent=upper - lower,
            rotation=np.eye(3),
        )

        embeddings = [item.semantic_embedding for item in observations]
        semantic_proto = None
        if all(item is not None for item in embeddings):
            dimensions = {item.shape for item in embeddings if item is not None}
            if len(dimensions) == 1:
                semantic_proto = np.average(np.stack(embeddings), axis=0, weights=qualities)
                norm = np.linalg.norm(semantic_proto)
                if norm > 1e-12:
                    semantic_proto /= norm

        class_weights: dict[str, float] = {}
        for observation, quality in zip(observations, qualities):
            class_weights[observation.class_text] = class_weights.get(observation.class_text, 0.0) + quality
        class_text = min(class_weights, key=lambda text: (-class_weights[text], text))
        confidence = float(1.0 - np.prod(1.0 - np.clip(qualities, 0.0, 0.999)))
        return MemoryObject(
            object_id=object_id,
            class_text=class_text,
            observations=observations,
            fused_center=fused_center,
            fused_obb=fused_obb,
            semantic_proto=semantic_proto,
            confidence=confidence,
        )

    def _evidence(self) -> dict[str, Any]:
        pending = list(self.pending_observations.values())
        associated = [
            observation
            for memory_object in self.objects.values()
            for observation in memory_object.observations
        ]
        combined = [*pending, *associated]
        return {
            "pending_observation_ids": [
                observation.obs_id for observation in pending
            ],
            "associated_observation_ids": [
                observation.obs_id for observation in associated
            ],
            "frame_ids": list(
                dict.fromkeys(item.frame_id for item in combined)
            ),
            "object_observation_ids": {
                object_id: [item.obs_id for item in memory_object.observations]
                for object_id, memory_object in self.objects.items()
            },
        }

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "observation_schema_version": OBJECT_OBSERVATION_SCHEMA_VERSION,
            "memory_object_schema_version": MEMORY_OBJECT_SCHEMA_VERSION,
            "association_config": asdict(self.config),
            "pending_observations": [
                item.to_dict()
                for item in self.pending_observations.values()
            ],
            "objects": [item.to_dict() for item in self.objects.values()],
            "decisions": [asdict(item) for item in self.decisions],
            "evidence": self._evidence(),
            "metadata": self.metadata,
        }
        if tuple(payload) != OBJECT_MEMORY_FIELDS:
            raise AssertionError("ObjectMemory serialization fields changed")
        return payload

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)

    @classmethod
    def load(
        cls,
        path: str | Path,
        config: AssociationConfig | None = None,
    ) -> "ObjectMemory":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("ObjectMemory root must be an object")
        actual, expected = set(payload), set(OBJECT_MEMORY_FIELDS)
        if actual != expected:
            raise ValueError(
                "ObjectMemory fields differ from frozen schema: "
                f"missing={sorted(expected - actual)} "
                f"unexpected={sorted(actual - expected)}"
            )
        if payload["observation_schema_version"] != OBJECT_OBSERVATION_SCHEMA_VERSION:
            raise ValueError("unsupported observation schema in memory")
        if payload["memory_object_schema_version"] != MEMORY_OBJECT_SCHEMA_VERSION:
            raise ValueError("unsupported object schema in memory")
        stored_config = AssociationConfig(**payload["association_config"])
        memory = cls(
            config=config or stored_config,
            schema_version=str(payload["schema_version"]),
            metadata=dict(payload["metadata"]),
        )
        pending = payload["pending_observations"]
        if not isinstance(pending, list):
            raise ValueError("pending_observations must be a list")
        for index, raw in enumerate(pending):
            if not isinstance(raw, dict) or set(raw) != set(OBJECT_OBSERVATION_FIELDS):
                raise ValueError(
                    f"pending observation {index} fields are not frozen"
                )
        memory.stage_many(ObjectObservation.from_dict(raw) for raw in pending)

        raw_objects = payload["objects"]
        if not isinstance(raw_objects, list):
            raise ValueError("objects must be a list")
        for raw in raw_objects:
            item = MemoryObject.from_dict(raw)
            if item.object_id in memory.objects:
                raise ValueError(f"duplicate object id: {item.object_id}")
            memory.objects[item.object_id] = item
        pending_ids = set(memory.pending_observations)
        associated_ids = memory._associated_observation_ids()
        associated_count = sum(
            len(item.observations) for item in memory.objects.values()
        )
        if pending_ids & associated_ids:
            raise ValueError("pending and associated observations overlap")
        if len(associated_ids) != associated_count:
            raise ValueError("associated observation ids must be globally unique")

        decisions = payload["decisions"]
        if not isinstance(decisions, list):
            raise ValueError("decisions must be a list")
        memory.decisions = [AssociationDecision(**item) for item in decisions]
        for decision in memory.decisions:
            obj = memory.objects.get(decision.object_id)
            object_ids = (
                {item.obs_id for item in obj.observations}
                if obj is not None
                else set()
            )
            if decision.obs_id not in object_ids:
                raise ValueError(
                    "association decision has inconsistent evidence"
                )

        evidence = payload["evidence"]
        if (
            not isinstance(evidence, dict)
            or set(evidence) != set(OBJECT_MEMORY_EVIDENCE_FIELDS)
        ):
            raise ValueError("ObjectMemory evidence fields are not frozen")
        if evidence != memory._evidence():
            raise ValueError("ObjectMemory evidence is inconsistent")
        used_ids = [
            int(match.group(1))
            for key in memory.objects
            if (match := re.fullmatch(r"obj_(\d+)", key))
        ]
        memory._next_id = max(used_ids, default=0) + 1
        return memory
