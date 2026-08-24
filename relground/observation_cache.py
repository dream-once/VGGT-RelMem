"""Frozen D7 cache contract for multi-frame ObjectObservation artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import json

from .schemas import (
    OBJECT_OBSERVATION_FIELDS,
    OBJECT_OBSERVATION_SCHEMA_VERSION,
    ObjectObservation,
)


SCENE_OBSERVATION_CACHE_VERSION = "1.0"
SCENE_OBSERVATION_CACHE_FIELDS = (
    "schema_version",
    "observation_schema_version",
    "scene_id",
    "query",
    "source_stage",
    "frame_ids",
    "observations",
    "metadata",
)


@dataclass
class SceneObservationCache:
    scene_id: str
    query: str
    frame_ids: list[str]
    observations: list[ObjectObservation]
    source_stage: str = "D6"
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCENE_OBSERVATION_CACHE_VERSION
    observation_schema_version: str = OBJECT_OBSERVATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.scene_id = str(self.scene_id).strip()
        self.query = str(self.query).strip()
        self.source_stage = str(self.source_stage).strip()
        self.frame_ids = [str(frame_id) for frame_id in self.frame_ids]
        if self.schema_version != SCENE_OBSERVATION_CACHE_VERSION:
            raise ValueError(
                f"unsupported scene cache schema: {self.schema_version}"
            )
        if self.observation_schema_version != OBJECT_OBSERVATION_SCHEMA_VERSION:
            raise ValueError(
                "scene cache uses an unsupported ObjectObservation schema"
            )
        if not self.scene_id or not self.query or not self.source_stage:
            raise ValueError("scene_id, query and source_stage are required")
        if len(self.frame_ids) < 2 or len(set(self.frame_ids)) != len(
            self.frame_ids
        ):
            raise ValueError("scene cache requires at least two unique frame_ids")
        if not self.observations:
            raise ValueError("scene cache requires at least one observation")
        observation_ids = [observation.obs_id for observation in self.observations]
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError("scene cache observation ids must be unique")
        unknown_frames = sorted(
            {
                observation.frame_id
                for observation in self.observations
                if observation.frame_id not in self.frame_ids
            }
        )
        if unknown_frames:
            raise ValueError(
                f"observations use frames absent from cache: {unknown_frames}"
            )
        observed_frames = {
            observation.frame_id for observation in self.observations
        }
        if len(observed_frames) < 2:
            raise ValueError(
                "scene cache requires valid observations from at least two frames"
            )
        mismatched_queries = [
            observation.obs_id
            for observation in self.observations
            if observation.class_text != self.query
        ]
        if mismatched_queries:
            raise ValueError(
                f"observation query mismatch: {mismatched_queries}"
            )
        if not isinstance(self.metadata, dict):
            raise ValueError("scene cache metadata must be an object")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "observation_schema_version": self.observation_schema_version,
            "scene_id": self.scene_id,
            "query": self.query,
            "source_stage": self.source_stage,
            "frame_ids": self.frame_ids,
            "observations": [
                observation.to_dict() for observation in self.observations
            ],
            "metadata": self.metadata,
        }
        if tuple(payload) != SCENE_OBSERVATION_CACHE_FIELDS:
            raise AssertionError("scene cache serialization field order changed")
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SceneObservationCache":
        actual_fields = set(data)
        expected_fields = set(SCENE_OBSERVATION_CACHE_FIELDS)
        if actual_fields != expected_fields:
            raise ValueError(
                "scene cache fields differ from frozen schema: "
                f"missing={sorted(expected_fields - actual_fields)} "
                f"unexpected={sorted(actual_fields - expected_fields)}"
            )
        raw_observations = data["observations"]
        if not isinstance(raw_observations, list):
            raise ValueError("scene cache observations must be a list")
        for index, raw in enumerate(raw_observations):
            if not isinstance(raw, Mapping):
                raise ValueError(f"observation {index} must be an object")
            actual = set(raw)
            expected = set(OBJECT_OBSERVATION_FIELDS)
            if actual != expected:
                raise ValueError(
                    f"observation {index} fields differ from frozen schema: "
                    f"missing={sorted(expected - actual)} "
                    f"unexpected={sorted(actual - expected)}"
                )
        return cls(
            scene_id=str(data["scene_id"]),
            query=str(data["query"]),
            source_stage=str(data["source_stage"]),
            frame_ids=[str(value) for value in data["frame_ids"]],
            observations=[
                ObjectObservation.from_dict(value)
                for value in raw_observations
            ],
            metadata=dict(data["metadata"]),
            schema_version=str(data["schema_version"]),
            observation_schema_version=str(
                data["observation_schema_version"]
            ),
        )


def save_observation_cache(
    path: str | Path,
    cache: SceneObservationCache,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(cache.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_observation_cache(path: str | Path) -> SceneObservationCache:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("scene cache root must be an object")
    return SceneObservationCache.from_dict(payload)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_inventory(
    root: str | Path,
    references: Sequence[str],
) -> list[dict[str, Any]]:
    base = Path(root).resolve()
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in sorted(str(reference) for reference in references):
        relative = Path(value)
        if (
            not value
            or relative.is_absolute()
            or ".." in relative.parts
            or value in seen
        ):
            raise ValueError(f"invalid or duplicate artifact reference: {value}")
        path = (base / relative).resolve()
        if base not in path.parents:
            raise ValueError(f"artifact escapes scene cache: {value}")
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing or empty cache artifact: {value}")
        seen.add(value)
        records.append(
            {
                "path": value,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records
