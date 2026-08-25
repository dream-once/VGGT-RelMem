"""D9 exact-class spatial gating and labelled pairwise evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import re

import numpy as np

from .association import AssociationConfig, ObjectMemory, aabb_iou
from .schemas import ObjectObservation


D9_SCHEMA_VERSION = "0.1"
D9_BASELINE_ID = "B2-topk-multiframe+D9-exact-class-spatial-gate"
D9_LABEL_SCHEMA_VERSION = "0.1"
D9_LABEL_FIELDS = (
    "schema_version",
    "scene_id",
    "query",
    "annotation_method",
    "notes",
    "instance_groups",
)
D9_INSTANCE_GROUP_FIELDS = ("instance_id", "observation_ids")
D9_GATE_CONFIG_FIELDS = (
    "same_class_rule",
    "center_distance_threshold",
    "overlap_measure",
    "min_overlap_iou",
    "spatial_rule",
    "min_distinct_frames",
)
D9_PAIR_FIELDS = (
    "obs_id_a",
    "obs_id_b",
    "expected_same",
    "same_class",
    "center_distance",
    "center_distance_pass",
    "overlap_iou",
    "overlap_pass",
    "predicted_same",
    "gate_reasons",
    "error_type",
)
D9_COMPONENT_FIELDS = (
    "component_id",
    "observation_ids",
    "frame_ids",
    "distinct_frame_count",
    "promoted",
    "object_id",
    "deferred_reason",
)
D9_METRIC_FIELDS = (
    "pair_count",
    "positive_pairs",
    "negative_pairs",
    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",
    "precision",
    "recall",
    "f1",
    "accuracy",
)
D9_RESULT_FIELDS = (
    "schema_version",
    "status",
    "stage",
    "baseline_id",
    "scene_id",
    "query",
    "source",
    "gate_config",
    "counts",
    "metrics",
    "components",
    "pairs",
    "failure_cases",
    "acceptance",
    "artifacts",
    "created_at",
)
D9_SOURCE_FIELDS = (
    "d8_memory",
    "d8_memory_sha256",
    "pair_labels",
    "pair_labels_sha256",
)
D9_COUNT_FIELDS = (
    "input_observations",
    "pair_count",
    "predicted_match_pairs",
    "candidate_components",
    "promoted_components",
    "deferred_components",
    "permanent_objects",
    "pending_observations",
    "association_decisions",
)
D9_ACCEPTANCE_FIELDS = (
    "min_pairwise_f1",
    "pairwise_f1_pass",
    "cross_frame_object_pass",
    "round_trip_equal",
)
D9_ARTIFACT_FIELDS = ("object_memory", "pair_labels")


def normalize_class_text(text: str) -> str:
    """Normalize separators/case without introducing semantic similarity."""

    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


@dataclass(frozen=True)
class SpatialGateConfig:
    center_distance_threshold: float = 0.15
    min_overlap_iou: float = 0.0
    min_distinct_frames: int = 2

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.center_distance_threshold)
            or self.center_distance_threshold <= 0.0
        ):
            raise ValueError("center_distance_threshold must be positive")
        if (
            not np.isfinite(self.min_overlap_iou)
            or not 0.0 <= self.min_overlap_iou <= 1.0
        ):
            raise ValueError("min_overlap_iou must be in [0, 1]")
        if self.min_distinct_frames < 2:
            raise ValueError("min_distinct_frames must be at least 2")

    def to_dict(self) -> dict[str, Any]:
        return {
            "same_class_rule": "normalized_exact",
            "center_distance_threshold": self.center_distance_threshold,
            "overlap_measure": "aabb_iou_from_observation_obb",
            "min_overlap_iou": self.min_overlap_iou,
            "spatial_rule": "center_distance_pass OR overlap_pass",
            "min_distinct_frames": self.min_distinct_frames,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SpatialGateConfig":
        if set(payload) != set(D9_GATE_CONFIG_FIELDS):
            raise ValueError("D9 gate config fields are not frozen")
        if payload["same_class_rule"] != "normalized_exact":
            raise ValueError("unsupported D9 class rule")
        if payload["overlap_measure"] != "aabb_iou_from_observation_obb":
            raise ValueError("unsupported D9 overlap measure")
        if payload["spatial_rule"] != "center_distance_pass OR overlap_pass":
            raise ValueError("unsupported D9 spatial rule")
        return cls(
            center_distance_threshold=float(
                payload["center_distance_threshold"]
            ),
            min_overlap_iou=float(payload["min_overlap_iou"]),
            min_distinct_frames=int(payload["min_distinct_frames"]),
        )


@dataclass(frozen=True)
class ManualInstanceGroup:
    instance_id: str
    observation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.instance_id.strip():
            raise ValueError("instance_id is required")
        if not self.observation_ids:
            raise ValueError("manual instance group cannot be empty")
        if len(set(self.observation_ids)) != len(self.observation_ids):
            raise ValueError("manual instance group ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "observation_ids": list(self.observation_ids),
        }


@dataclass(frozen=True)
class ManualInstanceLabels:
    scene_id: str
    query: str
    annotation_method: str
    notes: tuple[str, ...]
    instance_groups: tuple[ManualInstanceGroup, ...]
    schema_version: str = D9_LABEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != D9_LABEL_SCHEMA_VERSION:
            raise ValueError("unsupported D9 label schema")
        if not self.scene_id.strip() or not self.query.strip():
            raise ValueError("label scene_id and query are required")
        if not self.annotation_method.strip():
            raise ValueError("annotation_method is required")
        if not self.instance_groups:
            raise ValueError("at least one manual instance group is required")
        observation_ids = [
            obs_id
            for group in self.instance_groups
            for obs_id in group.observation_ids
        ]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("observation id appears in multiple label groups")

    @property
    def instance_by_observation(self) -> dict[str, str]:
        return {
            obs_id: group.instance_id
            for group in self.instance_groups
            for obs_id in group.observation_ids
        }

    def validate_observations(
        self,
        observations: Sequence[ObjectObservation],
    ) -> None:
        actual = {item.obs_id for item in observations}
        expected = set(self.instance_by_observation)
        if actual != expected:
            raise ValueError(
                "manual labels do not exactly cover observations: "
                f"missing={sorted(actual - expected)} "
                f"unknown={sorted(expected - actual)}"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "scene_id": self.scene_id,
            "query": self.query,
            "annotation_method": self.annotation_method,
            "notes": list(self.notes),
            "instance_groups": [
                group.to_dict() for group in self.instance_groups
            ],
        }
        if tuple(payload) != D9_LABEL_FIELDS:
            raise AssertionError("D9 label fields changed")
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ManualInstanceLabels":
        if set(payload) != set(D9_LABEL_FIELDS):
            raise ValueError("D9 label fields are not frozen")
        raw_groups = payload["instance_groups"]
        if not isinstance(raw_groups, list):
            raise ValueError("instance_groups must be a list")
        groups: list[ManualInstanceGroup] = []
        for raw in raw_groups:
            if (
                not isinstance(raw, Mapping)
                or set(raw) != set(D9_INSTANCE_GROUP_FIELDS)
                or not isinstance(raw["observation_ids"], list)
            ):
                raise ValueError("D9 instance group fields are not frozen")
            groups.append(ManualInstanceGroup(
                instance_id=str(raw["instance_id"]),
                observation_ids=tuple(
                    str(value) for value in raw["observation_ids"]
                ),
            ))
        notes = payload["notes"]
        if not isinstance(notes, list) or not all(
            isinstance(note, str) for note in notes
        ):
            raise ValueError("D9 label notes must be a list of strings")
        return cls(
            scene_id=str(payload["scene_id"]),
            query=str(payload["query"]),
            annotation_method=str(payload["annotation_method"]),
            notes=tuple(notes),
            instance_groups=tuple(groups),
            schema_version=str(payload["schema_version"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ManualInstanceLabels":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("D9 label root must be an object")
        return cls.from_dict(payload)


@dataclass(frozen=True)
class EvaluatedPair:
    obs_id_a: str
    obs_id_b: str
    expected_same: bool
    same_class: bool
    center_distance: float
    center_distance_pass: bool
    overlap_iou: float
    overlap_pass: bool
    predicted_same: bool
    gate_reasons: tuple[str, ...]
    error_type: str | None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "obs_id_a": self.obs_id_a,
            "obs_id_b": self.obs_id_b,
            "expected_same": self.expected_same,
            "same_class": self.same_class,
            "center_distance": self.center_distance,
            "center_distance_pass": self.center_distance_pass,
            "overlap_iou": self.overlap_iou,
            "overlap_pass": self.overlap_pass,
            "predicted_same": self.predicted_same,
            "gate_reasons": list(self.gate_reasons),
            "error_type": self.error_type,
        }
        if tuple(payload) != D9_PAIR_FIELDS:
            raise AssertionError("D9 pair fields changed")
        return payload


def evaluate_pair(
    first: ObjectObservation,
    second: ObjectObservation,
    *,
    expected_same: bool,
    config: SpatialGateConfig,
) -> EvaluatedPair:
    first_class = normalize_class_text(first.class_text)
    second_class = normalize_class_text(second.class_text)
    same_class = bool(first_class) and first_class == second_class
    distance = float(np.linalg.norm(first.center - second.center))
    center_pass = distance <= config.center_distance_threshold
    overlap = float(aabb_iou(first.obb, second.obb))
    overlap_pass = overlap > config.min_overlap_iou
    predicted = same_class and (center_pass or overlap_pass)
    reasons: list[str] = []
    if not same_class:
        reasons.append("class_mismatch")
    else:
        if center_pass:
            reasons.append("center_distance")
        if overlap_pass:
            reasons.append("aabb_overlap")
        if not center_pass and not overlap_pass:
            reasons.append("spatial_reject")
    error_type = None
    if predicted and not expected_same:
        error_type = "false_positive"
    elif expected_same and not predicted:
        error_type = "false_negative"
    return EvaluatedPair(
        obs_id_a=first.obs_id,
        obs_id_b=second.obs_id,
        expected_same=bool(expected_same),
        same_class=bool(same_class),
        center_distance=distance,
        center_distance_pass=bool(center_pass),
        overlap_iou=overlap,
        overlap_pass=bool(overlap_pass),
        predicted_same=bool(predicted),
        gate_reasons=tuple(reasons),
        error_type=error_type,
    )


def evaluate_all_pairs(
    observations: Sequence[ObjectObservation],
    labels: ManualInstanceLabels,
    config: SpatialGateConfig,
) -> list[EvaluatedPair]:
    labels.validate_observations(observations)
    instance_by_observation = labels.instance_by_observation
    return [
        evaluate_pair(
            first,
            second,
            expected_same=(
                instance_by_observation[first.obs_id]
                == instance_by_observation[second.obs_id]
            ),
            config=config,
        )
        for first, second in combinations(observations, 2)
    ]


def pairwise_metrics(pairs: Sequence[EvaluatedPair]) -> dict[str, Any]:
    true_positive = sum(
        pair.expected_same and pair.predicted_same for pair in pairs
    )
    false_positive = sum(
        not pair.expected_same and pair.predicted_same for pair in pairs
    )
    true_negative = sum(
        not pair.expected_same and not pair.predicted_same for pair in pairs
    )
    false_negative = sum(
        pair.expected_same and not pair.predicted_same for pair in pairs
    )
    positive = true_positive + false_negative
    negative = true_negative + false_positive
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = true_positive / positive if positive else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    accuracy = (
        (true_positive + true_negative) / len(pairs)
        if pairs
        else 0.0
    )
    payload = {
        "pair_count": len(pairs),
        "positive_pairs": positive,
        "negative_pairs": negative,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }
    if tuple(payload) != D9_METRIC_FIELDS:
        raise AssertionError("D9 metric fields changed")
    return payload


def predicted_components(
    observations: Sequence[ObjectObservation],
    pairs: Sequence[EvaluatedPair],
) -> list[list[ObjectObservation]]:
    parent = {item.obs_id: item.obs_id for item in observations}

    def find(obs_id: str) -> str:
        while parent[obs_id] != obs_id:
            parent[obs_id] = parent[parent[obs_id]]
            obs_id = parent[obs_id]
        return obs_id

    def union(first: str, second: str) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    for pair in pairs:
        if pair.predicted_same:
            union(pair.obs_id_a, pair.obs_id_b)

    grouped: dict[str, list[ObjectObservation]] = {}
    for observation in observations:
        grouped.setdefault(find(observation.obs_id), []).append(observation)
    return list(grouped.values())


def associate_pending(
    memory: ObjectMemory,
    labels: ManualInstanceLabels,
    config: SpatialGateConfig,
) -> dict[str, Any]:
    """Run D9 on a pristine D8 memory and mutate it to the D9 state."""

    if memory.objects or memory.decisions:
        raise ValueError("D9 input must be an unassociated D8 ObjectMemory")
    observations = list(memory.pending_observations.values())
    if not observations:
        raise ValueError("D9 input has no pending observations")
    if memory.metadata.get("scene_id") != labels.scene_id:
        raise ValueError("D9 labels scene_id differs from memory")
    if memory.metadata.get("query") != labels.query:
        raise ValueError("D9 labels query differs from memory")
    pairs = evaluate_all_pairs(observations, labels, config)
    components = predicted_components(observations, pairs)

    memory.config = AssociationConfig(
        distance_threshold=config.center_distance_threshold,
        semantic_threshold=1.0,
        min_match_score=0.0,
        distance_weight=1.0,
        semantic_weight=0.0,
        overlap_weight=0.0,
    )

    component_records: list[dict[str, Any]] = []
    for index, component in enumerate(components, start=1):
        frame_ids = list(dict.fromkeys(item.frame_id for item in component))
        promoted = len(frame_ids) >= config.min_distinct_frames
        object_id = None
        deferred_reason = None
        if promoted:
            decisions = memory.promote_group(component)
            object_id = decisions[0].object_id
        else:
            deferred_reason = "insufficient_distinct_frames"
        record = {
            "component_id": f"component_{index:04d}",
            "observation_ids": [item.obs_id for item in component],
            "frame_ids": frame_ids,
            "distinct_frame_count": len(frame_ids),
            "promoted": promoted,
            "object_id": object_id,
            "deferred_reason": deferred_reason,
        }
        if tuple(record) != D9_COMPONENT_FIELDS:
            raise AssertionError("D9 component fields changed")
        component_records.append(record)

    failures = [
        pair.to_dict() for pair in pairs if pair.error_type is not None
    ]
    return {
        "pairs": [pair.to_dict() for pair in pairs],
        "metrics": pairwise_metrics(pairs),
        "components": component_records,
        "failure_cases": failures,
    }
