"""D12 label-free evidence-aware association with complete-link clustering."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np

from .association import (
    AssociationConfig,
    ObjectMemory,
    aabb_iou,
    cosine_similarity,
)
from .d9_association import ManualInstanceLabels, normalize_class_text
from .schemas import ObjectObservation, observation_quality


A2_SCHEMA_VERSION = "0.1"
A2_EVALUATION_SCHEMA_VERSION = "0.1"
A2_ASSOCIATION_ID = "A2-evidence-aware-complete-link"
A2_WEIGHT_FIELDS = (
    "semantic",
    "center",
    "overlap",
    "obb_shape",
    "quality",
)
A2_CONFIG_FIELDS = (
    "semantic_rule",
    "semantic_threshold",
    "min_observation_quality",
    "center_distance_threshold",
    "overlap_measure",
    "min_overlap_iou",
    "spatial_rule",
    "obb_shape_measure",
    "quality_measure",
    "weights",
    "cluster_rule",
    "min_distinct_frames",
)
A2_PAIR_FIELDS = (
    "obs_id_a",
    "obs_id_b",
    "frame_id_a",
    "frame_id_b",
    "semantic_mode",
    "semantic_similarity",
    "semantic_score",
    "semantic_compatible",
    "center_distance",
    "center_score",
    "center_pass",
    "aabb_iou",
    "overlap_pass",
    "extent_ratios",
    "obb_shape_similarity",
    "quality_a",
    "quality_b",
    "pair_quality",
    "quality_pass",
    "pair_score",
    "gate_pass",
    "predicted_same",
    "reasons",
)
A2_MERGE_FIELDS = (
    "step",
    "left_observation_ids",
    "right_observation_ids",
    "cross_pairs",
    "complete_link_score",
    "accepted",
    "reason",
)
A2_CLUSTER_FIELDS = (
    "cluster_id",
    "observation_ids",
    "frame_ids",
    "distinct_frame_count",
    "promoted",
    "object_id",
    "deferred_reason",
)
A2_METRIC_FIELDS = (
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
A2_PREDICTION_RESULT_FIELDS = (
    "schema_version",
    "status",
    "stage",
    "association_id",
    "scene_id",
    "query",
    "source",
    "config",
    "counts",
    "pairs",
    "merge_decisions",
    "clusters",
    "acceptance",
    "artifacts",
    "created_at",
)
A2_PREDICTION_SOURCE_FIELDS = ("d8_memory", "d8_memory_sha256")
A2_PREDICTION_ARTIFACT_FIELDS = ("source_memory", "object_memory")
A2_COUNT_FIELDS = (
    "input_observations",
    "pair_count",
    "gate_pass_pairs",
    "predicted_match_pairs",
    "merge_count",
    "cluster_count",
    "promoted_clusters",
    "deferred_clusters",
    "permanent_objects",
    "pending_observations",
    "association_decisions",
)
A2_ACCEPTANCE_FIELDS = (
    "observation_conservation",
    "deterministic_recompute",
    "complete_link_pass",
    "cross_frame_object_pass",
    "round_trip_equal",
)
A2_EVALUATION_RESULT_FIELDS = (
    "schema_version",
    "status",
    "stage",
    "association_id",
    "scene_id",
    "query",
    "source",
    "metrics",
    "pairs",
    "failure_cases",
    "acceptance",
    "artifacts",
    "created_at",
)
A2_EVALUATION_SOURCE_FIELDS = (
    "prediction_result",
    "prediction_result_sha256",
    "pair_labels",
    "pair_labels_sha256",
)
A2_EVALUATION_ARTIFACT_FIELDS = ("pair_labels",)
A2_EVALUATION_ACCEPTANCE_FIELDS = (
    "prediction_valid",
    "evaluation_recomputed",
    "thresholds_frozen_before_evaluation",
)


def _strict_fields(
    payload: Mapping[str, Any],
    fields: tuple[str, ...],
    name: str,
) -> None:
    if set(payload) != set(fields):
        raise ValueError(f"{name} fields are not frozen")


@dataclass(frozen=True)
class EvidenceAssociationConfig:
    semantic_threshold: float = 0.70
    min_observation_quality: float = 0.25
    center_distance_threshold: float = 0.15
    min_overlap_iou: float = 0.0
    semantic_weight: float = 0.25
    center_weight: float = 0.25
    overlap_weight: float = 0.20
    obb_shape_weight: float = 0.15
    quality_weight: float = 0.15
    min_distinct_frames: int = 2

    def __post_init__(self) -> None:
        for name in (
            "semantic_threshold",
            "min_observation_quality",
            "min_overlap_iou",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if (
            not np.isfinite(self.center_distance_threshold)
            or self.center_distance_threshold <= 0.0
        ):
            raise ValueError("center_distance_threshold must be positive")
        weights = (
            self.semantic_weight,
            self.center_weight,
            self.overlap_weight,
            self.obb_shape_weight,
            self.quality_weight,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("A2 weights must be finite and non-negative")
        if not np.isclose(sum(weights), 1.0):
            raise ValueError("A2 weights must sum to one")
        if self.min_distinct_frames < 2:
            raise ValueError("min_distinct_frames must be at least two")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "semantic_rule": (
                "cosine_when_both_embeddings_else_normalized_exact_class"
            ),
            "semantic_threshold": self.semantic_threshold,
            "min_observation_quality": self.min_observation_quality,
            "center_distance_threshold": self.center_distance_threshold,
            "overlap_measure": "aabb_iou_from_observation_obb",
            "min_overlap_iou": self.min_overlap_iou,
            "spatial_rule": "center_pass OR overlap_pass",
            "obb_shape_measure": "mean_sorted_extent_ratio",
            "quality_measure": (
                "min_pair_geometric_mean_retrieval_sam_valid_points"
            ),
            "weights": {
                "semantic": self.semantic_weight,
                "center": self.center_weight,
                "overlap": self.overlap_weight,
                "obb_shape": self.obb_shape_weight,
                "quality": self.quality_weight,
            },
            "cluster_rule": "complete_link_all_cross_pairs_gate_pass",
            "min_distinct_frames": self.min_distinct_frames,
        }
        if tuple(payload) != A2_CONFIG_FIELDS:
            raise AssertionError("A2 config fields changed")
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceAssociationConfig":
        _strict_fields(payload, A2_CONFIG_FIELDS, "A2 config")
        expected_text = cls().to_dict()
        for key in (
            "semantic_rule",
            "overlap_measure",
            "spatial_rule",
            "obb_shape_measure",
            "quality_measure",
            "cluster_rule",
        ):
            if payload[key] != expected_text[key]:
                raise ValueError(f"unsupported A2 {key}")
        weights = payload["weights"]
        if not isinstance(weights, Mapping):
            raise ValueError("A2 weights must be an object")
        _strict_fields(weights, A2_WEIGHT_FIELDS, "A2 weights")
        return cls(
            semantic_threshold=float(payload["semantic_threshold"]),
            min_observation_quality=float(
                payload["min_observation_quality"]
            ),
            center_distance_threshold=float(
                payload["center_distance_threshold"]
            ),
            min_overlap_iou=float(payload["min_overlap_iou"]),
            semantic_weight=float(weights["semantic"]),
            center_weight=float(weights["center"]),
            overlap_weight=float(weights["overlap"]),
            obb_shape_weight=float(weights["obb_shape"]),
            quality_weight=float(weights["quality"]),
            min_distinct_frames=int(payload["min_distinct_frames"]),
        )


@dataclass(frozen=True)
class EvidencePair:
    obs_id_a: str
    obs_id_b: str
    frame_id_a: str
    frame_id_b: str
    semantic_mode: str
    semantic_similarity: float
    semantic_score: float
    semantic_compatible: bool
    center_distance: float
    center_score: float
    center_pass: bool
    aabb_iou: float
    overlap_pass: bool
    extent_ratios: tuple[float, float, float]
    obb_shape_similarity: float
    quality_a: float
    quality_b: float
    pair_quality: float
    quality_pass: bool
    pair_score: float
    gate_pass: bool
    predicted_same: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.obs_id_a or self.obs_id_a >= self.obs_id_b:
            raise ValueError("A2 pair ids must be canonical and unique")
        if self.semantic_mode not in {"embedding_cosine", "exact_class"}:
            raise ValueError("unsupported A2 semantic mode")
        finite = (
            self.semantic_similarity,
            self.semantic_score,
            self.center_distance,
            self.center_score,
            self.aabb_iou,
            *self.extent_ratios,
            self.obb_shape_similarity,
            self.quality_a,
            self.quality_b,
            self.pair_quality,
            self.pair_score,
        )
        if not all(np.isfinite(value) for value in finite):
            raise ValueError("A2 pair features must be finite")
        bounded = (
            self.semantic_score,
            self.center_score,
            self.aabb_iou,
            *self.extent_ratios,
            self.obb_shape_similarity,
            self.quality_a,
            self.quality_b,
            self.pair_quality,
            self.pair_score,
        )
        if any(value < 0.0 or value > 1.0 for value in bounded):
            raise ValueError("A2 normalized pair features must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "obs_id_a": self.obs_id_a,
            "obs_id_b": self.obs_id_b,
            "frame_id_a": self.frame_id_a,
            "frame_id_b": self.frame_id_b,
            "semantic_mode": self.semantic_mode,
            "semantic_similarity": self.semantic_similarity,
            "semantic_score": self.semantic_score,
            "semantic_compatible": self.semantic_compatible,
            "center_distance": self.center_distance,
            "center_score": self.center_score,
            "center_pass": self.center_pass,
            "aabb_iou": self.aabb_iou,
            "overlap_pass": self.overlap_pass,
            "extent_ratios": list(self.extent_ratios),
            "obb_shape_similarity": self.obb_shape_similarity,
            "quality_a": self.quality_a,
            "quality_b": self.quality_b,
            "pair_quality": self.pair_quality,
            "quality_pass": self.quality_pass,
            "pair_score": self.pair_score,
            "gate_pass": self.gate_pass,
            "predicted_same": self.predicted_same,
            "reasons": list(self.reasons),
        }
        if tuple(payload) != A2_PAIR_FIELDS:
            raise AssertionError("A2 pair fields changed")
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidencePair":
        _strict_fields(payload, A2_PAIR_FIELDS, "A2 pair")
        ratios = payload["extent_ratios"]
        reasons = payload["reasons"]
        if not isinstance(ratios, list) or len(ratios) != 3:
            raise ValueError("A2 extent ratios must have length three")
        if not isinstance(reasons, list) or not all(
            isinstance(value, str) for value in reasons
        ):
            raise ValueError("A2 reasons must be strings")
        return cls(
            obs_id_a=str(payload["obs_id_a"]),
            obs_id_b=str(payload["obs_id_b"]),
            frame_id_a=str(payload["frame_id_a"]),
            frame_id_b=str(payload["frame_id_b"]),
            semantic_mode=str(payload["semantic_mode"]),
            semantic_similarity=float(payload["semantic_similarity"]),
            semantic_score=float(payload["semantic_score"]),
            semantic_compatible=bool(payload["semantic_compatible"]),
            center_distance=float(payload["center_distance"]),
            center_score=float(payload["center_score"]),
            center_pass=bool(payload["center_pass"]),
            aabb_iou=float(payload["aabb_iou"]),
            overlap_pass=bool(payload["overlap_pass"]),
            extent_ratios=tuple(float(value) for value in ratios),
            obb_shape_similarity=float(payload["obb_shape_similarity"]),
            quality_a=float(payload["quality_a"]),
            quality_b=float(payload["quality_b"]),
            pair_quality=float(payload["pair_quality"]),
            quality_pass=bool(payload["quality_pass"]),
            pair_score=float(payload["pair_score"]),
            gate_pass=bool(payload["gate_pass"]),
            predicted_same=bool(payload["predicted_same"]),
            reasons=tuple(reasons),
        )


def _semantic_evidence(
    first: ObjectObservation,
    second: ObjectObservation,
    config: EvidenceAssociationConfig,
) -> tuple[str, float, float, bool]:
    if (
        first.semantic_embedding is not None
        and second.semantic_embedding is not None
    ):
        similarity = cosine_similarity(
            first.semantic_embedding,
            second.semantic_embedding,
        )
        score = max(0.0, similarity)
        return (
            "embedding_cosine",
            similarity,
            score,
            similarity >= config.semantic_threshold,
        )
    same_class = (
        bool(normalize_class_text(first.class_text))
        and normalize_class_text(first.class_text)
        == normalize_class_text(second.class_text)
    )
    score = 1.0 if same_class else 0.0
    return "exact_class", score, score, same_class


def _extent_ratio(
    first: ObjectObservation,
    second: ObjectObservation,
) -> tuple[float, float, float]:
    first_extent = np.sort(np.asarray(first.obb.extent, dtype=float))
    second_extent = np.sort(np.asarray(second.obb.extent, dtype=float))
    lower = np.minimum(first_extent, second_extent)
    upper = np.maximum(first_extent, second_extent)
    ratio = np.ones(3, dtype=float)
    nonzero = upper > 1e-12
    ratio[nonzero] = lower[nonzero] / upper[nonzero]
    return tuple(float(value) for value in ratio)


def compute_pair(
    first: ObjectObservation,
    second: ObjectObservation,
    config: EvidenceAssociationConfig,
) -> EvidencePair:
    """Compute one label-free A2 pair record."""

    if first.obs_id == second.obs_id:
        raise ValueError("cannot associate duplicate observation ids")
    if first.obs_id > second.obs_id:
        first, second = second, first
    semantic_mode, semantic_similarity, semantic_score, semantic_pass = (
        _semantic_evidence(first, second, config)
    )
    distance = float(np.linalg.norm(first.center - second.center))
    center_score = max(
        0.0, 1.0 - distance / config.center_distance_threshold
    )
    center_pass = distance <= config.center_distance_threshold
    overlap = float(aabb_iou(first.obb, second.obb))
    overlap_pass = overlap > config.min_overlap_iou
    extent_ratios = _extent_ratio(first, second)
    shape_score = float(np.mean(extent_ratios))
    quality_a = observation_quality(first)
    quality_b = observation_quality(second)
    pair_quality = min(quality_a, quality_b)
    quality_pass = (
        quality_a >= config.min_observation_quality
        and quality_b >= config.min_observation_quality
    )
    pair_score = (
        config.semantic_weight * semantic_score
        + config.center_weight * center_score
        + config.overlap_weight * overlap
        + config.obb_shape_weight * shape_score
        + config.quality_weight * pair_quality
    )
    spatial_pass = center_pass or overlap_pass
    gate_pass = semantic_pass and quality_pass and spatial_pass
    reasons: list[str] = []
    if semantic_pass:
        reasons.append(f"semantic_{semantic_mode}_pass")
    else:
        reasons.append(f"semantic_{semantic_mode}_reject")
    if quality_pass:
        reasons.append("quality_pass")
    else:
        if quality_a < config.min_observation_quality:
            reasons.append("low_quality_a")
        if quality_b < config.min_observation_quality:
            reasons.append("low_quality_b")
    if center_pass:
        reasons.append("center_distance_pass")
    if overlap_pass:
        reasons.append("aabb_overlap_pass")
    if not spatial_pass:
        reasons.append("spatial_reject")
    reasons.append("gate_accept" if gate_pass else "gate_reject")
    return EvidencePair(
        obs_id_a=first.obs_id,
        obs_id_b=second.obs_id,
        frame_id_a=first.frame_id,
        frame_id_b=second.frame_id,
        semantic_mode=semantic_mode,
        semantic_similarity=semantic_similarity,
        semantic_score=semantic_score,
        semantic_compatible=semantic_pass,
        center_distance=distance,
        center_score=center_score,
        center_pass=center_pass,
        aabb_iou=overlap,
        overlap_pass=overlap_pass,
        extent_ratios=extent_ratios,
        obb_shape_similarity=shape_score,
        quality_a=quality_a,
        quality_b=quality_b,
        pair_quality=pair_quality,
        quality_pass=quality_pass,
        pair_score=pair_score,
        gate_pass=gate_pass,
        predicted_same=False,
        reasons=tuple(reasons),
    )


def predict_all_pairs(
    observations: Sequence[ObjectObservation],
    config: EvidenceAssociationConfig,
) -> list[EvidencePair]:
    ordered = sorted(observations, key=lambda item: item.obs_id)
    ids = [item.obs_id for item in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("A2 observations must have unique ids")
    return [
        compute_pair(first, second, config)
        for first, second in combinations(ordered, 2)
    ]


def complete_link_clusters(
    observations: Sequence[ObjectObservation],
    pairs: Sequence[EvidencePair],
) -> tuple[list[list[ObjectObservation]], list[EvidencePair], list[dict[str, Any]]]:
    """Agglomerate only clusters whose every cross-pair passes A2 gates."""

    ordered = sorted(observations, key=lambda item: item.obs_id)
    by_id = {item.obs_id: item for item in ordered}
    pair_by_key = {
        (pair.obs_id_a, pair.obs_id_b): pair for pair in pairs
    }
    expected = {
        (first.obs_id, second.obs_id)
        for first, second in combinations(ordered, 2)
    }
    if set(pair_by_key) != expected:
        raise ValueError("A2 pairs must cover every observation pair exactly")
    clusters: list[tuple[str, ...]] = [(item.obs_id,) for item in ordered]
    merge_decisions: list[dict[str, Any]] = []
    while True:
        candidates: list[
            tuple[float, tuple[str, ...], tuple[str, ...], list[EvidencePair]]
        ] = []
        for left, right in combinations(clusters, 2):
            cross = [
                pair_by_key[tuple(sorted((first, second)))]
                for first in left
                for second in right
            ]
            if all(pair.gate_pass for pair in cross):
                candidates.append((
                    min(pair.pair_score for pair in cross),
                    left,
                    right,
                    cross,
                ))
        if not candidates:
            break
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        score, left, right, cross = candidates[0]
        merge_decision = {
            "step": len(merge_decisions) + 1,
            "left_observation_ids": list(left),
            "right_observation_ids": list(right),
            "cross_pairs": [
                [pair.obs_id_a, pair.obs_id_b]
                for pair in sorted(
                    cross, key=lambda item: (item.obs_id_a, item.obs_id_b)
                )
            ],
            "complete_link_score": score,
            "accepted": True,
            "reason": "all_cross_pairs_gate_pass",
        }
        if tuple(merge_decision) != A2_MERGE_FIELDS:
            raise AssertionError("A2 merge-decision fields changed")
        merge_decisions.append(merge_decision)
        clusters = [item for item in clusters if item not in {left, right}]
        clusters.append(tuple(sorted((*left, *right))))
        clusters.sort()

    clusters.sort(key=lambda item: item[0])
    cluster_by_id = {
        obs_id: index
        for index, cluster in enumerate(clusters)
        for obs_id in cluster
    }
    finalized: list[EvidencePair] = []
    for pair in sorted(
        pairs, key=lambda item: (item.obs_id_a, item.obs_id_b)
    ):
        predicted_same = (
            cluster_by_id[pair.obs_id_a] == cluster_by_id[pair.obs_id_b]
        )
        reasons = list(pair.reasons)
        if predicted_same:
            reasons.append("complete_link_member")
        elif pair.gate_pass:
            reasons.append("complete_link_conflict")
        else:
            reasons.append("not_mergeable")
        finalized.append(replace(
            pair,
            predicted_same=predicted_same,
            reasons=tuple(reasons),
        ))
    components = [[by_id[obs_id] for obs_id in cluster] for cluster in clusters]
    return components, finalized, merge_decisions


def associate_pending_a2(
    memory: ObjectMemory,
    config: EvidenceAssociationConfig,
) -> dict[str, Any]:
    """Run A2 on a pristine D8 memory and mutate it deterministically."""

    if memory.objects or memory.decisions:
        raise ValueError("A2 input must be an unassociated D8 ObjectMemory")
    observations = sorted(
        memory.pending_observations.values(), key=lambda item: item.obs_id
    )
    if not observations:
        raise ValueError("A2 input has no pending observations")
    memory.pending_observations = {
        item.obs_id: item for item in observations
    }
    raw_pairs = predict_all_pairs(observations, config)
    clusters, pairs, merge_decisions = complete_link_clusters(
        observations, raw_pairs
    )
    memory.config = AssociationConfig(
        distance_threshold=config.center_distance_threshold,
        semantic_threshold=config.semantic_threshold,
        min_match_score=0.0,
        distance_weight=1.0,
        semantic_weight=0.0,
        overlap_weight=0.0,
    )

    records: list[dict[str, Any]] = []
    for index, cluster in enumerate(clusters, start=1):
        frame_ids = sorted({item.frame_id for item in cluster})
        promoted = len(frame_ids) >= config.min_distinct_frames
        object_id = None
        deferred_reason = None
        if promoted:
            decisions = memory.promote_group(cluster)
            object_id = decisions[0].object_id
        else:
            deferred_reason = "insufficient_distinct_frames"
        record = {
            "cluster_id": f"a2_cluster_{index:04d}",
            "observation_ids": [item.obs_id for item in cluster],
            "frame_ids": frame_ids,
            "distinct_frame_count": len(frame_ids),
            "promoted": promoted,
            "object_id": object_id,
            "deferred_reason": deferred_reason,
        }
        if tuple(record) != A2_CLUSTER_FIELDS:
            raise AssertionError("A2 cluster fields changed")
        records.append(record)
    return {
        "pairs": [item.to_dict() for item in pairs],
        "merge_decisions": merge_decisions,
        "clusters": records,
    }


def evaluate_a2_predictions(
    observations: Sequence[ObjectObservation],
    predictions: Sequence[Mapping[str, Any]],
    labels: ManualInstanceLabels,
) -> dict[str, Any]:
    """Attach labels only after the frozen A2 prediction has completed."""

    ordered = sorted(observations, key=lambda item: item.obs_id)
    labels.validate_observations(ordered)
    normalized = [EvidencePair.from_dict(item) for item in predictions]
    expected_keys = {
        (first.obs_id, second.obs_id)
        for first, second in combinations(ordered, 2)
    }
    actual_keys = [(item.obs_id_a, item.obs_id_b) for item in normalized]
    if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != expected_keys:
        raise ValueError("A2 predictions must cover every pair exactly once")
    instance_by_observation = labels.instance_by_observation
    evaluated: list[dict[str, Any]] = []
    for pair in sorted(
        normalized, key=lambda item: (item.obs_id_a, item.obs_id_b)
    ):
        expected_same = (
            instance_by_observation[pair.obs_id_a]
            == instance_by_observation[pair.obs_id_b]
        )
        error_type = None
        if pair.predicted_same and not expected_same:
            error_type = "false_positive"
        elif expected_same and not pair.predicted_same:
            error_type = "false_negative"
        evaluated.append({
            "obs_id_a": pair.obs_id_a,
            "obs_id_b": pair.obs_id_b,
            "expected_same": expected_same,
            "gate_pass": pair.gate_pass,
            "predicted_same": pair.predicted_same,
            "error_type": error_type,
        })

    true_positive = sum(
        item["expected_same"] and item["predicted_same"]
        for item in evaluated
    )
    false_positive = sum(
        not item["expected_same"] and item["predicted_same"]
        for item in evaluated
    )
    true_negative = sum(
        not item["expected_same"] and not item["predicted_same"]
        for item in evaluated
    )
    false_negative = sum(
        item["expected_same"] and not item["predicted_same"]
        for item in evaluated
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
    metrics = {
        "pair_count": len(evaluated),
        "positive_pairs": positive,
        "negative_pairs": negative,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (
            (true_positive + true_negative) / len(evaluated)
            if evaluated
            else 0.0
        ),
    }
    if tuple(metrics) != A2_METRIC_FIELDS:
        raise AssertionError("A2 metric fields changed")
    return {
        "pairs": evaluated,
        "metrics": metrics,
        "failure_cases": [
            item for item in evaluated if item["error_type"] is not None
        ],
    }
