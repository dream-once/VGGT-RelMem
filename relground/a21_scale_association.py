"""A2.1 development candidate with scale-normalized center geometry."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np

from .a2_association import (
    A2_PAIR_FIELDS,
    EvidenceAssociationConfig,
    EvidencePair,
    complete_link_clusters,
    compute_pair,
    evaluate_a2_predictions,
)
from .association import AssociationConfig, ObjectMemory
from .d9_association import ManualInstanceLabels
from .schemas import ObjectObservation


A21_SCHEMA_VERSION = "0.1"
A21_ASSOCIATION_ID = "A2.1-scale-normalized-center-complete-link"
A21_STATUS = "DEVELOPMENT_CANDIDATE_MORE_NEGATIVES_AND_HELD_OUT_PENDING"


@dataclass(frozen=True)
class ScaleAwareAssociationConfig:
    max_normalized_center_distance: float = 1.0
    semantic_threshold: float = 0.70
    min_observation_quality: float = 0.25
    min_overlap_iou: float = 0.0
    min_distinct_frames: int = 2

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.max_normalized_center_distance)
            or self.max_normalized_center_distance <= 0.0
        ):
            raise ValueError("max normalized center distance must be positive")
        for name in ("semantic_threshold", "min_observation_quality", "min_overlap_iou"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.min_distinct_frames < 2:
            raise ValueError("min_distinct_frames must be at least two")

    def to_dict(self) -> dict[str, Any]:
        return {
            "center_scale_measure": "max_l2_obb_extent",
            "normalized_center_distance": "center_distance / center_scale",
            "max_normalized_center_distance": self.max_normalized_center_distance,
            "semantic_rule": "same_as_frozen_A2",
            "semantic_threshold": self.semantic_threshold,
            "min_observation_quality": self.min_observation_quality,
            "min_overlap_iou": self.min_overlap_iou,
            "spatial_rule": "normalized_center_pass OR aabb_overlap_pass",
            "cluster_rule": "complete_link_all_cross_pairs_gate_pass",
            "min_distinct_frames": self.min_distinct_frames,
            "development_status": A21_STATUS,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScaleAwareAssociationConfig":
        expected = cls().to_dict()
        if set(payload) != set(expected):
            raise ValueError("A2.1 config fields are not frozen")
        for key in (
            "center_scale_measure", "normalized_center_distance", "semantic_rule",
            "spatial_rule", "cluster_rule", "development_status",
        ):
            if payload[key] != expected[key]:
                raise ValueError(f"A2.1 config text changed: {key}")
        return cls(
            max_normalized_center_distance=float(payload["max_normalized_center_distance"]),
            semantic_threshold=float(payload["semantic_threshold"]),
            min_observation_quality=float(payload["min_observation_quality"]),
            min_overlap_iou=float(payload["min_overlap_iou"]),
            min_distinct_frames=int(payload["min_distinct_frames"]),
        )


def _a2_config(config: ScaleAwareAssociationConfig) -> EvidenceAssociationConfig:
    return EvidenceAssociationConfig(
        semantic_threshold=config.semantic_threshold,
        min_observation_quality=config.min_observation_quality,
        min_overlap_iou=config.min_overlap_iou,
        min_distinct_frames=config.min_distinct_frames,
    )


def compute_scale_aware_pair(
    first: ObjectObservation,
    second: ObjectObservation,
    config: ScaleAwareAssociationConfig,
) -> tuple[EvidencePair, dict[str, float | bool]]:
    base_config = _a2_config(config)
    base = compute_pair(first, second, base_config)
    first_scale = float(np.linalg.norm(first.obb.extent))
    second_scale = float(np.linalg.norm(second.obb.extent))
    center_scale = max(first_scale, second_scale)
    if not np.isfinite(center_scale) or center_scale <= 0.0:
        raise ValueError("A2.1 OBB center scale must be finite and positive")
    normalized = base.center_distance / center_scale
    center_pass = normalized <= config.max_normalized_center_distance
    center_score = max(
        0.0,
        1.0 - normalized / config.max_normalized_center_distance,
    )
    spatial_pass = center_pass or base.overlap_pass
    gate_pass = base.semantic_compatible and base.quality_pass and spatial_pass
    pair_score = (
        base_config.semantic_weight * base.semantic_score
        + base_config.center_weight * center_score
        + base_config.overlap_weight * base.aabb_iou
        + base_config.obb_shape_weight * base.obb_shape_similarity
        + base_config.quality_weight * base.pair_quality
    )
    remove = {
        "center_distance_pass", "spatial_reject", "gate_accept",
        "gate_reject", "not_mergeable", "complete_link_member",
        "complete_link_conflict",
    }
    reasons = [item for item in base.reasons if item not in remove]
    if center_pass:
        reasons.append("scale_normalized_center_pass")
    if not spatial_pass:
        reasons.append("spatial_reject")
    reasons.append("gate_accept" if gate_pass else "gate_reject")
    pair = replace(
        base,
        center_score=center_score,
        center_pass=center_pass,
        pair_score=pair_score,
        gate_pass=gate_pass,
        predicted_same=False,
        reasons=tuple(reasons),
    )
    return pair, {
        "center_scale": center_scale,
        "normalized_center_distance": normalized,
        "max_normalized_center_distance": config.max_normalized_center_distance,
        "scale_center_pass": center_pass,
    }


def predict_scale_aware_pairs(
    observations: Sequence[ObjectObservation],
    config: ScaleAwareAssociationConfig,
) -> tuple[list[EvidencePair], dict[tuple[str, str], dict[str, float | bool]]]:
    ordered = sorted(observations, key=lambda item: item.obs_id)
    if len({item.obs_id for item in ordered}) != len(ordered):
        raise ValueError("A2.1 observations must have unique ids")
    pairs: list[EvidencePair] = []
    diagnostics: dict[tuple[str, str], dict[str, float | bool]] = {}
    for first, second in combinations(ordered, 2):
        pair, diagnostic = compute_scale_aware_pair(first, second, config)
        key = (pair.obs_id_a, pair.obs_id_b)
        pairs.append(pair)
        diagnostics[key] = diagnostic
    return pairs, diagnostics


def _pair_payload(
    pair: EvidencePair,
    diagnostic: Mapping[str, float | bool],
) -> dict[str, Any]:
    payload = pair.to_dict()
    payload.update({
        "center_scale": float(diagnostic["center_scale"]),
        "normalized_center_distance": float(diagnostic["normalized_center_distance"]),
        "max_normalized_center_distance": float(diagnostic["max_normalized_center_distance"]),
        "scale_center_pass": bool(diagnostic["scale_center_pass"]),
    })
    return payload


def associate_pending_a21(
    memory: ObjectMemory,
    config: ScaleAwareAssociationConfig,
) -> dict[str, Any]:
    if memory.objects or memory.decisions:
        raise ValueError("A2.1 input must be an unassociated D8 ObjectMemory")
    observations = sorted(memory.pending_observations.values(), key=lambda item: item.obs_id)
    if not observations:
        raise ValueError("A2.1 input has no pending observations")
    memory.pending_observations = {item.obs_id: item for item in observations}
    raw_pairs, diagnostics = predict_scale_aware_pairs(observations, config)
    clusters, finalized_pairs, merge_decisions = complete_link_clusters(observations, raw_pairs)
    memory.config = AssociationConfig(
        distance_threshold=config.max_normalized_center_distance,
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
        records.append({
            "cluster_id": f"a21_cluster_{index:04d}",
            "observation_ids": [item.obs_id for item in cluster],
            "frame_ids": frame_ids,
            "distinct_frame_count": len(frame_ids),
            "promoted": promoted,
            "object_id": object_id,
            "deferred_reason": deferred_reason,
        })
    return {
        "pairs": [
            _pair_payload(pair, diagnostics[(pair.obs_id_a, pair.obs_id_b)])
            for pair in finalized_pairs
        ],
        "merge_decisions": merge_decisions,
        "clusters": records,
    }


def evaluate_a21_predictions(
    observations: Sequence[ObjectObservation],
    predictions: Sequence[Mapping[str, Any]],
    labels: ManualInstanceLabels,
) -> dict[str, Any]:
    a2_payloads = [
        {field: row[field] for field in A2_PAIR_FIELDS}
        for row in predictions
    ]
    return evaluate_a2_predictions(observations, a2_payloads, labels)
