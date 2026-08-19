"""Relation-constrained instance ranking in an explicit anchor frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .association import ObjectMemory, text_similarity
from .schemas import GroundingQuery, GroundingResult, MemoryObject


RELATION_ALIASES = {
    "left": "left_of",
    "left_of": "left_of",
    "right": "right_of",
    "right_of": "right_of",
    "front": "front_of",
    "front_of": "front_of",
    "in_front_of": "front_of",
    "behind": "behind",
    "back_of": "behind",
}


@dataclass(frozen=True)
class RelationConfig:
    margin: float = 0.10
    ambiguity_margin: float = 0.05
    confidence_threshold: float = 0.50

    def __post_init__(self) -> None:
        if self.margin < 0.0 or self.ambiguity_margin < 0.0:
            raise ValueError("relation margins must be non-negative")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")


class RelationGrounder:
    """Ranks target objects using class evidence and directional constraints.

    Pose convention: ``anchor_poses[frame_id]`` is ``world_from_anchor``.
    Anchor +x is right, +y is up and +z is front. Directional queries never
    silently fall back to an unspecified world frame.
    """

    def __init__(
        self,
        memory: ObjectMemory,
        anchor_poses: Mapping[str, np.ndarray] | None = None,
        config: RelationConfig | None = None,
    ) -> None:
        self.memory = memory
        self.anchor_poses = dict(anchor_poses or {})
        self.config = config or RelationConfig()

    def ground(self, query: GroundingQuery) -> GroundingResult:
        targets = self._matching_objects(query.target)
        if not targets:
            return self._abstain(query, "target_not_found")

        if query.relation is None:
            ranked = sorted(
                ((self._semantic_score(query.target, item) * item.confidence, item) for item in targets),
                key=lambda pair: (-pair[0], pair[1].object_id),
            )
            return self._finish_semantic(query, ranked)

        relation = RELATION_ALIASES.get(query.relation.lower())
        if relation is None:
            return self._abstain(query, "unsupported_relation")
        if not query.reference:
            return self._abstain(query, "missing_reference")
        references = self._matching_objects(query.reference)
        if not references:
            return self._abstain(query, "reference_not_found")
        if not query.anchor_frame:
            return self._abstain(query, "missing_anchor_frame")
        if query.anchor_frame not in self.anchor_poses:
            return self._abstain(query, "anchor_pose_not_found")

        pose = np.asarray(self.anchor_poses[query.anchor_frame], dtype=np.float64)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            return self._abstain(query, "invalid_anchor_pose")
        world_rotation_from_anchor = pose[:3, :3]
        if not np.allclose(world_rotation_from_anchor.T @ world_rotation_from_anchor, np.eye(3), atol=1e-4):
            return self._abstain(query, "invalid_anchor_rotation")

        ranked: list[tuple[float, float, float, MemoryObject, MemoryObject]] = []
        for target in targets:
            best: tuple[float, float, float, MemoryObject] | None = None
            for reference in references:
                if target.object_id == reference.object_id:
                    continue
                delta_world = target.fused_center - reference.fused_center
                delta_anchor = world_rotation_from_anchor.T @ delta_world
                signed_distance = self._signed_distance(relation, delta_anchor)
                scale = max(float(np.linalg.norm(delta_anchor)), self.config.margin, 1e-6)
                normalized = signed_distance / scale
                relation_score = float(1.0 / (1.0 + np.exp(-6.0 * normalized)))
                combined = (
                    self._semantic_score(query.target, target)
                    * target.confidence
                    * reference.confidence
                    * (0.20 + 0.80 * relation_score)
                )
                candidate = (combined, signed_distance, relation_score, reference)
                if best is None or candidate[0] > best[0]:
                    best = candidate
            if best is not None:
                ranked.append((*best[:3], target, best[3]))

        ranked.sort(key=lambda item: (-item[0], item[3].object_id))
        if not ranked:
            return self._abstain(query, "no_distinct_target_reference_pair")

        top_score, signed_distance, relation_score, target, reference = ranked[0]
        runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
        ranking_margin = top_score - runner_up
        relation_scores = {item[3].object_id: float(item[2]) for item in ranked}
        evidence = list(dict.fromkeys(target.evidence_frames + reference.evidence_frames))
        reason = None
        if signed_distance <= self.config.margin:
            reason = "relation_conflict_or_boundary"
        elif len(ranked) > 1 and ranking_margin < self.config.ambiguity_margin:
            reason = "ambiguous_candidates"
        elif top_score < self.config.confidence_threshold:
            reason = "low_confidence"
        explanation = {
            "relation": relation,
            "reference_id": reference.object_id,
            "signed_distance": float(signed_distance),
            "required_margin": self.config.margin,
            "relation_score": float(relation_score),
            "ranking_margin": float(ranking_margin),
            "anchor_frame": query.anchor_frame,
            "axis_convention": "+x right, +y up, +z front",
        }
        return GroundingResult(
            query_id=query.query_id,
            ranked_ids=[item[3].object_id for item in ranked],
            relation_scores=relation_scores,
            confidence=float(np.clip(top_score, 0.0, 1.0)),
            evidence_frames=evidence,
            abstain=reason is not None,
            reason=reason,
            explanation=explanation,
        )

    def _finish_semantic(
        self,
        query: GroundingQuery,
        ranked: list[tuple[float, MemoryObject]],
    ) -> GroundingResult:
        top_score, top_object = ranked[0]
        runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
        reason = None
        if len(ranked) > 1 and top_score - runner_up < self.config.ambiguity_margin:
            reason = "ambiguous_candidates"
        elif top_score < self.config.confidence_threshold:
            reason = "low_confidence"
        return GroundingResult(
            query_id=query.query_id,
            ranked_ids=[item.object_id for _, item in ranked],
            relation_scores={},
            confidence=float(np.clip(top_score, 0.0, 1.0)),
            evidence_frames=top_object.evidence_frames,
            abstain=reason is not None,
            reason=reason,
            explanation={"ranking_margin": float(top_score - runner_up), "mode": "semantic_only"},
        )

    def _matching_objects(self, class_text: str) -> list[MemoryObject]:
        return [item for item in self.memory if self._semantic_score(class_text, item) > 0.0]

    @staticmethod
    def _semantic_score(class_text: str, memory_object: MemoryObject) -> float:
        return text_similarity(class_text, memory_object.class_text)

    @staticmethod
    def _signed_distance(relation: str, delta: np.ndarray) -> float:
        if relation == "left_of":
            return float(-delta[0])
        if relation == "right_of":
            return float(delta[0])
        if relation == "front_of":
            return float(delta[2])
        if relation == "behind":
            return float(-delta[2])
        raise ValueError(f"unsupported relation: {relation}")

    @staticmethod
    def _abstain(query: GroundingQuery, reason: str) -> GroundingResult:
        return GroundingResult(
            query_id=query.query_id,
            ranked_ids=[],
            relation_scores={},
            confidence=0.0,
            evidence_frames=[],
            abstain=True,
            reason=reason,
        )
