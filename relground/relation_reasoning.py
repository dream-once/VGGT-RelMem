"""Label-free reference-first relation selection and evidence-aware abstention.

This adapter works with upstream single-frame or multi-view candidates. It does
not fit probabilities and never consumes evaluator labels or scene alignment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .association import aabb_iou, text_similarity
from .relations import RELATION_ALIASES, RelationGrounder
from .schemas import GroundingQuery, GroundingResult, OrientedBoundingBox


@dataclass(frozen=True)
class PairReasoningConfig:
    relation_margin: float = 0.10
    minimum_entity_score: float = 0.25
    ambiguity_score_gap: float = 0.05
    equivalence_center_distance: float = 0.15
    equivalence_min_aabb_iou: float = 0.50

    def __post_init__(self):
        values = (self.relation_margin, self.minimum_entity_score,
                  self.ambiguity_score_gap, self.equivalence_center_distance,
                  self.equivalence_min_aabb_iou)
        if not all(np.isfinite(v) and v >= 0 for v in values):
            raise ValueError("reasoning thresholds must be finite and non-negative")
        if any(v > 1 for v in (self.minimum_entity_score, self.ambiguity_score_gap, self.equivalence_min_aabb_iou)):
            raise ValueError("score and IoU thresholds must be <= 1")


def validate_candidates(candidates: Sequence[Mapping[str, Any]]) -> None:
    seen = set()
    for item in candidates:
        if not item['object_id'] or item['object_id'] in seen:
            raise ValueError('candidate ids must be nonempty and unique')
        seen.add(item['object_id'])
        center = np.asarray(item['center'], dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError('candidate center must be finite xyz')
        if not np.isfinite(item['quality']) or not 0 <= item['quality'] <= 1:
            raise ValueError('candidate quality must be in [0, 1]')
        if not isinstance(item['class_text'], str) or not item['class_text'].strip():
            raise ValueError('candidate class text is missing')
        OrientedBoundingBox.from_dict(item['obb'])


class PairRelationGrounder:
    MODES = ('no_relation', 'check_after_selection', 'filter_before_selection', 'filter_with_abstention', 'filter_with_entity_guard')

    def __init__(self, candidates, anchor_poses, config=None):
        validate_candidates(candidates)
        self.candidates = sorted(candidates, key=lambda c: c['object_id'])
        self.anchors = anchor_poses
        self.config = config or PairReasoningConfig()

    def _matching(self, text):
        rows = [(max(text_similarity(text, label) for label in [c['class_text'], *c.get('query_aliases', [])]) * c['quality'], c) for c in self.candidates]
        return sorted(((s, c) for s, c in rows if s > 0), key=lambda r: (-r[0], r[1]['object_id']))

    def _equivalent(self, first, second):
        return (np.linalg.norm(np.asarray(first['center']) - second['center']) <= self.config.equivalence_center_distance
                and aabb_iou(OrientedBoundingBox.from_dict(first['obb']), OrientedBoundingBox.from_dict(second['obb'])) >= self.config.equivalence_min_aabb_iou)

    def _groups(self, ranked):
        # Deterministic complete-link grouping prevents transitive bridge merges.
        groups = []
        for score, candidate in ranked:
            for group in groups:
                if all(self._equivalent(candidate, other[1]) for other in group):
                    group.append((score, candidate))
                    break
            else:
                groups.append([(score, candidate)])
        return groups

    @staticmethod
    def _empty(query, reason):
        return GroundingResult(query.query_id, [], {}, 0.0, [], True, reason)

    def ground(self, query: GroundingQuery, mode='filter_with_entity_guard') -> GroundingResult:
        if mode not in self.MODES:
            raise ValueError('unknown relation reasoning mode')
        targets, references = self._matching(query.target), self._matching(query.reference or '')
        if not targets:
            return self._empty(query, 'target_not_found')
        if not references:
            return self._empty(query, 'reference_not_found')
        relation = RELATION_ALIASES.get((query.relation or '').lower())
        if relation is None:
            return self._empty(query, 'unsupported_relation')
        if query.anchor_frame not in self.anchors:
            return self._empty(query, 'anchor_pose_not_found')
        pose = np.asarray(self.anchors[query.anchor_frame], dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-4):
            return self._empty(query, 'invalid_anchor_pose')
        reference_score, reference = references[0]
        targets = [(s, c) for s, c in targets if c['object_id'] != reference['object_id']]
        if not targets:
            return self._empty(query, 'no_distinct_target_reference_pair')
        distances = {}
        for _, target in targets:
            delta = pose[:3, :3].T @ (np.asarray(target['center']) - reference['center'])
            distances[target['object_id']] = RelationGrounder._signed_distance(relation, delta)
        admissible = [(s, c) for s, c in targets if distances[c['object_id']] > self.config.relation_margin]
        ranking = admissible if mode in ('filter_before_selection', 'filter_with_abstention', 'filter_with_entity_guard') and admissible else targets
        score, target = ranking[0]
        reason = None
        if mode != 'no_relation' and distances[target['object_id']] <= self.config.relation_margin:
            reason = 'relation_conflict_or_boundary'
        target_groups = self._groups(ranking)
        reference_groups = self._groups(references)
        target_gap = target_groups[0][0][0] - target_groups[1][0][0] if len(target_groups) > 1 else None
        reference_gap = reference_groups[0][0][0] - reference_groups[1][0][0] if len(reference_groups) > 1 else None
        if mode == 'filter_with_entity_guard' and min(score, reference_score) < self.config.minimum_entity_score:
            reason = 'insufficient_entity_evidence'
        if mode == 'filter_with_abstention':
            # Entity evidence takes precedence over geometry for rejection reasons.
            if min(score, reference_score) < self.config.minimum_entity_score:
                reason = 'insufficient_entity_evidence'
            elif reference_gap is not None and reference_gap < self.config.ambiguity_score_gap:
                reason = 'ambiguous_reference'
            elif reason is None and target_gap is not None and target_gap < self.config.ambiguity_score_gap:
                reason = 'ambiguous_target'
        return GroundingResult(
            query_id=query.query_id,
            ranked_ids=[c['object_id'] for _, c in ranking],
            relation_scores=distances,
            confidence=float(min(score, reference_score)),
            evidence_frames=sorted(set(target['frame_ids'] + reference['frame_ids'])),
            abstain=reason is not None,
            reason=reason,
            explanation={
                'reference_id': reference['object_id'], 'relation': relation,
                'signed_distance': float(distances[target['object_id']]),
                'required_margin': self.config.relation_margin,
                'anchor_frame': query.anchor_frame, 'mode': mode,
                'target_entity_score': float(score), 'reference_entity_score': float(reference_score),
                'target_group_count': len(target_groups), 'reference_group_count': len(reference_groups),
                'target_score_gap': target_gap, 'reference_score_gap': reference_gap,
                'admissible_target_count': len(admissible),
                'confidence_is_calibrated_probability': False,
            },
        )
