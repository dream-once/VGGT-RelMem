"""D15 gain-based sequential search with reveal-after-selection traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import copy

import numpy as np

from .candidate_cache import CandidateOutcomeCache
from .q1_fixed_topk import Q1_POLICY_ID


Q2_SCHEMA_VERSION = "0.1"
Q2_POLICY_ID = "Q2-gain-based-sequential-search"
Q2_METADATA_FIELDS = (
    "rank",
    "frame_id",
    "geometry_index",
    "camera_center",
    "view_direction",
    "retrieval_score",
    "retrieval_cosine",
)


@dataclass(frozen=True)
class SequentialSearchConfig:
    max_budget: int = 5
    low_gain_threshold: int = 1
    low_gain_patience: int = 2
    retrieval_weight: float = 0.65
    novelty_weight: float = 0.35
    translation_scale: float = 0.15
    view_angle_scale_deg: float = 3.0

    def __post_init__(self) -> None:
        if self.max_budget < 1:
            raise ValueError("max_budget must be positive")
        if self.low_gain_threshold < 0 or self.low_gain_patience < 1:
            raise ValueError("low-gain stop parameters are invalid")
        if not np.isclose(self.retrieval_weight + self.novelty_weight, 1.0):
            raise ValueError("policy weights must sum to one")
        if min(
            self.retrieval_weight,
            self.novelty_weight,
            self.translation_scale,
            self.view_angle_scale_deg,
        ) <= 0.0:
            raise ValueError("policy weights and pose scales must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_budget": self.max_budget,
            "low_gain_threshold": self.low_gain_threshold,
            "low_gain_patience": self.low_gain_patience,
            "retrieval_weight": self.retrieval_weight,
            "novelty_weight": self.novelty_weight,
            "translation_scale": self.translation_scale,
            "view_angle_scale_deg": self.view_angle_scale_deg,
            "retrieval_normalization": "minmax_over_candidate_universe",
            "all_equal_retrieval_value": 1.0,
            "pose_novelty": "0.5*clip(min_translation/0.15)+0.5*clip(min_view_angle/3deg)",
            "first_step": "retrieval_score_only",
            "later_score": "0.65*normalized_retrieval+0.35*pose_novelty",
            "selection_input": "candidate_metadata_only",
            "outcome_access": "selected_candidate_only_after_selection",
            "observed_gain": "new_3d_observation_ids",
        }


def sequential_metadata(
    cache_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    CandidateOutcomeCache.from_dict(cache_payload)
    return [
        {field: copy.deepcopy(candidate[field]) for field in Q2_METADATA_FIELDS}
        for candidate in cache_payload["candidates"]
    ]


def _unit_direction(value: Any) -> np.ndarray:
    direction = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if direction.shape != (3,) or norm < 1e-12:
        raise ValueError("view direction must be a nonzero length-3 vector")
    return direction / norm


class GainBasedSequentialPolicy:
    """Score visible metadata without inspecting any candidate outcome."""

    def __init__(self, config: SequentialSearchConfig | None = None) -> None:
        self.config = config or SequentialSearchConfig()

    @staticmethod
    def _normalized_retrieval(
        metadata: Sequence[Mapping[str, Any]],
    ) -> dict[str, float]:
        scores = np.asarray(
            [float(row["retrieval_score"]) for row in metadata],
            dtype=np.float64,
        )
        low, high = float(np.min(scores)), float(np.max(scores))
        if np.isclose(low, high):
            return {str(row["frame_id"]): 1.0 for row in metadata}
        return {
            str(row["frame_id"]): float((score - low) / (high - low))
            for row, score in zip(metadata, scores)
        }

    def score_candidates(
        self,
        metadata: Sequence[Mapping[str, Any]],
        selected_frame_ids: Sequence[str],
    ) -> list[dict[str, Any]]:
        if not metadata:
            return []
        by_id = {str(row["frame_id"]): row for row in metadata}
        if len(by_id) != len(metadata):
            raise ValueError("sequential metadata frame ids must be unique")
        unknown = set(selected_frame_ids) - set(by_id)
        if unknown:
            raise ValueError("selected frame is outside the candidate universe")
        normalized = self._normalized_retrieval(metadata)
        selected = [by_id[item] for item in selected_frame_ids]
        records: list[dict[str, Any]] = []
        for row in metadata:
            frame_id = str(row["frame_id"])
            if frame_id in selected_frame_ids:
                continue
            if not selected:
                min_translation = None
                translation_novelty = None
                min_angle = None
                angle_novelty = None
                pose_novelty = None
                policy_score = normalized[frame_id]
            else:
                center = np.asarray(row["camera_center"], dtype=np.float64)
                direction = _unit_direction(row["view_direction"])
                distances = [
                    float(np.linalg.norm(
                        center - np.asarray(item["camera_center"], dtype=np.float64)
                    ))
                    for item in selected
                ]
                angles = [
                    float(np.degrees(np.arccos(np.clip(
                        np.dot(direction, _unit_direction(item["view_direction"])),
                        -1.0,
                        1.0,
                    ))))
                    for item in selected
                ]
                min_translation = min(distances)
                min_angle = min(angles)
                translation_novelty = min(
                    min_translation / self.config.translation_scale,
                    1.0,
                )
                angle_novelty = min(
                    min_angle / self.config.view_angle_scale_deg,
                    1.0,
                )
                pose_novelty = 0.5 * (
                    translation_novelty + angle_novelty
                )
                policy_score = (
                    self.config.retrieval_weight * normalized[frame_id]
                    + self.config.novelty_weight * pose_novelty
                )
            records.append({
                "source_rank": int(row["rank"]),
                "frame_id": frame_id,
                "geometry_index": int(row["geometry_index"]),
                "retrieval_score": float(row["retrieval_score"]),
                "normalized_retrieval": normalized[frame_id],
                "min_translation": min_translation,
                "translation_novelty": translation_novelty,
                "min_view_angle_deg": min_angle,
                "view_angle_novelty": angle_novelty,
                "pose_novelty": pose_novelty,
                "policy_score": policy_score,
            })
        records.sort(key=lambda item: (
            -item["policy_score"],
            item["source_rank"],
            item["frame_id"],
        ))
        return records


def run_sequential_search(
    cache_payload: Mapping[str, Any],
    *,
    cache_ref: str,
    cache_sha256: str,
    created_at: str,
    config: SequentialSearchConfig | None = None,
) -> dict[str, Any]:
    CandidateOutcomeCache.from_dict(cache_payload)
    policy = GainBasedSequentialPolicy(config)
    metadata = sequential_metadata(cache_payload)
    candidates = {
        str(item["frame_id"]): item for item in cache_payload["candidates"]
    }
    selected_ids: list[str] = []
    observed_ids: set[str] = set()
    low_gain_streak = 0
    cumulative = {
        "sam_calls": 0,
        "sam_instances": 0,
        "lifted_instances": 0,
        "rejected_instances": 0,
        "observed_gain": 0,
    }
    steps: list[dict[str, Any]] = []
    stop_reason = None
    status = "PASS"
    while len(selected_ids) < policy.config.max_budget:
        visible_scores = policy.score_candidates(metadata, selected_ids)
        if not visible_scores:
            stop_reason = "candidate_exhausted"
            break
        selected_score = visible_scores[0]
        frame_id = str(selected_score["frame_id"])
        candidate = candidates[frame_id]
        selected_before = list(selected_ids)
        selected_ids.append(frame_id)
        outcome_status = str(candidate["outcome_status"])
        if outcome_status != "available":
            revealed = {
                "outcome_status": outcome_status,
                "failure_reason": candidate["failure_reason"],
                "sam_instances": None,
                "lifted_instances": None,
                "rejected_instances": None,
                "new_observation_ids": [],
                "observed_gain": None,
            }
            stop_reason = "BLOCKED_MISSING_OUTCOME"
            status = "BLOCKED_MISSING_OUTCOME"
        else:
            outcome = candidate["outcome"]
            observation_ids = [
                str(item["obs_id"]) for item in outcome["observations"]
            ]
            new_ids = sorted(set(observation_ids) - observed_ids)
            observed_ids.update(observation_ids)
            gain = len(new_ids)
            low_gain_streak = (
                low_gain_streak + 1
                if gain < policy.config.low_gain_threshold
                else 0
            )
            revealed = {
                "outcome_status": "available",
                "failure_reason": None,
                "sam_instances": int(outcome["sam_instances"]),
                "lifted_instances": int(outcome["lifted_instances"]),
                "rejected_instances": int(outcome["rejected_instances"]),
                "new_observation_ids": new_ids,
                "observed_gain": gain,
            }
            cumulative["sam_calls"] += int(candidate["cost"]["sam_calls"])
            cumulative["sam_instances"] += int(outcome["sam_instances"])
            cumulative["lifted_instances"] += int(outcome["lifted_instances"])
            cumulative["rejected_instances"] += int(outcome["rejected_instances"])
            cumulative["observed_gain"] += gain
            if low_gain_streak >= policy.config.low_gain_patience:
                stop_reason = "two_consecutive_low_gain"
            elif len(selected_ids) >= policy.config.max_budget:
                stop_reason = "max_budget_reached"
            elif len(selected_ids) == len(metadata):
                stop_reason = "candidate_exhausted"
        steps.append({
            "step": len(selected_ids),
            "selected_before": selected_before,
            "visible_candidate_scores": visible_scores,
            "selected": {
                **copy.deepcopy(selected_score),
                "selection_reason": (
                    "retrieval_score_only"
                    if len(selected_ids) == 1
                    else "retrieval_plus_pose_novelty"
                ),
            },
            "revealed_outcome": revealed,
            "cumulative_cost": copy.deepcopy(cumulative),
            "low_gain_streak": low_gain_streak,
            "stop_decision": stop_reason,
        })
        if stop_reason is not None:
            break
    q0_frame = str(cache_payload["candidates"][0]["frame_id"])
    return {
        "schema_version": Q2_SCHEMA_VERSION,
        "status": status,
        "stage": "D15-policy-trace",
        "policy_id": Q2_POLICY_ID,
        "scene_id": cache_payload["scene_id"],
        "query_id": cache_payload["query_id"],
        "query_text": cache_payload["query_text"],
        "development_replay": True,
        "source": {
            "candidate_cache": cache_ref,
            "candidate_cache_sha256": cache_sha256,
            "cache_materialization_status": cache_payload[
                "materialization_status"
            ],
        },
        "config": policy.config.to_dict(),
        "steps": steps,
        "summary": {
            "selected_frames": selected_ids,
            "selected_count": len(selected_ids),
            "q0_top1_frame": q0_frame,
            "budget1_matches_q0": bool(selected_ids and selected_ids[0] == q0_frame),
            "stop_reason": stop_reason,
            "cumulative_cost": cumulative,
            "performance_claim": None,
            "gpu_acceptance": "GPU_ACCEPTANCE_PENDING",
        },
        "created_at": created_at,
    }


def _prefix_cost(trace: Mapping[str, Any], budget: int) -> dict[str, Any]:
    steps = trace["steps"]
    if not steps:
        return {
            "selected_frames": [],
            "selected_count": 0,
            "sam_calls": 0,
            "lifted_instances": 0,
            "rejected_instances": 0,
        }
    index = min(budget, len(steps)) - 1
    step = steps[index]
    cumulative = step["cumulative_cost"]
    return {
        "selected_frames": [
            item["selected"]["frame_id"] for item in steps[: index + 1]
        ],
        "selected_count": index + 1,
        "sam_calls": int(cumulative["sam_calls"]),
        "lifted_instances": int(cumulative["lifted_instances"]),
        "rejected_instances": int(cumulative["rejected_instances"]),
    }


def build_engineering_comparison(
    q1_prediction: Mapping[str, Any],
    q2_trace: Mapping[str, Any],
    *,
    q1_ref: str,
    q1_sha256: str,
    q2_ref: str,
    q2_sha256: str,
    created_at: str,
) -> dict[str, Any]:
    if q1_prediction["policy_id"] != Q1_POLICY_ID:
        raise ValueError("comparison requires the frozen Q1 prediction")
    if q2_trace["policy_id"] != Q2_POLICY_ID or q2_trace["status"] != "PASS":
        raise ValueError("comparison requires a complete Q2 trace")
    if q1_prediction["scene_id"] != q2_trace["scene_id"]:
        raise ValueError("Q1/Q2 comparison scene mismatch")
    curves = {
        int(item["requested_budget"]): item for item in q1_prediction["curves"]
    }
    q0 = curves[1]
    rows: list[dict[str, Any]] = []
    for budget in (1, 3, 5):
        q1 = curves[budget]
        rows.append({
            "budget": budget,
            "q0_fixed_top1": {
                "selected_frames": [q0["selected_frames"][0]["frame_id"]],
                "selected_count": 1,
                "sam_calls": int(q0["sam_calls"]),
                "lifted_instances": int(q0["lifted_instances"]),
                "rejected_instances": int(q0["rejected_instances"]),
            },
            "q1_fixed_topk": {
                "selected_frames": [
                    item["frame_id"] for item in q1["selected_frames"]
                ],
                "selected_count": int(q1["selected_count"]),
                "sam_calls": int(q1["sam_calls"]),
                "lifted_instances": int(q1["lifted_instances"]),
                "rejected_instances": int(q1["rejected_instances"]),
            },
            "q2_sequential_prefix": _prefix_cost(q2_trace, budget),
        })
    return {
        "schema_version": Q2_SCHEMA_VERSION,
        "status": "PASS",
        "stage": "D15-engineering-comparison",
        "scene_id": q2_trace["scene_id"],
        "query_id": q2_trace["query_id"],
        "query_text": q2_trace["query_text"],
        "development_replay": True,
        "comparison_scope": "synthetic_engineering_counts_not_performance",
        "source": {
            "q1_prediction": q1_ref,
            "q1_prediction_sha256": q1_sha256,
            "q2_trace": q2_ref,
            "q2_trace_sha256": q2_sha256,
        },
        "rows": rows,
        "created_at": created_at,
    }


def validate_trace_payload(
    payload: Mapping[str, Any],
    cache_payload: Mapping[str, Any],
) -> None:
    expected = run_sequential_search(
        cache_payload,
        cache_ref=str(payload["source"]["candidate_cache"]),
        cache_sha256=str(payload["source"]["candidate_cache_sha256"]),
        created_at=str(payload["created_at"]),
        config=SequentialSearchConfig(),
    )
    if dict(payload) != expected:
        raise ValueError("Q2 trace differs from deterministic replay")
    if not payload["summary"]["budget1_matches_q0"]:
        raise ValueError("Q2 budget=1 must match Q0")


def validate_comparison_payload(
    payload: Mapping[str, Any],
    q1_prediction: Mapping[str, Any],
    q2_trace: Mapping[str, Any],
) -> None:
    expected = build_engineering_comparison(
        q1_prediction,
        q2_trace,
        q1_ref=str(payload["source"]["q1_prediction"]),
        q1_sha256=str(payload["source"]["q1_prediction_sha256"]),
        q2_ref=str(payload["source"]["q2_trace"]),
        q2_sha256=str(payload["source"]["q2_trace_sha256"]),
        created_at=str(payload["created_at"]),
    )
    if dict(payload) != expected:
        raise ValueError("D15 comparison differs from deterministic replay")
