"""D14 fixed-Top-K metadata selection and outcome replay contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import copy

import numpy as np

from .candidate_cache import CandidateOutcomeCache
from .d9_association import ManualInstanceLabels
from .retrieval import FrameCandidate, RetrievalConfig, TopKFrameRetriever


Q1_SCHEMA_VERSION = "0.1"
Q1_POLICY_ID = "Q1-fixed-topk-hybrid"
Q1_BUDGETS = (1, 3, 5)
Q1_METADATA_FIELDS = (
    "rank",
    "frame_id",
    "geometry_index",
    "camera_center",
    "view_direction",
    "retrieval_score",
    "retrieval_cosine",
)


@dataclass(frozen=True)
class FixedTopKConfig:
    budgets: tuple[int, ...] = Q1_BUDGETS
    min_frame_gap: int = 2
    min_camera_distance: float = 0.15
    min_view_angle_deg: float = 3.0

    def __post_init__(self) -> None:
        if not self.budgets or tuple(sorted(set(self.budgets))) != self.budgets:
            raise ValueError("budgets must be unique and increasing")
        if self.budgets[0] != 1 or any(value < 1 for value in self.budgets):
            raise ValueError("budgets must start at one and remain positive")
        RetrievalConfig(
            top_k=max(self.budgets),
            redundancy="hybrid",
            min_frame_gap=self.min_frame_gap,
            min_camera_distance=self.min_camera_distance,
            min_view_angle_deg=self.min_view_angle_deg,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "budgets": list(self.budgets),
            "redundancy": "hybrid",
            "min_frame_gap": self.min_frame_gap,
            "min_camera_distance": self.min_camera_distance,
            "min_view_angle_deg": self.min_view_angle_deg,
            "selection_input": "candidate_metadata_only",
            "outcome_access": "after_selection_only",
        }


def candidate_metadata(cache_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project a cache to the fields a Q1 policy is allowed to inspect."""

    CandidateOutcomeCache.from_dict(cache_payload)
    return [
        {field: copy.deepcopy(candidate[field]) for field in Q1_METADATA_FIELDS}
        for candidate in cache_payload["candidates"]
    ]


class FixedTopKPolicy:
    """Apply the frozen D5 hybrid selector without reading cached outcomes."""

    def __init__(self, config: FixedTopKConfig | None = None) -> None:
        self.config = config or FixedTopKConfig()

    def select(
        self,
        metadata: Sequence[Mapping[str, Any]],
        budget: int,
    ) -> list[dict[str, Any]]:
        if budget not in self.config.budgets:
            raise ValueError("budget is not part of the frozen Q1 curve")
        candidates = [
            FrameCandidate(
                frame_id=str(row["frame_id"]),
                score=float(row["retrieval_score"]),
                index=int(row["geometry_index"]),
                camera_center=np.asarray(row["camera_center"], dtype=float),
                view_direction=np.asarray(row["view_direction"], dtype=float),
                metadata={"rank": int(row["rank"])},
            )
            for row in metadata
        ]
        selector = TopKFrameRetriever(RetrievalConfig(
            top_k=budget,
            redundancy="hybrid",
            min_frame_gap=self.config.min_frame_gap,
            min_camera_distance=self.config.min_camera_distance,
            min_view_angle_deg=self.config.min_view_angle_deg,
        ))
        selected = selector.retrieve(candidates)
        return [
            {
                "selection_rank": index,
                "source_rank": int(item.metadata["rank"]),
                "frame_id": item.frame_id,
                "geometry_index": int(item.index),
                "retrieval_score": item.score,
            }
            for index, item in enumerate(selected, start=1)
        ]


def _curve(
    cache_payload: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    requested_budget: int,
) -> dict[str, Any]:
    candidates = {
        str(item["frame_id"]): item for item in cache_payload["candidates"]
    }
    selected_frames: list[dict[str, Any]] = []
    sam_calls = sam_instances = lifted = rejected = 0
    evidence_frames: list[str] = []
    status = "PASS"
    reason = None
    for selection in selected:
        candidate = candidates[str(selection["frame_id"])]
        if candidate["outcome_status"] != "available":
            status = "BLOCKED_MISSING_OUTCOME"
            reason = (
                f"selected candidate {candidate['frame_id']} has status "
                f"{candidate['outcome_status']}"
            )
            selected_frames.append({
                **dict(selection),
                "outcome_status": candidate["outcome_status"],
            })
            break
        outcome = candidate["outcome"]
        selected_frames.append({
            **dict(selection),
            "outcome_status": "available",
        })
        sam_calls += int(candidate["cost"]["sam_calls"])
        sam_instances += int(outcome["sam_instances"])
        lifted += int(outcome["lifted_instances"])
        rejected += int(outcome["rejected_instances"])
        if int(outcome["lifted_instances"]) > 0:
            evidence_frames.append(str(candidate["frame_id"]))
    if status == "PASS" and len(selected) < requested_budget:
        reason = "nonredundant_candidates_exhausted"
    return {
        "requested_budget": requested_budget,
        "selected_count": len(selected_frames),
        "status": status,
        "exhaustion_reason": reason,
        "selected_frames": selected_frames,
        "sam_calls": sam_calls,
        "sam_instances": sam_instances,
        "lifted_instances": lifted,
        "rejected_instances": rejected,
        "evidence_frames": evidence_frames,
        "cost": {
            "sam_calls": sam_calls,
            "runtime_seconds": None,
            "peak_vram_mb": None,
        },
    }


def replay_fixed_topk(
    cache_payload: Mapping[str, Any],
    *,
    cache_ref: str,
    cache_sha256: str,
    created_at: str,
    config: FixedTopKConfig | None = None,
) -> dict[str, Any]:
    """Select from metadata, then reveal only the selected cached outcomes."""

    CandidateOutcomeCache.from_dict(cache_payload)
    policy = FixedTopKPolicy(config)
    metadata = candidate_metadata(cache_payload)
    curves = [
        _curve(cache_payload, policy.select(metadata, budget), budget)
        for budget in policy.config.budgets
    ]
    q0_frame = str(cache_payload["candidates"][0]["frame_id"])
    q0_match = curves[0]["selected_frames"][0]["frame_id"] == q0_frame
    statuses = {curve["status"] for curve in curves}
    status = (
        "BLOCKED_MISSING_OUTCOME"
        if "BLOCKED_MISSING_OUTCOME" in statuses
        else "PASS"
    )
    return {
        "schema_version": Q1_SCHEMA_VERSION,
        "status": status,
        "stage": "D14-prediction",
        "policy_id": Q1_POLICY_ID,
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
        "policy": policy.config.to_dict(),
        "curves": curves,
        "acceptance": {
            "q0_top1_frame": q0_frame,
            "k1_matches_q0": q0_match,
            "selection_reads_outcome": False,
            "gpu_acceptance": "GPU_ACCEPTANCE_PENDING",
        },
        "created_at": created_at,
    }


def evaluate_budget_curve(
    prediction: Mapping[str, Any],
    cache_payload: Mapping[str, Any],
    labels: ManualInstanceLabels,
    *,
    prediction_ref: str,
    prediction_sha256: str,
    labels_ref: str,
    labels_sha256: str,
    created_at: str,
) -> dict[str, Any]:
    """Evaluate a prediction after policy execution; labels never enter Q1."""

    CandidateOutcomeCache.from_dict(cache_payload)
    if labels.scene_id != prediction["scene_id"]:
        raise ValueError("labels scene differs from prediction")
    if labels.query != prediction["query_text"]:
        raise ValueError("labels query differs from prediction")
    observation_frame: dict[str, str] = {}
    frame_observations: dict[str, list[str]] = {}
    for candidate in cache_payload["candidates"]:
        if candidate["outcome_status"] != "available":
            continue
        ids = [
            str(item["obs_id"])
            for item in candidate["outcome"]["observations"]
        ]
        frame_observations[str(candidate["frame_id"])] = ids
        observation_frame.update({obs_id: str(candidate["frame_id"]) for obs_id in ids})
    groups = {
        group.instance_id: set(group.observation_ids)
        for group in labels.instance_groups
    }
    eligible_cross_frame = {
        instance_id
        for instance_id, ids in groups.items()
        if len({observation_frame[item] for item in ids if item in observation_frame}) >= 2
    }
    details: list[dict[str, Any]] = []
    for curve in prediction["curves"]:
        selected_ids = {
            obs_id
            for row in curve["selected_frames"]
            if row["outcome_status"] == "available"
            for obs_id in frame_observations.get(str(row["frame_id"]), [])
        }
        instance_counts = {
            instance_id: len(ids & selected_ids)
            for instance_id, ids in groups.items()
        }
        observed = {key for key, value in instance_counts.items() if value > 0}
        cross_frame_observed = {
            instance_id
            for instance_id in eligible_cross_frame
            if len({
                observation_frame[item]
                for item in groups[instance_id] & selected_ids
            }) >= 2
        }
        matched_count = sum(instance_counts.values())
        duplicates = sum(max(0, value - 1) for value in instance_counts.values())
        details.append({
            "requested_budget": int(curve["requested_budget"]),
            "sample_count": 1,
            "selected_frame_count": int(curve["selected_count"]),
            "selected_observation_count": len(selected_ids),
            "labelled_observation_count": matched_count,
            "observed_instances": len(observed),
            "total_instances": len(groups),
            "observed_instance_recall": (
                len(observed) / len(groups) if groups else 0.0
            ),
            "cross_frame_instances": len(cross_frame_observed),
            "eligible_cross_frame_instances": len(eligible_cross_frame),
            "cross_frame_instance_recall": (
                len(cross_frame_observed) / len(eligible_cross_frame)
                if eligible_cross_frame else 0.0
            ),
            "duplicate_observations": duplicates,
            "duplicate_observation_rate": (
                duplicates / matched_count if matched_count else 0.0
            ),
        })
    return {
        "schema_version": Q1_SCHEMA_VERSION,
        "status": "PASS" if prediction["status"] == "PASS" else prediction["status"],
        "stage": "D14-evaluation",
        "policy_id": Q1_POLICY_ID,
        "scene_id": prediction["scene_id"],
        "query_id": prediction["query_id"],
        "query_text": prediction["query_text"],
        "development_replay": True,
        "source": {
            "prediction": prediction_ref,
            "prediction_sha256": prediction_sha256,
            "labels": labels_ref,
            "labels_sha256": labels_sha256,
        },
        "metric_definitions": {
            "observed_instance_recall": "labelled instances with >=1 selected observation / all labelled instances",
            "cross_frame_instance_recall": "eligible labelled instances observed in >=2 selected frames / instances visible in >=2 cached frames",
            "duplicate_observation_rate": "label-matched observations beyond the first per observed instance / label-matched observations",
        },
        "budget_details": details,
        "created_at": created_at,
    }


def validate_prediction_payload(
    payload: Mapping[str, Any],
    cache_payload: Mapping[str, Any],
) -> None:
    expected = replay_fixed_topk(
        cache_payload,
        cache_ref=str(payload["source"]["candidate_cache"]),
        cache_sha256=str(payload["source"]["candidate_cache_sha256"]),
        created_at=str(payload["created_at"]),
        config=FixedTopKConfig(),
    )
    if dict(payload) != expected:
        raise ValueError("Q1 prediction differs from deterministic replay")
    if not payload["acceptance"]["k1_matches_q0"]:
        raise ValueError("Q1 K=1 must match Q0")


def validate_evaluation_payload(
    payload: Mapping[str, Any],
    prediction: Mapping[str, Any],
    cache_payload: Mapping[str, Any],
    labels: ManualInstanceLabels,
) -> None:
    expected = evaluate_budget_curve(
        prediction,
        cache_payload,
        labels,
        prediction_ref=str(payload["source"]["prediction"]),
        prediction_sha256=str(payload["source"]["prediction_sha256"]),
        labels_ref=str(payload["source"]["labels"]),
        labels_sha256=str(payload["source"]["labels_sha256"]),
        created_at=str(payload["created_at"]),
    )
    if dict(payload) != expected:
        raise ValueError("Q1 evaluation differs from deterministic replay")
