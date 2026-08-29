"""D19 one-factor ablations and frozen failure-taxonomy audit."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence
import copy
import json
import re

import numpy as np

from .a2_association import (
    EvidenceAssociationConfig,
    complete_link_clusters,
    evaluate_a2_predictions,
    predict_all_pairs,
)
from .candidate_cache import CandidateOutcomeCache
from .d9_association import (
    ManualInstanceGroup,
    ManualInstanceLabels,
)
from .experiment_protocol import (
    A2_ID,
    D18_EXPERIMENT_ID,
    Q1_ID,
    canonical_sha256,
    load_json,
    replay_query_policy,
    sha256_file,
)
from .q2_sequential import (
    GainBasedSequentialPolicy,
    SequentialSearchConfig,
    sequential_metadata,
)
from .schemas import ObjectObservation


D19_SCHEMA_VERSION = "0.1"
D19_STATUS = "CPU_COMPLETE"
D19_REAL_STATUS = "REAL_ABLATION_PENDING"
Q2_VARIANTS = (
    "base",
    "retrieval_only",
    "no_gain_patience",
)
A2_VARIANTS = (
    "base",
    "without_semantic",
    "without_obb_shape",
    "without_quality",
    "without_complete_link",
)
FAILURE_CATEGORIES = (
    "retrieval",
    "segmentation",
    "lifting",
    "association",
    "relation",
    "abstention",
)
MANIFEST_FIELDS = (
    "schema_version",
    "stage",
    "status",
    "real_ablation_status",
    "repository_base_commit",
    "d18_manifest",
    "d18_manifest_sha256",
    "q2_variants",
    "a2_variants",
    "association_input",
    "failure_taxonomy",
    "historical_success_ablation",
    "label_policy",
    "claims",
    "frozen_at",
)


def validate_ablation_manifest(
    payload: Mapping[str, Any],
    *,
    project_root: str | Path | None = None,
) -> None:
    if set(payload) != set(MANIFEST_FIELDS):
        raise ValueError("D19 ablation manifest fields are not frozen")
    if payload["schema_version"] != D19_SCHEMA_VERSION:
        raise ValueError("unsupported D19 schema")
    if payload["stage"] != "D19-ablation-audit":
        raise ValueError("D19 stage changed")
    if payload["status"] != D19_STATUS:
        raise ValueError("D19 status must remain CPU_COMPLETE")
    if payload["real_ablation_status"] != D19_REAL_STATUS:
        raise ValueError("D19 real-data status changed")
    if re.fullmatch(
        r"[0-9a-f]{40}", str(payload["repository_base_commit"])
    ) is None:
        raise ValueError("D19 base commit must be a Git hash")
    reference = Path(str(payload["d18_manifest"]))
    if reference.is_absolute() or ".." in reference.parts:
        raise ValueError("D19 D18 manifest reference must be repository-relative")
    digest = str(payload["d18_manifest_sha256"])
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("D19 D18 manifest hash must be SHA-256")
    q2 = payload["q2_variants"]
    if not isinstance(q2, list) or tuple(
        item.get("variant_id") for item in q2
    ) != Q2_VARIANTS:
        raise ValueError("D19 Q2 variants changed")
    if [item.get("changed_factor") for item in q2] != [
        None,
        "pose_novelty",
        "gain_patience",
    ]:
        raise ValueError("D19 Q2 one-factor definitions changed")
    a2 = payload["a2_variants"]
    if not isinstance(a2, list) or tuple(
        item.get("variant_id") for item in a2
    ) != A2_VARIANTS:
        raise ValueError("D19 A2 variants changed")
    if [item.get("changed_factor") for item in a2] != [
        None,
        "semantic",
        "obb_shape",
        "quality",
        "complete_link",
    ]:
        raise ValueError("D19 A2 one-factor definitions changed")
    if payload["association_input"] != {
        "q_policy": Q1_ID,
        "budget": 5,
        "outcome_access": "selected_candidates_only",
    }:
        raise ValueError("D19 association input changed")
    if payload["failure_taxonomy"] != list(FAILURE_CATEGORIES):
        raise ValueError("D19 failure taxonomy changed")
    if payload["historical_success_ablation"] != {
        "status": "NOT_IMPLEMENTED",
        "reason": "Q2_has_no_historical_success_candidate_score_feature",
    }:
        raise ValueError("D19 historical-success boundary changed")
    if payload["label_policy"] != {
        "prediction": "FORBIDDEN",
        "evaluation": "SYNTHETIC_LABEL_FILE_ONLY",
    }:
        raise ValueError("D19 label policy changed")
    if payload["claims"] != {
        "synthetic": "correctness_ablation_not_performance",
        "office_loop": "engineering_structure_only",
        "real_data": "REAL_ABLATION_PENDING",
    }:
        raise ValueError("D19 claims changed")
    if project_root is not None:
        root = Path(project_root).resolve()
        if sha256_file(root / reference) != digest:
            raise ValueError("D19 D18 manifest hash mismatch")


@dataclass(frozen=True)
class Q2AblationResult:
    status: str
    selected_frames: tuple[str, ...]
    stop_reason: str
    sam_calls: int
    observations: tuple[ObjectObservation, ...]
    lifted_instances: int
    rejected_instances: int


def _normalized_retrieval(
    metadata: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    scores = np.asarray(
        [float(item["retrieval_score"]) for item in metadata],
        dtype=np.float64,
    )
    low, high = float(scores.min()), float(scores.max())
    if np.isclose(low, high):
        return {str(item["frame_id"]): 1.0 for item in metadata}
    return {
        str(item["frame_id"]): float((score - low) / (high - low))
        for item, score in zip(metadata, scores)
    }


def q2_variant_config(variant_id: str) -> dict[str, Any]:
    config = SequentialSearchConfig().to_dict()
    if variant_id == "base":
        return config
    if variant_id == "retrieval_only":
        config.update({
            "retrieval_weight": 1.0,
            "novelty_weight": 0.0,
            "later_score": "1.0*normalized_retrieval",
            "ablation": "pose_novelty_removed",
        })
        return config
    if variant_id == "no_gain_patience":
        config.update({
            "low_gain_patience": None,
            "ablation": "gain_patience_disabled",
        })
        return config
    raise ValueError(f"unsupported Q2 ablation: {variant_id}")


def run_q2_ablation(
    cache_payload: Mapping[str, Any],
    variant_id: str,
) -> Q2AblationResult:
    CandidateOutcomeCache.from_dict(cache_payload)
    if variant_id not in Q2_VARIANTS:
        raise ValueError(f"unsupported Q2 ablation: {variant_id}")
    config = SequentialSearchConfig()
    policy = GainBasedSequentialPolicy(config)
    metadata = sequential_metadata(cache_payload)
    normalized = _normalized_retrieval(metadata)
    candidates = {
        str(item["frame_id"]): item for item in cache_payload["candidates"]
    }
    selected: list[str] = []
    observations: list[ObjectObservation] = []
    observed_ids: set[str] = set()
    low_gain_streak = 0
    sam_calls = lifted = rejected = 0
    status = "PASS"
    stop_reason = "max_budget_reached"

    while len(selected) < config.max_budget:
        if variant_id == "retrieval_only":
            scores = [
                {
                    "frame_id": str(item["frame_id"]),
                    "source_rank": int(item["rank"]),
                    "policy_score": normalized[str(item["frame_id"])],
                }
                for item in metadata
                if str(item["frame_id"]) not in selected
            ]
            scores.sort(
                key=lambda item: (
                    -item["policy_score"],
                    item["source_rank"],
                    item["frame_id"],
                )
            )
        else:
            scores = policy.score_candidates(metadata, selected)
        if not scores:
            stop_reason = "candidate_exhausted"
            break
        frame_id = str(scores[0]["frame_id"])
        selected.append(frame_id)
        candidate = candidates[frame_id]
        if candidate["outcome_status"] != "available":
            status = "BLOCKED_MISSING_OUTCOME"
            stop_reason = "BLOCKED_MISSING_OUTCOME"
            break
        outcome = candidate["outcome"]
        current = [
            ObjectObservation.from_dict(item)
            for item in outcome["observations"]
        ]
        new_ids = {item.obs_id for item in current} - observed_ids
        observed_ids.update(item.obs_id for item in current)
        observations.extend(current)
        gain = len(new_ids)
        low_gain_streak = (
            low_gain_streak + 1
            if gain < config.low_gain_threshold
            else 0
        )
        sam_calls += int(candidate["cost"]["sam_calls"])
        lifted += int(outcome["lifted_instances"])
        rejected += int(outcome["rejected_instances"])
        if (
            variant_id != "no_gain_patience"
            and low_gain_streak >= config.low_gain_patience
        ):
            stop_reason = "two_consecutive_low_gain"
            break
        if len(selected) == len(metadata):
            stop_reason = "candidate_exhausted"
            break

    return Q2AblationResult(
        status=status,
        selected_frames=tuple(selected),
        stop_reason=stop_reason,
        sam_calls=sam_calls,
        observations=tuple(observations),
        lifted_instances=lifted,
        rejected_instances=rejected,
    )


def _renormalized_weights(removed: str) -> dict[str, float]:
    weights = {
        "semantic": 0.25,
        "center": 0.25,
        "overlap": 0.20,
        "obb_shape": 0.15,
        "quality": 0.15,
    }
    weights[removed] = 0.0
    total = sum(weights.values())
    return {name: value / total for name, value in weights.items()}


def a2_variant_config(
    variant_id: str,
) -> tuple[EvidenceAssociationConfig, dict[str, Any]]:
    base = EvidenceAssociationConfig()
    if variant_id == "base":
        return base, base.to_dict()
    factor = {
        "without_semantic": "semantic",
        "without_obb_shape": "obb_shape",
        "without_quality": "quality",
    }.get(variant_id)
    if factor is not None:
        weights = _renormalized_weights(factor)
        config = EvidenceAssociationConfig(
            semantic_threshold=(
                0.0 if factor == "semantic" else base.semantic_threshold
            ),
            min_observation_quality=(
                0.0
                if factor == "quality"
                else base.min_observation_quality
            ),
            center_distance_threshold=base.center_distance_threshold,
            min_overlap_iou=base.min_overlap_iou,
            semantic_weight=weights["semantic"],
            center_weight=weights["center"],
            overlap_weight=weights["overlap"],
            obb_shape_weight=weights["obb_shape"],
            quality_weight=weights["quality"],
            min_distinct_frames=base.min_distinct_frames,
        )
        payload = config.to_dict()
        payload["ablation"] = f"{factor}_removed"
        return config, payload
    if variant_id == "without_complete_link":
        payload = copy.deepcopy(base.to_dict())
        payload["cluster_rule"] = "single_link_any_gate_pair"
        payload["ablation"] = "complete_link_removed"
        return base, payload
    raise ValueError(f"unsupported A2 ablation: {variant_id}")


def _single_link(
    observations: Sequence[ObjectObservation],
    pairs: Sequence[Any],
) -> tuple[list[list[ObjectObservation]], list[Any]]:
    ordered = sorted(observations, key=lambda item: item.obs_id)
    parent = {item.obs_id: item.obs_id for item in ordered}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for pair in sorted(pairs, key=lambda item: (item.obs_id_a, item.obs_id_b)):
        if pair.gate_pass:
            first, second = find(pair.obs_id_a), find(pair.obs_id_b)
            if first != second:
                parent[second] = first
    groups: dict[str, list[ObjectObservation]] = {}
    for observation in ordered:
        groups.setdefault(find(observation.obs_id), []).append(observation)
    components = sorted(groups.values(), key=lambda items: items[0].obs_id)
    cluster_by_obs = {
        observation.obs_id: index
        for index, component in enumerate(components)
        for observation in component
    }
    finalized = [
        replace(
            pair,
            predicted_same=(
                cluster_by_obs[pair.obs_id_a]
                == cluster_by_obs[pair.obs_id_b]
            ),
            reasons=tuple((*pair.reasons, "single_link_ablation")),
        )
        for pair in sorted(
            pairs, key=lambda item: (item.obs_id_a, item.obs_id_b)
        )
    ]
    return components, finalized


def run_a2_ablation(
    observations: Sequence[ObjectObservation],
    variant_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if variant_id not in A2_VARIANTS:
        raise ValueError(f"unsupported A2 ablation: {variant_id}")
    config, config_payload = a2_variant_config(variant_id)
    raw_pairs = predict_all_pairs(observations, config)
    if variant_id == "without_complete_link":
        clusters, pairs = _single_link(observations, raw_pairs)
    else:
        clusters, pairs, _ = complete_link_clusters(
            observations, raw_pairs
        )
    promoted = sum(
        len({item.frame_id for item in cluster})
        >= config.min_distinct_frames
        for cluster in clusters
    )
    pair_payload = [item.to_dict() for item in pairs]
    result = {
        "variant_id": variant_id,
        "changed_factor": {
            "base": None,
            "without_semantic": "semantic",
            "without_obb_shape": "obb_shape",
            "without_quality": "quality",
            "without_complete_link": "complete_link",
        }[variant_id],
        "base_config_sha256": canonical_sha256(
            EvidenceAssociationConfig().to_dict()
        ),
        "variant_config_sha256": canonical_sha256(config_payload),
        "config": config_payload,
        "input_observations": len(observations),
        "pair_count": len(pair_payload),
        "cluster_count": len(clusters),
        "promoted_clusters": promoted,
        "predicted_match_pairs": sum(
            bool(item["predicted_same"]) for item in pair_payload
        ),
        "pair_prediction_sha256": canonical_sha256(pair_payload),
    }
    return result, pair_payload


def run_ablation_prediction(
    manifest: Mapping[str, Any],
    cache_payload: Mapping[str, Any],
    *,
    source_id: str,
    cache_ref: str,
    cache_sha256: str,
    manifest_ref: str,
    manifest_sha256: str,
    created_at: str,
) -> dict[str, Any]:
    validate_ablation_manifest(manifest)
    CandidateOutcomeCache.from_dict(cache_payload)
    q2_rows = []
    base_q2_hash = canonical_sha256(q2_variant_config("base"))
    for spec in manifest["q2_variants"]:
        variant_id = str(spec["variant_id"])
        result = run_q2_ablation(cache_payload, variant_id)
        config = q2_variant_config(variant_id)
        q2_rows.append({
            "variant_id": variant_id,
            "changed_factor": spec["changed_factor"],
            "base_config_sha256": base_q2_hash,
            "variant_config_sha256": canonical_sha256(config),
            "config": config,
            "status": result.status,
            "selected_frames": list(result.selected_frames),
            "selected_count": len(result.selected_frames),
            "stop_reason": result.stop_reason,
            "sam_calls": result.sam_calls,
            "observation_count": len(result.observations),
            "lifted_instances": result.lifted_instances,
            "rejected_instances": result.rejected_instances,
        })

    selection = replay_query_policy(cache_payload, Q1_ID)
    a2_rows = []
    if selection.status == "PASS":
        for spec in manifest["a2_variants"]:
            row, _ = run_a2_ablation(
                selection.observations, str(spec["variant_id"])
            )
            a2_rows.append(row)
    overall = (
        "PASS"
        if all(row["status"] == "PASS" for row in q2_rows)
        else "PASS_WITH_BLOCKED_ROWS"
    )
    return {
        "schema_version": D19_SCHEMA_VERSION,
        "status": overall,
        "stage": "D19-ablation-prediction",
        "source_id": source_id,
        "scene_id": cache_payload["scene_id"],
        "query_id": cache_payload["query_id"],
        "query_text": cache_payload["query_text"],
        "source": {
            "manifest": manifest_ref,
            "manifest_sha256": manifest_sha256,
            "candidate_cache": cache_ref,
            "candidate_cache_sha256": cache_sha256,
        },
        "q2_ablations": q2_rows,
        "a2_ablations": a2_rows,
        "historical_success_ablation": manifest[
            "historical_success_ablation"
        ],
        "failure_audit_readiness": {
            "taxonomy": list(FAILURE_CATEGORIES),
            "status": (
                "SYNTHETIC_EVALUATOR_AVAILABLE"
                if source_id == "synthetic-correctness"
                else "UNLABELLED_ENGINEERING_ONLY"
            ),
            "category_counts": None,
        },
        "performance_claim": None,
        "created_at": created_at,
    }


def _subset_labels(
    labels: ManualInstanceLabels,
    observations: Sequence[ObjectObservation],
) -> ManualInstanceLabels:
    selected = {item.obs_id for item in observations}
    groups = tuple(
        ManualInstanceGroup(
            group.instance_id,
            tuple(
                obs_id
                for obs_id in group.observation_ids
                if obs_id in selected
            ),
        )
        for group in labels.instance_groups
        if any(obs_id in selected for obs_id in group.observation_ids)
    )
    return ManualInstanceLabels(
        scene_id=labels.scene_id,
        query=labels.query,
        annotation_method=labels.annotation_method,
        notes=labels.notes,
        instance_groups=groups,
    )


def _q2_label_metrics(
    result: Q2AblationResult,
    labels: ManualInstanceLabels,
) -> dict[str, Any]:
    selected_ids = {item.obs_id for item in result.observations}
    groups = {
        group.instance_id: set(group.observation_ids)
        for group in labels.instance_groups
    }
    observed = {
        instance_id
        for instance_id, ids in groups.items()
        if ids & selected_ids
    }
    cross_frame = {
        instance_id
        for instance_id, ids in groups.items()
        if len(ids & selected_ids) >= 2
    }
    return {
        "instance_count": len(groups),
        "observed_instance_recall": (
            len(observed) / len(groups) if groups else 0.0
        ),
        "cross_frame_instance_recall": (
            len(cross_frame) / len(groups) if groups else 0.0
        ),
        "selected_observations": len(selected_ids),
        "sam_calls": result.sam_calls,
    }


def evaluate_synthetic_ablations(
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
    if prediction["source_id"] != "synthetic-correctness":
        raise ValueError("D19 labelled evaluation is synthetic-only")
    q2_rows = []
    for row in prediction["q2_ablations"]:
        result = run_q2_ablation(cache_payload, str(row["variant_id"]))
        q2_rows.append({
            "variant_id": row["variant_id"],
            "changed_factor": row["changed_factor"],
            "metrics": _q2_label_metrics(result, labels),
        })
    selection = replay_query_policy(cache_payload, Q1_ID)
    subset = _subset_labels(labels, selection.observations)
    a2_rows = []
    base_failures = None
    for row in prediction["a2_ablations"]:
        _, pairs = run_a2_ablation(
            selection.observations, str(row["variant_id"])
        )
        evaluated = evaluate_a2_predictions(
            selection.observations, pairs, subset
        )
        a2_rows.append({
            "variant_id": row["variant_id"],
            "changed_factor": row["changed_factor"],
            "metrics": evaluated["metrics"],
            "failure_count": len(evaluated["failure_cases"]),
        })
        if row["variant_id"] == "base":
            base_failures = len(evaluated["failure_cases"])
    failed = int(bool(base_failures))
    counts = {category: 0 for category in FAILURE_CATEGORIES}
    counts["association"] = failed
    failure_audit = {
        "frozen_query_ids": [cache_payload["query_id"]],
        "denominator_query_count": 1,
        "failed_query_count": failed,
        "category_counts": counts,
        "uncategorized_failed_queries": 0,
        "classification_rule": (
            "base_A2_has_any_false_positive_or_false_negative"
        ),
    }
    return {
        "schema_version": D19_SCHEMA_VERSION,
        "status": "PASS",
        "stage": "D19-synthetic-evaluation",
        "scope": "synthetic_correctness_ablation_not_performance",
        "source": {
            "prediction": prediction_ref,
            "prediction_sha256": prediction_sha256,
            "labels": labels_ref,
            "labels_sha256": labels_sha256,
        },
        "q2_ablations": q2_rows,
        "a2_ablations": a2_rows,
        "failure_audit": failure_audit,
        "real_ablation_status": D19_REAL_STATUS,
        "performance_claim": None,
        "created_at": created_at,
    }


def validate_prediction_replay(
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    cache_payload: Mapping[str, Any],
) -> None:
    expected = run_ablation_prediction(
        manifest,
        cache_payload,
        source_id=str(payload["source_id"]),
        cache_ref=str(payload["source"]["candidate_cache"]),
        cache_sha256=str(payload["source"]["candidate_cache_sha256"]),
        manifest_ref=str(payload["source"]["manifest"]),
        manifest_sha256=str(payload["source"]["manifest_sha256"]),
        created_at=str(payload["created_at"]),
    )
    if dict(payload) != expected:
        raise ValueError("D19 prediction differs from deterministic replay")
    serialized = json.dumps(payload, sort_keys=True).lower()
    for forbidden in (
        '"labels"',
        '"ground_truth"',
        '"expected_same"',
        '"metrics"',
        '"answer"',
    ):
        if forbidden in serialized:
            raise ValueError(f"D19 prediction leaks evaluator field: {forbidden}")
