"""D17 label-separated relation grounding and reliable abstention protocol."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.metrics import (
    area_under_risk_coverage,
    brier_score,
    expected_calibration_error,
    grounding_metrics,
    risk_coverage_curve,
)
from .association import ObjectMemory
from .calibration import AbstentionPolicy, FEATURE_NAMES
from .relations import RelationConfig, RelationGrounder
from .schemas import GroundingQuery


RELATION_PROTOCOL_SCHEMA_VERSION = "0.1"
RELATION_PROTOCOL_ID = "D17-relation-grounding-reliable-abstention"
QUERY_BUNDLE_FIELDS = ("schema_version", "scene_id", "split_role", "queries")
QUERY_FIELDS = (
    "query_id", "target", "relation", "reference", "anchor_frame"
)
LABEL_BUNDLE_FIELDS = ("schema_version", "scene_id", "split_role", "labels")
LABEL_FIELDS = (
    "query_id", "answerable", "answer_object_id",
    "expected_abstain_reason",
)
CALIBRATION_FIELDS = (
    "schema_version", "status", "feature_names", "answer_threshold",
    "fitted_on_split_role", "fit_sample_ids", "fit_allowed_roles",
    "threshold_selection_allowed_roles", "held_out_fit_allowed", "notes",
)
SOURCE_FIELDS = (
    "object_memory", "object_memory_sha256", "anchor_poses",
    "anchor_poses_sha256", "queries", "queries_sha256", "calibration",
    "calibration_sha256",
)
ALLOWED_SPLIT_ROLES = {"synthetic", "calibration", "development", "held-out"}
FORBIDDEN_PREDICTION_KEYS = {
    "answerable", "answer_object_id", "expected_abstain_reason",
    "ground_truth", "metrics", "labels",
}


def _strict_fields(
    payload: Mapping[str, Any], fields: tuple[str, ...], name: str
) -> None:
    if set(payload) != set(fields):
        raise ValueError(f"{name} fields are not frozen")


def _safe_ref(value: Any, name: str) -> str:
    text = str(value)
    parts = text.replace("\\", "/").split("/")
    if not text or text.startswith("/") or ".." in parts:
        raise ValueError(f"{name} must be a safe relative path")
    return text


def _contains_forbidden_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_PREDICTION_KEYS:
                return str(key)
            found = _contains_forbidden_key(item)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _contains_forbidden_key(item)
            if found is not None:
                return found
    return None


def validate_query_bundle(payload: Mapping[str, Any]) -> dict[str, Any]:
    _strict_fields(payload, QUERY_BUNDLE_FIELDS, "relation query bundle")
    if payload["schema_version"] != RELATION_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported relation query schema")
    if payload["split_role"] not in ALLOWED_SPLIT_ROLES:
        raise ValueError("invalid query split role")
    queries = payload["queries"]
    if not isinstance(queries, list) or not queries:
        raise ValueError("relation query bundle must be non-empty")
    seen: set[str] = set()
    for index, query in enumerate(queries):
        if not isinstance(query, Mapping):
            raise ValueError(f"query {index} must be an object")
        _strict_fields(query, QUERY_FIELDS, f"query {index}")
        query_id = str(query["query_id"])
        if not query_id or query_id in seen:
            raise ValueError("query ids must be non-empty and unique")
        seen.add(query_id)
        if not isinstance(query["target"], str) or not query["target"]:
            raise ValueError("query target must be non-empty")
        for field in ("relation", "reference", "anchor_frame"):
            if query[field] is not None and not isinstance(query[field], str):
                raise ValueError(f"query {field} must be a string or null")
    forbidden = _contains_forbidden_key(payload)
    if forbidden is not None:
        raise ValueError(f"prediction query leaks evaluator field: {forbidden}")
    return deepcopy(dict(payload))


def validate_label_bundle(
    payload: Mapping[str, Any],
    query_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    _strict_fields(payload, LABEL_BUNDLE_FIELDS, "relation label bundle")
    if payload["schema_version"] != RELATION_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported relation label schema")
    if (
        payload["scene_id"] != query_bundle["scene_id"]
        or payload["split_role"] != query_bundle["split_role"]
    ):
        raise ValueError("label/query source mismatch")
    labels = payload["labels"]
    if not isinstance(labels, list):
        raise ValueError("labels must be a list")
    expected_ids = [str(item["query_id"]) for item in query_bundle["queries"]]
    actual_ids: list[str] = []
    for index, label in enumerate(labels):
        if not isinstance(label, Mapping):
            raise ValueError(f"label {index} must be an object")
        _strict_fields(label, LABEL_FIELDS, f"label {index}")
        query_id = str(label["query_id"])
        actual_ids.append(query_id)
        if not isinstance(label["answerable"], bool):
            raise ValueError("answerable must be boolean")
        if label["answerable"]:
            if not isinstance(label["answer_object_id"], str) or not label[
                "answer_object_id"
            ]:
                raise ValueError("answerable labels require an object id")
            if label["expected_abstain_reason"] is not None:
                raise ValueError("answerable labels cannot require abstention")
        else:
            if label["answer_object_id"] is not None:
                raise ValueError("negative labels cannot name an answer object")
            if (
                label["expected_abstain_reason"] is not None
                and not isinstance(label["expected_abstain_reason"], str)
            ):
                raise ValueError("expected abstention reason must be text or null")
    if actual_ids != expected_ids:
        raise ValueError("labels must cover queries exactly and in order")
    return deepcopy(dict(payload))


def validate_calibration_manifest(
    payload: Mapping[str, Any],
    *,
    execution_split_role: str,
) -> dict[str, Any]:
    _strict_fields(payload, CALIBRATION_FIELDS, "calibration manifest")
    if payload["schema_version"] != RELATION_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported calibration schema")
    if payload["status"] != "ENGINEERING_DEFAULT_UNCALIBRATED":
        raise ValueError("real fitted calibration is not accepted in D17 CPU scope")
    if payload["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("calibration feature order changed")
    if float(payload["answer_threshold"]) != 0.60:
        raise ValueError("engineering answer threshold must remain 0.60")
    if payload["fitted_on_split_role"] is not None or payload["fit_sample_ids"] != []:
        raise ValueError("uncalibrated manifest cannot claim fitted samples")
    if payload["fit_allowed_roles"] != ["calibration"]:
        raise ValueError("calibrator may only fit on calibration")
    if payload["threshold_selection_allowed_roles"] != ["development"]:
        raise ValueError("threshold selection may only use development")
    if payload["held_out_fit_allowed"] is not False:
        raise ValueError("held-out calibration must be forbidden")
    if execution_split_role not in ALLOWED_SPLIT_ROLES:
        raise ValueError("invalid execution split role")
    return deepcopy(dict(payload))


def validate_prediction_source(payload: Mapping[str, Any]) -> dict[str, Any]:
    _strict_fields(payload, SOURCE_FIELDS, "relation prediction source")
    for field in ("object_memory", "anchor_poses", "queries", "calibration"):
        _safe_ref(payload[field], field)
    for field in (
        "object_memory_sha256", "anchor_poses_sha256", "queries_sha256",
        "calibration_sha256",
    ):
        value = payload[field]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{field} must be SHA-256")
    return deepcopy(dict(payload))


def _validated_anchor_poses(
    anchor_poses: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for frame_id, value in anchor_poses.items():
        pose = np.asarray(value, dtype=np.float64)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError(f"invalid anchor pose: {frame_id}")
        result[str(frame_id)] = pose
    return result


def run_relation_prediction(
    memory: ObjectMemory,
    anchor_poses: Mapping[str, Any],
    query_payload: Mapping[str, Any],
    calibration_payload: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    queries = validate_query_bundle(query_payload)
    calibration = validate_calibration_manifest(
        calibration_payload, execution_split_role=str(queries["split_role"])
    )
    validated_source = validate_prediction_source(source)
    grounder = RelationGrounder(
        memory,
        _validated_anchor_poses(anchor_poses),
        RelationConfig(),
    )
    policy = AbstentionPolicy(float(calibration["answer_threshold"]))
    results: list[dict[str, Any]] = []
    for query in queries["queries"]:
        result = grounder.ground(GroundingQuery.from_dict(query))
        if not result.abstain:
            result = policy.apply(result, result.confidence)
        results.append(result.to_dict())
    reasons: dict[str, int] = {}
    for result in results:
        if result["abstain"]:
            reason = str(result["reason"])
            reasons[reason] = reasons.get(reason, 0) + 1
    payload = {
        "schema_version": RELATION_PROTOCOL_SCHEMA_VERSION,
        "status": "PASS",
        "stage": "D17-prediction",
        "protocol_id": RELATION_PROTOCOL_ID,
        "scene_id": queries["scene_id"],
        "split_role": queries["split_role"],
        "source": validated_source,
        "config": {
            "axis_convention": "+x right, +y up, +z front",
            "anchor_pose": "world_from_anchor",
            "relations": ["left_of", "right_of", "front_of", "behind"],
            "relation_margin": 0.10,
            "ambiguity_margin": 0.05,
            "grounder_confidence_threshold": 0.50,
            "answer_threshold": 0.60,
            "calibration_status": calibration["status"],
        },
        "queries": deepcopy(queries["queries"]),
        "results": results,
        "counts": {
            "queries": len(results),
            "answered": sum(not item["abstain"] for item in results),
            "abstained": sum(item["abstain"] for item in results),
            "abstain_reasons": reasons,
        },
        "acceptance": {
            "prediction_label_free": True,
            "gt_reader_used": False,
            "calibration_fitted": False,
            "gpu_used": False,
        },
        "created_at": str(created_at),
    }
    forbidden = _contains_forbidden_key({
        "queries": payload["queries"],
        "results": payload["results"],
    })
    if forbidden is not None:
        raise ValueError(f"prediction output leaks evaluator field: {forbidden}")
    return payload


def evaluate_relation_prediction(
    prediction: Mapping[str, Any],
    label_payload: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    query_bundle = {
        "schema_version": prediction["schema_version"],
        "scene_id": prediction["scene_id"],
        "split_role": prediction["split_role"],
        "queries": prediction["queries"],
    }
    labels = validate_label_bundle(label_payload, query_bundle)
    results = prediction["results"]
    if len(results) != len(labels["labels"]):
        raise ValueError("prediction and label counts differ")
    rows: list[dict[str, Any]] = []
    answers: list[str | None] = []
    rankings: list[Sequence[str]] = []
    abstentions: list[bool] = []
    correct: list[bool] = []
    decision_confidences: list[float] = []
    for query, result, label in zip(
        prediction["queries"], results, labels["labels"]
    ):
        if query["query_id"] != result["query_id"]:
            raise ValueError("prediction query/result order mismatch")
        predicted_id = result["ranked_ids"][0] if result["ranked_ids"] else None
        if label["answerable"]:
            is_correct = (
                not result["abstain"]
                and predicted_id == label["answer_object_id"]
            )
        else:
            is_correct = bool(result["abstain"]) and (
                label["expected_abstain_reason"] is None
                or result["reason"] == label["expected_abstain_reason"]
            )
        confidence = float(result["confidence"])
        decision_confidence = 1.0 - confidence if result["abstain"] else confidence
        answers.append(label["answer_object_id"])
        rankings.append(result["ranked_ids"])
        abstentions.append(bool(result["abstain"]))
        correct.append(bool(is_correct))
        decision_confidences.append(float(decision_confidence))
        rows.append({
            "query_id": query["query_id"],
            "relation": query["relation"],
            "answerable": label["answerable"],
            "answer_object_id": label["answer_object_id"],
            "predicted_object_id": predicted_id,
            "abstain": result["abstain"],
            "reason": result["reason"],
            "expected_abstain_reason": label["expected_abstain_reason"],
            "decision_confidence": decision_confidence,
            "correct": bool(is_correct),
        })
    base = grounding_metrics(answers, rankings, abstentions)
    curve = risk_coverage_curve(decision_confidences, correct)
    metrics = {
        **base,
        "task_accuracy": float(np.mean(correct)),
        "brier": brier_score(decision_confidences, correct),
        "ece_10": expected_calibration_error(
            decision_confidences, correct, bins=10
        ),
        "aurc_discrete": area_under_risk_coverage(curve),
        "query_count": len(rows),
        "positive_count": sum(label["answerable"] for label in labels["labels"]),
        "negative_count": sum(
            not label["answerable"] for label in labels["labels"]
        ),
    }
    return {
        "schema_version": RELATION_PROTOCOL_SCHEMA_VERSION,
        "status": "PASS",
        "stage": "D17-evaluation",
        "protocol_id": RELATION_PROTOCOL_ID,
        "scene_id": prediction["scene_id"],
        "split_role": prediction["split_role"],
        "source": deepcopy(dict(source)),
        "metrics": metrics,
        "risk_coverage": curve,
        "rows": rows,
        "acceptance": {
            "labels_read_by_evaluator_only": True,
            "negative_rejection_counted_as_task_success": True,
            "real_calibration_status": "REAL_DATA_CALIBRATION_PENDING",
            "performance_scope": "synthetic_protocol_fixture",
        },
        "created_at": str(created_at),
    }
