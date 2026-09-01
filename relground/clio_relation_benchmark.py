"""Real Clio directional grounding and selective-abstention benchmark."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.metrics import (
    area_under_risk_coverage,
    brier_score,
    expected_calibration_error,
    selective_answer_risk_coverage,
)
from .association import ObjectMemory
from .clio_association_benchmark import _task_paths
from .clio_retrieval_evaluation import slugify_task
from .clio_task_evaluation import _parse_gt_boxes, point_in_obb
from .relation_protocol import (
    RELATION_PROTOCOL_SCHEMA_VERSION,
    run_relation_prediction,
    validate_query_bundle,
)
from .schemas import MemoryObject


SCHEMA_VERSION = "0.2"
STAGE = "clio-relation-confirmatory-benchmark"
OPPOSITE = {
    "left_of": "right_of",
    "right_of": "left_of",
    "front_of": "behind",
    "behind": "front_of",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("relation benchmark source escapes project root") from error


def _rounded(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _rounded(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), 12)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _a2_memory_path(run_root: Path, scene_id: str, task: str) -> Path:
    slug = slugify_task(task)
    if scene_id == "apartment":
        if task == "bring me a pillow":
            return run_root / "a2-bring-me-a-pillow-k5/prediction/object_memory.json"
        return run_root / f"dev-a2-{slug}-k5/prediction/object_memory.json"
    return run_root / f"a2-{slug}-k5/prediction/object_memory.json"


def build_scene_object_memory(
    *,
    project_root: Path,
    query_manifest_path: Path,
    run_root: Path,
) -> tuple[ObjectMemory, list[dict[str, Any]]]:
    """Merge every permanent A2 object without consulting task GT."""

    root = project_root.resolve()
    manifest = json.loads(query_manifest_path.read_text(encoding="utf-8"))
    scene_id = str(manifest["scene_id"])
    split_role = str(manifest["role"])
    merged = ObjectMemory(metadata={
        "scene_id": scene_id,
        "split_role": split_role,
        "source_stage": "A2 permanent objects",
        "gt_read": False,
        "merge_rule": "all permanent objects with task-prefixed ids",
    })
    sources: list[dict[str, Any]] = []
    for query in manifest["queries"]:
        if query["split"] != split_role:
            continue
        task = str(query["task"])
        slug = slugify_task(task)
        path = _a2_memory_path(run_root, scene_id, task)
        row = {
            "task": task,
            "sam_query": str(query["sam_query"]),
            "a2_memory": None,
            "a2_memory_sha256": None,
            "permanent_objects": 0,
        }
        if not path.is_file():
            sources.append(row)
            continue
        source_memory = ObjectMemory.load(path)
        for source_object in sorted(source_memory.objects.values(), key=lambda item: item.object_id):
            raw = deepcopy(source_object.to_dict())
            prefix = f"{slug}__"
            raw["object_id"] = prefix + source_object.object_id
            raw["class_text"] = str(query["sam_query"])
            for observation, evidence in zip(raw["observations"], raw["evidence"]):
                observation["obs_id"] = prefix + str(observation["obs_id"])
                observation["mask_ref"] = None
                observation["points_ref"] = None
                metadata = dict(observation.get("metadata", {}))
                metadata.update({
                    "scene_memory_source_task": task,
                    "scene_memory_source_object_id": source_object.object_id,
                })
                observation["metadata"] = metadata
                evidence["obs_id"] = observation["obs_id"]
            item = MemoryObject.from_dict(raw)
            if item.object_id in merged.objects:
                raise ValueError(f"duplicate merged object id: {item.object_id}")
            merged.objects[item.object_id] = item
        row.update({
            "a2_memory": _relative(root, path),
            "a2_memory_sha256": _sha256(path),
            "permanent_objects": len(source_memory.objects),
        })
        sources.append(row)
    # Round-trip through the frozen schema to assert global observation ids.
    payload = merged.to_dict()
    if len(payload["evidence"]["associated_observation_ids"]) != len(set(payload["evidence"]["associated_observation_ids"])):
        raise ValueError("merged scene memory has duplicate observation ids")
    return merged, sources


def _transform_center(center: Sequence[float], alignment: Mapping[str, Any]) -> np.ndarray:
    scale = float(alignment["sim3"]["scale"])
    rotation = np.asarray(alignment["sim3"]["rotation"], dtype=np.float64)
    translation = np.asarray(alignment["sim3"]["translation"], dtype=np.float64)
    return scale * (rotation @ np.asarray(center, dtype=np.float64)) + translation


def _object_matches_by_task(
    memory: ObjectMemory,
    task_rows: Sequence[Mapping[str, Any]],
    gt_by_task: Mapping[str, Sequence[Mapping[str, Any]]],
    alignment: Mapping[str, Any],
) -> dict[str, dict[str, list[str]]]:
    padding = float(alignment["error_m"]["rmse"])
    matches: dict[str, dict[str, list[str]]] = {}
    for row in task_rows:
        task = str(row["task"])
        slug = slugify_task(task)
        boxes = gt_by_task[task]
        strict: list[str] = []
        padded: list[str] = []
        for item in memory.objects.values():
            if not item.object_id.startswith(f"{slug}__"):
                continue
            center = _transform_center(item.fused_center, alignment)
            if any(point_in_obb(
                center, center=box["center"], extent=box["extent"], rotation=box["rotation"]
            ) for box in boxes):
                strict.append(item.object_id)
            if any(point_in_obb(
                center,
                center=box["center"],
                extent=box["extent"],
                rotation=box["rotation"],
                padding_m=padding,
            ) for box in boxes):
                padded.append(item.object_id)
        matches[task] = {
            "strict": sorted(strict),
            "alignment_rmse_padded": sorted(padded),
        }
    return matches


def build_relation_queries_and_labels(
    *,
    memory: ObjectMemory,
    query_manifest: Mapping[str, Any],
    task_yaml_path: Path,
    world_alignment: Mapping[str, Any],
    anchor_poses: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    split_role = str(query_manifest["role"])
    tasks = [row for row in query_manifest["queries"] if row["split"] == split_role]
    gt_by_task = {
        str(row["task"]): _parse_gt_boxes(task_yaml_path, str(row["task"]))
        for row in tasks
    }
    eligible = [row for row in tasks if len(gt_by_task[str(row["task"])]) == 1]
    if not anchor_poses:
        raise ValueError("relation benchmark requires anchor poses")
    anchor_frame = next(iter(anchor_poses))
    anchor = np.asarray(anchor_poses[anchor_frame], dtype=np.float64)
    if anchor.shape != (4, 4):
        raise ValueError("relation anchor pose must be 4x4")
    world_from_vggt = np.asarray(world_alignment["sim3"]["rotation"], dtype=np.float64)
    world_from_anchor_rotation = world_from_vggt @ anchor[:3, :3]
    minimum_axis = float(protocol["query_generation"]["minimum_absolute_axis_m"])
    matches = _object_matches_by_task(memory, tasks, gt_by_task, world_alignment)
    queries: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    skipped_boundary = 0
    for target_index, target in enumerate(eligible):
        for reference in eligible[target_index + 1:]:
            target_task = str(target["task"])
            reference_task = str(reference["task"])
            delta_world = (
                np.asarray(gt_by_task[target_task][0]["center"], dtype=np.float64)
                - np.asarray(gt_by_task[reference_task][0]["center"], dtype=np.float64)
            )
            delta_anchor = world_from_anchor_rotation.T @ delta_world
            if abs(float(delta_anchor[0])) >= abs(float(delta_anchor[2])):
                magnitude = abs(float(delta_anchor[0]))
                positive_relation = "right_of" if delta_anchor[0] > 0 else "left_of"
            else:
                magnitude = abs(float(delta_anchor[2]))
                positive_relation = "front_of" if delta_anchor[2] > 0 else "behind"
            if magnitude < minimum_axis:
                skipped_boundary += 1
                continue
            base = f"{slugify_task(target_task)}--{slugify_task(reference_task)}"
            positive_id = f"positive--{base}"
            negative_id = f"negative--{base}"
            common = {
                "target": str(target["sam_query"]),
                "reference": str(reference["sam_query"]),
                "anchor_frame": anchor_frame,
            }
            queries.extend([
                {"query_id": positive_id, "relation": positive_relation, **common},
                {"query_id": negative_id, "relation": OPPOSITE[positive_relation], **common},
            ])
            target_matches = matches[target_task]
            reference_matches = matches[reference_task]
            labels.extend([
                {
                    "query_id": positive_id,
                    "answerable": True,
                    "target_task": target_task,
                    "reference_task": reference_task,
                    "relation": positive_relation,
                    "signed_axis_distance_m": magnitude,
                    "acceptable_target_object_ids_strict": target_matches["strict"],
                    "acceptable_target_object_ids_alignment_rmse_padded": target_matches["alignment_rmse_padded"],
                    "acceptable_reference_object_ids_strict": reference_matches["strict"],
                    "acceptable_reference_object_ids_alignment_rmse_padded": reference_matches["alignment_rmse_padded"],
                    "expected_abstain_reason": None,
                },
                {
                    "query_id": negative_id,
                    "answerable": False,
                    "target_task": target_task,
                    "reference_task": reference_task,
                    "relation": OPPOSITE[positive_relation],
                    "signed_axis_distance_m": -magnitude,
                    "acceptable_target_object_ids_strict": target_matches["strict"],
                    "acceptable_target_object_ids_alignment_rmse_padded": target_matches["alignment_rmse_padded"],
                    "acceptable_reference_object_ids_strict": reference_matches["strict"],
                    "acceptable_reference_object_ids_alignment_rmse_padded": reference_matches["alignment_rmse_padded"],
                    "expected_abstain_reason": "relation_conflict_or_boundary",
                },
            ])
    query_bundle = {
        "schema_version": RELATION_PROTOCOL_SCHEMA_VERSION,
        "scene_id": str(query_manifest["scene_id"]),
        "split_role": split_role,
        "queries": queries,
    }
    validate_query_bundle(query_bundle)
    label_bundle = {
        "schema_version": SCHEMA_VERSION,
        "scene_id": str(query_manifest["scene_id"]),
        "split_role": split_role,
        "labels": labels,
    }
    generation = {
        "frozen_task_count": len(tasks),
        "eligible_single_gt_task_count": len(eligible),
        "ineligible_multi_gt_task_count": len(tasks) - len(eligible),
        "skipped_near_boundary_pairs": skipped_boundary,
        "positive_queries": sum(row["answerable"] for row in labels),
        "negative_queries": sum(not row["answerable"] for row in labels),
        "anchor_frame": anchor_frame,
        "minimum_absolute_axis_m": minimum_axis,
    }
    return _rounded(query_bundle), _rounded(label_bundle), _rounded(generation)


def _pair_gt_matches(
    label: Mapping[str, Any],
    *,
    predicted_target_id: str | None,
    predicted_reference_id: str | None,
) -> dict[str, bool]:
    """Check both semantic roles against their evaluator-only GT instances."""

    target_strict = predicted_target_id in label["acceptable_target_object_ids_strict"]
    target_padded = predicted_target_id in label[
        "acceptable_target_object_ids_alignment_rmse_padded"
    ]
    reference_strict = predicted_reference_id in label[
        "acceptable_reference_object_ids_strict"
    ]
    reference_padded = predicted_reference_id in label[
        "acceptable_reference_object_ids_alignment_rmse_padded"
    ]
    return {
        "target_strict": target_strict,
        "target_padded": target_padded,
        "reference_strict": reference_strict,
        "reference_padded": reference_padded,
        "pair_strict": target_strict and reference_strict,
        "pair_padded": target_padded and reference_padded,
    }


def evaluate_clio_relation_prediction(
    prediction: Mapping[str, Any],
    labels: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    generation: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    label_by_id = {str(row["query_id"]): row for row in labels["labels"]}
    rows: list[dict[str, Any]] = []
    confidences: list[float] = []
    answered_flags: list[bool] = []
    answer_correct: list[bool] = []
    answerability: list[bool] = []
    positive_strict = positive_padded = negative_rejected = 0
    reason_matched_negative = pair_grounded_relation_negative = 0
    positive_count = negative_count = 0
    reasons: dict[str, int] = {}
    for query, result in zip(prediction["queries"], prediction["results"]):
        label = label_by_id[str(query["query_id"])]
        predicted_id = result["ranked_ids"][0] if result["ranked_ids"] else None
        explanation = result.get("explanation") or {}
        predicted_reference_id = explanation.get("reference_id")
        pair_match = _pair_gt_matches(
            label,
            predicted_target_id=predicted_id,
            predicted_reference_id=predicted_reference_id,
        )
        answered = not bool(result["abstain"])
        if label["answerable"]:
            positive_count += 1
            strict_correct = answered and pair_match["pair_strict"]
            padded_correct = answered and pair_match["pair_padded"]
            positive_strict += int(strict_correct)
            positive_padded += int(padded_correct)
            current_correct = bool(padded_correct)
        else:
            negative_count += 1
            strict_correct = padded_correct = False
            negative_rejected += int(result["abstain"])
            reason_matched_negative += int(
                result["abstain"] and result["reason"] == label["expected_abstain_reason"]
            )
            pair_grounded_relation_negative += int(
                result["abstain"]
                and result["reason"] == label["expected_abstain_reason"]
                and pair_match["pair_padded"]
            )
            current_correct = False
        reason = str(result["reason"]) if result["reason"] is not None else "answered"
        reasons[reason] = reasons.get(reason, 0) + 1
        confidence = float(result["confidence"])
        confidences.append(confidence)
        answered_flags.append(answered)
        answer_correct.append(current_correct)
        answerability.append(bool(label["answerable"]))
        rows.append({
            "query_id": query["query_id"],
            "answerable": label["answerable"],
            "target_task": label["target_task"],
            "reference_task": label["reference_task"],
            "relation": query["relation"],
            "predicted_object_id": predicted_id,
            "predicted_reference_object_id": predicted_reference_id,
            "target_strict_gt_match": pair_match["target_strict"],
            "target_alignment_rmse_padded_gt_match": pair_match["target_padded"],
            "reference_strict_gt_match": pair_match["reference_strict"],
            "reference_alignment_rmse_padded_gt_match": pair_match["reference_padded"],
            "pair_strict_gt_match": pair_match["pair_strict"],
            "pair_alignment_rmse_padded_gt_match": pair_match["pair_padded"],
            "abstain": result["abstain"],
            "reason": result["reason"],
            "confidence": confidence,
            "strict_correct": strict_correct,
            "alignment_rmse_padded_correct": padded_correct,
            "negative_rejection_correct": bool(not label["answerable"] and result["abstain"]),
            "relation_aware_negative_correct": bool(
                not label["answerable"]
                and result["abstain"]
                and result["reason"] == label["expected_abstain_reason"]
                and pair_match["pair_padded"]
            ),
            "reason_matched_negative_rejection": bool(
                not label["answerable"]
                and result["abstain"]
                and result["reason"] == label["expected_abstain_reason"]
            ),
        })
    curve = selective_answer_risk_coverage(confidences, answered_flags, answer_correct)
    total = len(rows)
    metrics = {
        "query_count": total,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_pair_grounding_acc_at_1_strict": positive_strict / positive_count if positive_count else 0.0,
        "positive_pair_grounding_acc_at_1_alignment_rmse_padded": positive_padded / positive_count if positive_count else 0.0,
        "negative_rejection_accuracy": negative_rejected / negative_count if negative_count else 0.0,
        "reason_matched_negative_rejection_accuracy": reason_matched_negative / negative_count if negative_count else 0.0,
        "relation_aware_negative_rejection_accuracy": pair_grounded_relation_negative / negative_count if negative_count else 0.0,
        "end_to_end_task_accuracy_alignment_rmse_padded": (positive_padded + negative_rejected) / total if total else 0.0,
        "pair_grounded_task_accuracy_alignment_rmse_padded": (
            (positive_padded + pair_grounded_relation_negative) / total if total else 0.0
        ),
        "answer_coverage": sum(answered_flags) / total if total else 0.0,
        "answer_aurc_discrete": area_under_risk_coverage(curve) if curve else None,
        "answerability_proxy_brier": brier_score(confidences, answerability),
        "answerability_proxy_ece_10": expected_calibration_error(confidences, answerability, bins=10),
    }
    return _rounded({
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "stage": STAGE,
        "scene_id": prediction["scene_id"],
        "split_role": prediction["split_role"],
        "contract": {
            "prediction_is_label_free": True,
            "gt_and_alignment_are_evaluator_only": True,
            "positive_correctness_requires_target_and_reference_gt": True,
            "relation_aware_negative_requires_grounded_pair": True,
            "correct_rejections_excluded_from_answer_coverage": True,
            "answer_coverage_denominator": "all frozen positive and negative queries",
            "background_or_missing_detection_not_removed": True,
            "calibration_status": "ENGINEERING_DEFAULT_UNCALIBRATED",
            "official_clio_metric_claim": False,
        },
        "source": deepcopy(dict(source)),
        "generation": deepcopy(dict(generation)),
        "metrics": metrics,
        "abstain_reasons": reasons,
        "selective_answer_risk_coverage": curve,
        "rows": rows,
        "created_at": created_at,
    })


def validate_relation_bundle(bundle: Path, *, project_root: Path) -> dict[str, Any]:
    failures: list[str] = []
    required = (
        "scene_object_memory.json", "anchor_poses.json", "queries.json",
        "labels.json", "calibration_manifest.json", "prediction.json",
        "evaluation.json", "bundle_sources.json",
    )
    try:
        for name in required:
            if not (bundle / name).is_file():
                raise ValueError(f"missing relation bundle artifact: {name}")
        root = project_root.resolve()
        sources = json.loads((bundle / "bundle_sources.json").read_text(encoding="utf-8"))

        def resolve(reference: str) -> Path:
            path = Path(reference)
            if path.is_absolute():
                raise ValueError("relation source references must be relative")
            resolved = (root / path).resolve()
            try:
                resolved.relative_to(root)
            except ValueError as error:
                raise ValueError("relation source reference escapes project root") from error
            return resolved

        query_manifest_path = resolve(sources["query_manifest"])
        run_root = resolve(sources["run_root"])
        rebuilt_memory, rebuilt_sources = build_scene_object_memory(
            project_root=root,
            query_manifest_path=query_manifest_path,
            run_root=run_root,
        )
        memory = ObjectMemory.load(bundle / "scene_object_memory.json")
        if rebuilt_memory.to_dict() != memory.to_dict():
            raise ValueError("scene object memory differs from label-free replay")
        anchors = json.loads((bundle / "anchor_poses.json").read_text(encoding="utf-8"))
        original_anchors = json.loads(resolve(sources["geometry_anchor_poses"]).read_text(encoding="utf-8"))
        expected_anchor_key = next(iter(original_anchors))
        if anchors != {expected_anchor_key: original_anchors[expected_anchor_key]}:
            raise ValueError("relation anchor bundle differs from frozen first-frame rule")
        query_manifest = json.loads(query_manifest_path.read_text(encoding="utf-8"))
        alignment = json.loads(resolve(sources["world_alignment"]).read_text(encoding="utf-8"))
        protocol = json.loads(resolve(sources["protocol"]).read_text(encoding="utf-8"))
        queries, labels, generation = build_relation_queries_and_labels(
            memory=memory,
            query_manifest=query_manifest,
            task_yaml_path=resolve(sources["task_gt"]),
            world_alignment=alignment,
            anchor_poses=anchors,
            protocol=protocol,
        )
        actual_queries = json.loads((bundle / "queries.json").read_text(encoding="utf-8"))
        actual_labels = json.loads((bundle / "labels.json").read_text(encoding="utf-8"))
        if queries != actual_queries or labels != actual_labels:
            raise ValueError("relation queries or evaluator labels differ from frozen replay")
        calibration = json.loads((bundle / "calibration_manifest.json").read_text(encoding="utf-8"))
        prediction = json.loads((bundle / "prediction.json").read_text(encoding="utf-8"))
        prediction_source = {
            "object_memory": "scene_object_memory.json",
            "object_memory_sha256": _sha256(bundle / "scene_object_memory.json"),
            "anchor_poses": "anchor_poses.json",
            "anchor_poses_sha256": _sha256(bundle / "anchor_poses.json"),
            "queries": "queries.json",
            "queries_sha256": _sha256(bundle / "queries.json"),
            "calibration": "calibration_manifest.json",
            "calibration_sha256": _sha256(bundle / "calibration_manifest.json"),
        }
        replay_prediction = run_relation_prediction(
            memory, anchors, queries, calibration,
            source=prediction_source,
            created_at=str(prediction["created_at"]),
        )
        if replay_prediction != prediction:
            raise ValueError("relation prediction differs from deterministic replay")
        evaluation = json.loads((bundle / "evaluation.json").read_text(encoding="utf-8"))
        evaluation_source = {
            **sources,
            "scene_memory_sources": rebuilt_sources,
            "scene_object_memory_sha256": _sha256(bundle / "scene_object_memory.json"),
            "queries_sha256": _sha256(bundle / "queries.json"),
            "labels_sha256": _sha256(bundle / "labels.json"),
            "prediction_sha256": _sha256(bundle / "prediction.json"),
        }
        replay_evaluation = evaluate_clio_relation_prediction(
            prediction, labels,
            source=evaluation_source,
            generation=generation,
            created_at=str(evaluation["created_at"]),
        )
        if replay_evaluation != evaluation:
            raise ValueError("relation evaluation differs from deterministic replay")
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        failures.append(str(error))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not failures else "FAIL",
        "stage": f"{STAGE}-validation",
        "checks": {
            "label_free_scene_memory_replayed": not failures,
            "frozen_queries_replayed": not failures,
            "prediction_replayed": not failures,
            "evaluation_replayed": not failures,
            "selective_answer_coverage_valid": not failures,
        },
        "failures": failures,
    }
