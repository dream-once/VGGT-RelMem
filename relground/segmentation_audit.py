"""Label-isolated frame inventory for segmentation-recall audits."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SEGMENTATION_AUDIT_SCHEMA_VERSION = "0.1"
SEGMENTATION_AUDIT_STAGE = "D21.1-segmentation-recall-audit"
VISIBILITY_LABEL_SCHEMA_VERSION = "0.1"
VISIBILITY_VALUES = ("VISIBLE", "NOT_VISIBLE", "PENDING")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _project_reference(project_root: Path, path: Path) -> str:
    root = project_root.resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"artifact is outside project root: {path}") from error


def _resolve_reference(project_root: Path, reference: str) -> Path:
    path = Path(reference)
    if path.is_absolute():
        raise ValueError("audit references must be project-relative")
    root = project_root.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"audit reference escapes project root: {reference}") from error
    return resolved


def _observation_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = payload.get("observations")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("observations.json must contain an observations list")
    return rows


def build_segmentation_inventory(
    *,
    project_root: Path,
    d6_result_path: Path,
    observations_path: Path,
    scene_id: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a label-free inventory from an already materialized D6 run."""

    project_root = project_root.resolve()
    d6_result_path = d6_result_path.resolve()
    observations_path = observations_path.resolve()
    result = read_json_object(d6_result_path)
    observations_payload = read_json_object(observations_path)
    processed = result.get("processed_frames")
    if not isinstance(processed, list) or not processed:
        raise ValueError("D6 result must contain processed_frames")
    if any(not isinstance(row, Mapping) for row in processed):
        raise ValueError("processed frame rows must be objects")
    observations = _observation_rows(observations_payload)
    observation_ids: set[str] = set()
    observations_by_frame: dict[str, list[str]] = {}
    for observation in observations:
        obs_id = str(observation.get("obs_id", ""))
        frame_id = str(observation.get("frame_id", ""))
        if not obs_id or not frame_id or obs_id in observation_ids:
            raise ValueError("observation ids and frame ids must be non-empty and unique")
        observation_ids.add(obs_id)
        observations_by_frame.setdefault(frame_id, []).append(obs_id)

    d6_root = d6_result_path.parent
    frame_ids: set[str] = set()
    ranks: set[int] = set()
    frames: list[dict[str, Any]] = []
    for raw in processed:
        frame_id = str(raw.get("frame_id", ""))
        rank = int(raw.get("rank", 0))
        if not frame_id or frame_id in frame_ids or rank < 1 or rank in ranks:
            raise ValueError("processed frame ids/ranks must be positive and unique")
        frame_ids.add(frame_id)
        ranks.add(rank)
        sam_instances = int(raw.get("sam_instances", -1))
        lifted_instances = int(raw.get("lifted_instances", -1))
        rejected_instances = int(raw.get("rejected_instances", -1))
        if min(sam_instances, lifted_instances, rejected_instances) < 0:
            raise ValueError("per-frame counts must be non-negative")
        if sam_instances != lifted_instances + rejected_instances:
            raise ValueError("SAM count must equal lifted plus rejected count")
        frame_observations = sorted(observations_by_frame.get(frame_id, []))
        if len(frame_observations) != lifted_instances:
            raise ValueError("lifted count differs from observations.json")
        sam_input = d6_root / str(raw.get("sam_input", ""))
        preview = d6_root / str(raw.get("preview", ""))
        if not sam_input.is_file() or not preview.is_file():
            raise ValueError(f"missing SAM input or preview for {frame_id}")
        frames.append({
            "rank": rank,
            "frame_id": frame_id,
            "retrieval_score": float(raw.get("retrieval_score", 0.0)),
            "sam_instances": sam_instances,
            "lifted_instances": lifted_instances,
            "rejected_instances": rejected_instances,
            "observation_ids": frame_observations,
            "sam_input_ref": _project_reference(project_root, sam_input),
            "sam_input_sha256": sha256_file(sam_input),
            "preview_ref": _project_reference(project_root, preview),
            "preview_sha256": sha256_file(preview),
        })
    unknown_frames = set(observations_by_frame) - frame_ids
    if unknown_frames:
        raise ValueError(f"observations reference unknown frames: {sorted(unknown_frames)}")
    frames.sort(key=lambda row: (row["rank"], row["frame_id"]))

    counts = {
        "processed_frames": len(frames),
        "frames_with_masks": sum(row["sam_instances"] > 0 for row in frames),
        "frames_without_masks": sum(row["sam_instances"] == 0 for row in frames),
        "sam_instances": sum(row["sam_instances"] for row in frames),
        "lifted_instances": sum(row["lifted_instances"] for row in frames),
        "rejected_instances": sum(row["rejected_instances"] for row in frames),
    }
    expected_counts = {
        "sam_instances": int(result.get("sam_instances", -1)),
        "lifted_instances": int(result.get("lifted_instances", -1)),
        "rejected_instances": len(result.get("rejected_instances", [])),
    }
    for key, expected in expected_counts.items():
        if counts[key] != expected:
            raise ValueError(f"D6 total differs for {key}: {counts[key]} != {expected}")
    query = str(result.get("query", "")).strip()
    if not query:
        raise ValueError("D6 query is required")
    return {
        "schema_version": SEGMENTATION_AUDIT_SCHEMA_VERSION,
        "status": "PASS_WITH_FRAME_VISIBILITY_PENDING",
        "stage": SEGMENTATION_AUDIT_STAGE,
        "scene_id": scene_id,
        "query": query,
        "source": {
            "d6_result": _project_reference(project_root, d6_result_path),
            "d6_result_sha256": sha256_file(d6_result_path),
            "observations": _project_reference(project_root, observations_path),
            "observations_sha256": sha256_file(observations_path),
        },
        "sam_threshold": float(result.get("sam_threshold", 0.0)),
        "counts": counts,
        "frames": frames,
        "claim_boundary": {
            "frame_visibility_labels": "PENDING_MANUAL_ANNOTATION",
            "segmentation_recall": None,
            "mask_presence_is_not_frame_label": True,
        },
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }


def build_visibility_template(inventory: Mapping[str, Any]) -> dict[str, Any]:
    """Create a separate, initially unlabelled frame-visibility sheet."""

    frames = inventory.get("frames")
    if not isinstance(frames, list):
        raise ValueError("inventory frames are required")
    return {
        "schema_version": VISIBILITY_LABEL_SCHEMA_VERSION,
        "scene_id": str(inventory["scene_id"]),
        "query": str(inventory["query"]),
        "split_role": "development_annotation",
        "annotation_rule": (
            "Review the exact SAM input, not the colored overlay; mark whether "
            "at least one physical query instance is visibly present."
        ),
        "frames": [
            {
                "frame_id": str(row["frame_id"]),
                "visibility": "PENDING",
                "visible_instance_ids": [],
                "notes": "",
            }
            for row in frames
        ],
    }


def evaluate_visibility(
    inventory: Mapping[str, Any],
    labels: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate mask-presence recall only when frame labels are complete."""

    if labels.get("schema_version") != VISIBILITY_LABEL_SCHEMA_VERSION:
        raise ValueError("unsupported visibility-label schema")
    if labels.get("scene_id") != inventory.get("scene_id"):
        raise ValueError("visibility labels scene differs from inventory")
    if labels.get("query") != inventory.get("query"):
        raise ValueError("visibility labels query differs from inventory")
    frame_rows = inventory.get("frames")
    label_rows = labels.get("frames")
    if not isinstance(frame_rows, list) or not isinstance(label_rows, list):
        raise ValueError("inventory and labels must contain frame lists")
    predictions = {str(row["frame_id"]): int(row["sam_instances"]) > 0 for row in frame_rows}
    normalized: dict[str, str] = {}
    for row in label_rows:
        if not isinstance(row, Mapping):
            raise ValueError("visibility rows must be objects")
        frame_id = str(row.get("frame_id", ""))
        value = str(row.get("visibility", ""))
        if frame_id in normalized or value not in VISIBILITY_VALUES:
            raise ValueError("visibility frame ids must be unique and values frozen")
        normalized[frame_id] = value
    if set(normalized) != set(predictions):
        raise ValueError("visibility labels must cover the exact frame universe")
    pending = sorted(frame_id for frame_id, value in normalized.items() if value == "PENDING")
    if pending:
        return {
            "status": "PENDING_FRAME_VISIBILITY_LABELS",
            "labelled_frames": len(normalized) - len(pending),
            "pending_frames": pending,
            "metrics": None,
        }
    true_positive = false_positive = true_negative = false_negative = 0
    for frame_id, prediction in predictions.items():
        visible = normalized[frame_id] == "VISIBLE"
        true_positive += int(visible and prediction)
        false_positive += int(not visible and prediction)
        true_negative += int(not visible and not prediction)
        false_negative += int(visible and not prediction)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {
        "status": "PASS",
        "labelled_frames": len(normalized),
        "pending_frames": [],
        "metrics": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
        },
    }


def validate_segmentation_inventory(
    inventory: Mapping[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    failures: list[str] = []
    try:
        source = inventory["source"]
        d6_path = _resolve_reference(project_root, str(source["d6_result"]))
        observations_path = _resolve_reference(project_root, str(source["observations"]))
        if sha256_file(d6_path) != source["d6_result_sha256"]:
            raise ValueError("D6 source hash mismatch")
        if sha256_file(observations_path) != source["observations_sha256"]:
            raise ValueError("observations source hash mismatch")
        recomputed = build_segmentation_inventory(
            project_root=project_root,
            d6_result_path=d6_path,
            observations_path=observations_path,
            scene_id=str(inventory["scene_id"]),
            created_at=str(inventory["created_at"]),
        )
        if recomputed != dict(inventory):
            raise ValueError("segmentation inventory differs from deterministic replay")
        forbidden_keys = {
            "ground_truth",
            "instance_id",
            "instance_label",
            "labels",
            "visibility",
            "visible_instance_ids",
            "answer_key",
        }

        def walk_keys(value: Any) -> None:
            if isinstance(value, Mapping):
                leaked = forbidden_keys & set(value)
                if leaked:
                    raise ValueError(
                        "label data leaked into segmentation inventory: "
                        + ", ".join(sorted(leaked))
                    )
                for nested in value.values():
                    walk_keys(nested)
            elif isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                for nested in value:
                    walk_keys(nested)

        walk_keys(inventory)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        failures.append(str(error))
    return {
        "schema_version": "0.1",
        "status": "PASS" if not failures else "FAIL",
        "stage": "D21.1-segmentation-audit-validation",
        "checks": {
            "source_hashes": not failures,
            "deterministic_replay": not failures,
            "prediction_label_free": not failures,
        },
        "failures": failures,
    }
