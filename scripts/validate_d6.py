"""Validate D6 multi-frame SAM masks and robust 3D observations without a GPU."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from adapters.masks import load_mask, load_mask_manifest
from adapters.open_vocab import SAM3_SOURCE_COMMIT
from relground.schemas import ObjectObservation


def _safe_artifact(root: Path, reference: str) -> Path:
    relative = Path(reference)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"artifact reference escapes output directory: {reference}")
    candidate = (root / relative).resolve()
    if candidate != root.resolve() and root.resolve() not in candidate.parents:
        raise ValueError(f"artifact reference escapes output directory: {reference}")
    return candidate


def validate_output(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    errors: list[str] = []
    result_path = root / "d6_result.json"
    selection_path = root / "selection.json"
    observations_path = root / "observations.json"
    masks_path = root / "masks.json"
    manifest_path = root / "run_manifest.json"
    required = [
        result_path,
        selection_path,
        observations_path,
        masks_path,
        manifest_path,
    ]
    for artifact in required:
        if not artifact.is_file() or artifact.stat().st_size == 0:
            errors.append(f"missing or empty artifact: {artifact.name}")
    if errors:
        return {"status": "FAIL", "errors": errors}

    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        observation_payload = json.loads(
            observations_path.read_text(encoding="utf-8")
        )
        records = load_mask_manifest(masks_path)
        observations = [
            ObjectObservation.from_dict(row)
            for row in observation_payload.get("observations", [])
        ]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {"status": "FAIL", "errors": [f"invalid D6 artifact: {error}"]}

    if result.get("stage") != "D6":
        errors.append("result stage is not D6")
    if result.get("status") != "PASS":
        errors.append(f"D6 result status is {result.get('status')!r}")
    if result.get("source_commits", {}).get("sam3") != SAM3_SOURCE_COMMIT:
        errors.append("SAM 3 source commit is not the pinned D6 commit")
    query = str(result.get("query", "")).strip()
    if not query:
        errors.append("result query is empty")
    if selection.get("stage") != "D5":
        errors.append("selection snapshot stage is not D5")
    if str(selection.get("query", "")).strip() != query:
        errors.append("selection and D6 query disagree")

    selected_rows = result.get("selected_frames", [])
    selection_rows = selection.get("frames", [])
    if not isinstance(selected_rows, list) or not isinstance(selection_rows, list):
        errors.append("selected frame records must be lists")
        selected_rows = []
        selection_rows = []
    if any(not isinstance(row, dict) for row in selected_rows):
        errors.append("D6 selected frame entries must be objects")
        selected_rows = [row for row in selected_rows if isinstance(row, dict)]
    if any(not isinstance(row, dict) for row in selection_rows):
        errors.append("D5 selection frame entries must be objects")
        selection_rows = [row for row in selection_rows if isinstance(row, dict)]
    selected_ids = [str(row.get("frame_id", "")) for row in selected_rows]
    snapshot_ids = [str(row.get("frame_id", "")) for row in selection_rows]
    if selected_ids != snapshot_ids:
        errors.append("D6 selected frames differ from the D5 snapshot")
    selected_by_id = {str(row.get("frame_id", "")): row for row in selected_rows}
    if len(selected_ids) < 2 or len(set(selected_ids)) != len(selected_ids):
        errors.append("D6 must contain at least two unique selected frames")
    expected_ranks = list(range(1, len(selected_rows) + 1))
    if [int(row.get("rank", -1)) for row in selected_rows] != expected_ranks:
        errors.append("selected frame ranks are invalid")
    if int(selection.get("selected_count", -1)) != len(selection_rows):
        errors.append("selection selected_count is inconsistent")
    for selected, snapshot in zip(selected_rows, selection_rows):
        try:
            if int(selected.get("geometry_index", -1)) != int(
                snapshot.get("geometry_index", -2)
            ):
                errors.append(
                    f"geometry index changed for {selected.get('frame_id')}"
                )
            for name in ("retrieval_score", "retrieval_cosine"):
                if not np.isclose(
                    float(selected[name]),
                    float(snapshot[name]),
                    atol=1e-12,
                ):
                    errors.append(
                        f"{name} changed for {selected.get('frame_id')}"
                    )
        except (KeyError, TypeError, ValueError):
            errors.append(
                f"invalid selected-frame values for {selected.get('frame_id')}"
            )
    if int(result.get("requested_k", -1)) != int(
        selection.get("requested_k", -2)
    ):
        errors.append("D6 requested_k differs from the D5 snapshot")

    processed = result.get("processed_frames", [])
    if not isinstance(processed, list):
        errors.append("processed_frames must be a list")
        processed = []
    processed_ids = [
        str(row.get("frame_id", "")) if isinstance(row, dict) else ""
        for row in processed
    ]
    if processed_ids != selected_ids:
        errors.append("processed frames do not exactly match selected frames")
    processed_by_id = {
        str(row.get("frame_id", "")): row
        for row in processed
        if isinstance(row, dict)
    }
    for row in processed:
        if not isinstance(row, dict):
            errors.append("processed frame entries must be objects")
            continue
        frame_id = str(row.get("frame_id", ""))
        expected_rank = (
            selected_ids.index(frame_id) + 1 if frame_id in selected_ids else -1
        )
        if int(row.get("rank", -1)) != expected_rank:
            errors.append(f"processed rank mismatch for {frame_id}")
        for name in ("sam_instances", "lifted_instances", "rejected_instances"):
            if int(row.get(name, -1)) < 0:
                errors.append(f"invalid {name} for {frame_id}")
        try:
            preview = _safe_artifact(root, str(row["preview"]))
            if not preview.is_file() or preview.stat().st_size == 0:
                errors.append(f"missing preview for {frame_id}")
        except (KeyError, OSError, ValueError) as error:
            errors.append(f"invalid preview for {frame_id}: {error}")

    record_ids = [record.obs_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        errors.append("mask observation ids are not unique")
    mask_counts: Counter[str] = Counter()
    for record in records:
        mask_counts[record.frame_id] += 1
        if record.frame_id not in selected_ids:
            errors.append(f"mask uses unselected frame: {record.obs_id}")
        if record.class_text != query:
            errors.append(f"mask query mismatch: {record.obs_id}")
        selected_row = selected_by_id.get(record.frame_id)
        try:
            if selected_row is not None and not np.isclose(
                record.retrieval_score,
                float(selected_row["retrieval_score"]),
                atol=1e-12,
            ):
                errors.append(f"mask retrieval score mismatch: {record.obs_id}")
        except (KeyError, TypeError, ValueError):
            errors.append(f"invalid retrieval score for mask: {record.obs_id}")
        try:
            _safe_artifact(root, record.mask_ref)
            mask = load_mask(record, masks_path)
            if mask.ndim != 2 or not mask.any():
                errors.append(f"invalid or empty mask: {record.obs_id}")
        except (OSError, ValueError) as error:
            errors.append(f"cannot load mask {record.obs_id}: {error}")

    observation_ids = [observation.obs_id for observation in observations]
    if len(observation_ids) != len(set(observation_ids)):
        errors.append("3D observation ids are not unique")
    observation_counts: Counter[str] = Counter()
    records_by_id = {record.obs_id: record for record in records}
    min_points = int(result.get("lifter_config", {}).get("min_points", 3))
    if min_points < 3:
        errors.append("lifter min_points must be at least 3")
        min_points = 3
    for observation in observations:
        observation_counts[observation.frame_id] += 1
        record = records_by_id.get(observation.obs_id)
        if record is None:
            errors.append(f"3D observation has no mask record: {observation.obs_id}")
        else:
            if observation.frame_id != record.frame_id:
                errors.append(f"mask/observation frame mismatch: {observation.obs_id}")
            if observation.mask_ref != record.mask_ref:
                errors.append(f"mask/observation reference mismatch: {observation.obs_id}")
            if not np.isclose(
                observation.retrieval_score,
                record.retrieval_score,
                atol=1e-12,
            ):
                errors.append(
                    f"observation retrieval score mismatch: {observation.obs_id}"
                )
            if not np.isclose(
                observation.sam_score, record.sam_score, atol=1e-12
            ):
                errors.append(f"observation SAM score mismatch: {observation.obs_id}")
        if observation.frame_id not in selected_ids:
            errors.append(f"observation uses unselected frame: {observation.obs_id}")
        if observation.class_text != query:
            errors.append(f"observation query mismatch: {observation.obs_id}")
        expected_rank = (
            selected_ids.index(observation.frame_id) + 1
            if observation.frame_id in selected_ids
            else -1
        )
        if int(observation.metadata.get("selected_rank", -1)) != expected_rank:
            errors.append(f"observation rank mismatch: {observation.obs_id}")
        if not observation.points_ref:
            errors.append(f"observation has no points_ref: {observation.obs_id}")
            continue
        try:
            point_path = _safe_artifact(root, observation.points_ref)
            with np.load(point_path, allow_pickle=False) as archive:
                points = np.asarray(archive["points"])
            if (
                points.ndim != 2
                or points.shape[1] != 3
                or len(points) < min_points
                or not np.all(np.isfinite(points))
            ):
                errors.append(f"invalid lifted points: {observation.obs_id}")
        except (OSError, KeyError, ValueError) as error:
            errors.append(f"cannot load points {observation.obs_id}: {error}")

    rejected = result.get("rejected_instances", [])
    if not isinstance(rejected, list):
        errors.append("rejected_instances must be a list")
        rejected = []
    if any(not isinstance(row, dict) for row in rejected):
        errors.append("rejected instance entries must be objects")
        rejected = [row for row in rejected if isinstance(row, dict)]
    rejected_ids = [str(row.get("obs_id", "")) for row in rejected]
    if len(rejected_ids) != len(set(rejected_ids)):
        errors.append("rejected observation ids are not unique")
    rejected_counts: Counter[str] = Counter(
        str(row.get("frame_id", "")) for row in rejected
    )
    if set(observation_ids) & set(rejected_ids):
        errors.append("an instance cannot be both lifted and rejected")
    if set(record_ids) != set(observation_ids) | set(rejected_ids):
        errors.append("every SAM mask must be either lifted or rejected")

    for frame_id in selected_ids:
        summary = processed_by_id.get(frame_id, {})
        if int(summary.get("sam_instances", -1)) != mask_counts[frame_id]:
            errors.append(f"SAM count mismatch for {frame_id}")
        if int(summary.get("lifted_instances", -1)) != observation_counts[frame_id]:
            errors.append(f"lifted count mismatch for {frame_id}")
        if int(summary.get("rejected_instances", -1)) != rejected_counts[frame_id]:
            errors.append(f"rejected count mismatch for {frame_id}")

    frames_with_masks = [
        frame_id for frame_id in selected_ids if mask_counts[frame_id] > 0
    ]
    frames_with_lifted = [
        frame_id for frame_id in selected_ids if observation_counts[frame_id] > 0
    ]
    if result.get("frames_with_masks") != frames_with_masks:
        errors.append("frames_with_masks is inconsistent")
    if result.get("frames_with_lifted_observations") != frames_with_lifted:
        errors.append("frames_with_lifted_observations is inconsistent")
    if len(frames_with_lifted) < 2:
        errors.append("fewer than two frames produced valid 3D observations")
    if int(result.get("sam_instances", -1)) != len(records):
        errors.append("total SAM instance count is inconsistent")
    if int(result.get("lifted_instances", -1)) != len(observations):
        errors.append("total lifted instance count is inconsistent")

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "query": query,
        "selected_frames": selected_ids,
        "frames_with_masks": frames_with_masks,
        "frames_with_lifted_observations": frames_with_lifted,
        "mask_instances": len(records),
        "lifted_instances": len(observations),
        "rejected_instances": len(rejected),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    args = parser.parse_args()
    result = validate_output(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
