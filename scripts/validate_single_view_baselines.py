"""Validate controlled B0-official/B1-robust-single-view artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from adapters.masks import load_mask_manifest
from relground.observation_cache import sha256_file
from relground.schemas import ObjectObservation
from relground.single_view import (
    BASELINE_SCHEMA_VERSION,
    B0_OFFICIAL,
    B1_ROBUST_SINGLE_VIEW,
    SUPPORTED_SINGLE_VIEW_BASELINES,
    VGGTImageTransform,
)


def safe_artifact(root: Path, reference: str) -> Path:
    relative = Path(reference)
    if not reference or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe artifact reference: {reference!r}")
    path = (root.resolve() / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"artifact escapes output directory: {reference!r}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} root must be an object")
    return payload


def validate_preprocess(
    root: Path,
    result: dict[str, Any],
    *,
    expected_frame: str,
    errors: list[str],
) -> tuple[int, int] | None:
    controlled = result.get("controlled_inputs", {})
    if not isinstance(controlled, dict):
        errors.append("controlled_inputs must be an object")
        return None
    if controlled.get("mask_resizing_after_sam") is not False:
        errors.append("SAM masks were resized after inference")
    try:
        preprocess_path = safe_artifact(
            root,
            str(controlled.get("preprocess", "")),
        )
        sam_input_path = safe_artifact(
            root,
            str(controlled.get("sam_input", "")),
        )
        if not preprocess_path.is_file() or not sam_input_path.is_file():
            raise FileNotFoundError("preprocess.json or SAM input is missing")
        if sha256_file(preprocess_path) != controlled.get(
            "preprocess_sha256"
        ):
            errors.append("preprocess artifact hash mismatch")
        if sha256_file(sam_input_path) != controlled.get(
            "sam_input_sha256"
        ):
            errors.append("SAM input hash mismatch")
        preprocess = read_json(preprocess_path)
        if preprocess.get("frame_id") != expected_frame:
            errors.append("preprocess frame differs from top-1 frame")
        transform = VGGTImageTransform.from_dict(preprocess["transform"])
        recorded_shape = tuple(
            int(value) for value in controlled.get("sam_mask_shape", [])
        )
        if recorded_shape != transform.output_shape:
            errors.append("SAM mask shape differs from VGGT transform output")
        from PIL import Image

        with Image.open(sam_input_path) as image:
            if tuple(image.size) != transform.output_size:
                errors.append("saved SAM input size differs from transform")
        return transform.output_shape
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        errors.append(f"invalid controlled preprocessing: {error}")
        return None


def validate_output(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    errors: list[str] = []
    result_path = root / "single_view_result.json"
    mask_manifest_path = root / "masks.json"
    run_manifest_path = root / "run_manifest.json"
    preview_path = root / "preview.png"
    required = (
        result_path,
        mask_manifest_path,
        run_manifest_path,
        preview_path,
    )
    for artifact in required:
        if not artifact.is_file() or artifact.stat().st_size == 0:
            errors.append(f"missing or empty artifact: {artifact.name}")
    if errors:
        return {"status": "FAIL", "errors": errors}

    try:
        result = read_json(result_path)
        run_manifest = read_json(run_manifest_path)
        records = load_mask_manifest(mask_manifest_path)
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        return {"status": "FAIL", "errors": [f"invalid root artifact: {error}"]}

    if result.get("schema_version") != BASELINE_SCHEMA_VERSION:
        errors.append("single-view schema version is not supported")
    if result.get("status") != "PASS":
        errors.append(f"single_view_result status is {result.get('status')!r}")
    if result.get("stage") != "D4-controlled-correction":
        errors.append("result is not the controlled D4 correction")
    query = str(result.get("query", "")).strip()
    top1 = result.get("top1", {})
    if not query or not isinstance(top1, dict):
        errors.append("result query or top1 is invalid")
        top1 = {}
    frame_id = str(top1.get("frame_id", ""))
    retrieval_score = top1.get("retrieval_score")
    try:
        retrieval_score = float(retrieval_score)
        if not np.isfinite(retrieval_score) or not 0.0 <= retrieval_score <= 1.0:
            raise ValueError
    except (TypeError, ValueError):
        errors.append("top-1 retrieval score is not finite in [0,1]")
        retrieval_score = None

    expected_shape = validate_preprocess(
        root,
        result,
        expected_frame=frame_id,
        errors=errors,
    )
    controlled = result.get("controlled_inputs", {})
    if int(controlled.get("sam_instances", -1)) != len(records):
        errors.append("SAM instance count differs from mask manifest")
    if not records:
        errors.append("shared SAM mask manifest is empty")

    mask_by_id = {record.obs_id: record for record in records}
    if len(mask_by_id) != len(records):
        errors.append("shared SAM instance ids are not unique")
    mask_shapes: set[tuple[int, ...]] = set()
    for record in records:
        if record.frame_id != frame_id:
            errors.append(f"mask frame mismatch: {record.obs_id}")
        if record.class_text != query:
            errors.append(f"mask query mismatch: {record.obs_id}")
        if retrieval_score is not None and not np.isclose(
            record.retrieval_score,
            retrieval_score,
            atol=1e-12,
        ):
            errors.append(f"mask retrieval score mismatch: {record.obs_id}")
        try:
            mask_path = safe_artifact(root, record.mask_ref)
            mask = np.load(mask_path, allow_pickle=False)
            if mask.dtype != np.bool_:
                errors.append(f"mask is not boolean: {record.obs_id}")
            mask = np.asarray(mask, dtype=bool)
            mask_shapes.add(mask.shape)
            if mask.ndim != 2 or not mask.any():
                errors.append(f"mask is invalid or empty: {record.obs_id}")
            if expected_shape is not None and mask.shape != expected_shape:
                errors.append(
                    f"mask does not directly match VGGT grid: {record.obs_id}"
                )
        except (OSError, ValueError) as error:
            errors.append(f"cannot load mask {record.obs_id}: {error}")

    baseline_result = result.get("baselines", {})
    if not isinstance(baseline_result, dict) or set(baseline_result) != set(
        SUPPORTED_SINGLE_VIEW_BASELINES
    ):
        errors.append("result must contain exactly the controlled B0/B1 pair")
        baseline_result = {}

    lifted_ids: dict[str, set[str]] = {}
    lifted_counts: dict[str, int] = {}
    point_counts: dict[str, int] = {}
    for baseline_id in SUPPORTED_SINGLE_VIEW_BASELINES:
        record = baseline_result.get(baseline_id, {})
        if not isinstance(record, dict):
            errors.append(f"invalid baseline record: {baseline_id}")
            continue
        try:
            observations_path = safe_artifact(
                root,
                str(record.get("observations", "")),
            )
            payload = read_json(observations_path)
            if set(payload) != {
                "schema_version",
                "baseline_id",
                "observations",
            }:
                errors.append(
                    f"observation fields are not frozen: {baseline_id}"
                )
            if payload.get("schema_version") != BASELINE_SCHEMA_VERSION:
                errors.append(
                    f"observation schema version mismatch: {baseline_id}"
                )
            if payload.get("baseline_id") != baseline_id:
                errors.append(f"observation baseline mismatch: {baseline_id}")
            raw_observations = payload.get("observations", [])
            if not isinstance(raw_observations, list):
                raise ValueError("observations must be a list")
            observations = [
                ObjectObservation.from_dict(value)
                for value in raw_observations
            ]
        except (
            FileNotFoundError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            errors.append(f"cannot load {baseline_id} observations: {error}")
            continue

        if int(record.get("lifted_instances", -1)) != len(observations):
            errors.append(f"lifted count mismatch: {baseline_id}")
        if not observations:
            errors.append(f"no valid observation: {baseline_id}")
        shared_ids: set[str] = set()
        baseline_points = 0
        for observation in observations:
            if observation.frame_id != frame_id:
                errors.append(f"observation frame mismatch: {observation.obs_id}")
            if observation.class_text != query:
                errors.append(f"observation query mismatch: {observation.obs_id}")
            if observation.metadata.get("baseline_id") != baseline_id:
                errors.append(
                    f"observation baseline metadata mismatch: {observation.obs_id}"
                )
            shared_id = str(
                observation.metadata.get("shared_instance_id", "")
            )
            if shared_id not in mask_by_id:
                errors.append(
                    f"observation has no shared SAM instance: {observation.obs_id}"
                )
            else:
                shared_ids.add(shared_id)
                if observation.mask_ref != mask_by_id[shared_id].mask_ref:
                    errors.append(
                        f"observation mask reference mismatch: {observation.obs_id}"
                    )
                if not np.isclose(
                    observation.sam_score,
                    mask_by_id[shared_id].sam_score,
                    atol=1e-12,
                ):
                    errors.append(
                        f"observation SAM score mismatch: {observation.obs_id}"
                    )
            if retrieval_score is not None and not np.isclose(
                observation.retrieval_score,
                retrieval_score,
                atol=1e-12,
            ):
                errors.append(
                    f"observation retrieval mismatch: {observation.obs_id}"
                )
            try:
                points_path = safe_artifact(
                    root,
                    str(observation.points_ref or ""),
                )
                with np.load(points_path, allow_pickle=False) as archive:
                    points = np.asarray(archive["points"])
                if (
                    points.ndim != 2
                    or points.shape[1:] != (3,)
                    or len(points) < 2
                    or not np.all(np.isfinite(points))
                ):
                    errors.append(
                        f"invalid or non-finite points: {observation.obs_id}"
                    )
                else:
                    baseline_points += len(points)
            except (OSError, KeyError, ValueError) as error:
                errors.append(
                    f"cannot load points {observation.obs_id}: {error}"
                )
        lifted_ids[baseline_id] = shared_ids
        lifted_counts[baseline_id] = len(observations)
        point_counts[baseline_id] = baseline_points

    if set(lifted_ids) == set(SUPPORTED_SINGLE_VIEW_BASELINES):
        if not lifted_ids[B0_OFFICIAL] & lifted_ids[B1_ROBUST_SINGLE_VIEW]:
            errors.append("B0 and B1 have no commonly lifted SAM instance")
    config = run_manifest.get("config", {})
    if not isinstance(config, dict) or config.get("query") != query:
        errors.append("run manifest query differs from result")
    if config.get("pipeline") != "controlled single-view B0/B1":
        errors.append("run manifest pipeline is not controlled B0/B1")

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "query": query,
        "top1_frame": frame_id,
        "sam_instances": len(records),
        "mask_shapes": [list(shape) for shape in sorted(mask_shapes)],
        "lifted_instances": lifted_counts,
        "point_counts": point_counts,
        "shared_lifted_instances": len(
            lifted_ids.get(B0_OFFICIAL, set())
            & lifted_ids.get(B1_ROBUST_SINGLE_VIEW, set())
        ),
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
