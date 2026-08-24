"""Validate historical D4 artifacts originally labelled B0.

Use validate_single_view_baselines for the strict controlled B0/B1 pair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from adapters.masks import load_mask, load_mask_manifest
from relground.schemas import ObjectObservation


def validate_output(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    errors: list[str] = []
    result_path = root / "b0_result.json"
    observations_path = root / "observations.json"
    mask_manifest_path = root / "masks.json"
    preview_path = root / "preview.png"
    run_manifest_path = root / "run_manifest.json"
    required = [result_path, observations_path, mask_manifest_path, preview_path, run_manifest_path]
    for item in required:
        if not item.is_file() or item.stat().st_size == 0:
            errors.append(f"missing or empty artifact: {item.name}")
    if errors:
        return {"status": "FAIL", "errors": errors}

    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS":
        errors.append(f"b0_result status is {result.get('status')!r}")
    records = load_mask_manifest(mask_manifest_path)
    observation_payload = json.loads(observations_path.read_text(encoding="utf-8"))
    observations = [
        ObjectObservation.from_dict(value)
        for value in observation_payload.get("observations", [])
    ]
    if not records:
        errors.append("mask manifest contains no SAM instances")
    if not observations:
        errors.append("observations.json contains no valid 3D observation")
    frame_ids = {record.frame_id for record in records}
    if len(frame_ids) > 1:
        errors.append("B0 masks must all come from one top-1 frame")

    mask_shapes: set[tuple[int, ...]] = set()
    for record in records:
        try:
            mask = load_mask(record, mask_manifest_path)
            mask_shapes.add(mask.shape)
            if mask.ndim != 2 or not mask.any():
                errors.append(f"invalid or empty mask: {record.obs_id}")
        except (OSError, ValueError) as error:
            errors.append(f"cannot load mask {record.obs_id}: {error}")
    for observation in observations:
        if not observation.points_ref:
            errors.append(f"observation has no points_ref: {observation.obs_id}")
            continue
        point_path = root / observation.points_ref
        try:
            with np.load(point_path, allow_pickle=False) as archive:
                points = np.asarray(archive["points"])
            if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
                errors.append(f"invalid lifted points: {observation.obs_id}")
        except (OSError, KeyError, ValueError) as error:
            errors.append(f"cannot load points {observation.obs_id}: {error}")

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "query": result.get("query"),
        "top1_frame": result.get("top1", {}).get("frame_id"),
        "mask_instances": len(records),
        "lifted_instances": len(observations),
        "mask_shapes": [list(shape) for shape in sorted(mask_shapes)],
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
