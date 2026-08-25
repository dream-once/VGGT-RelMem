"""Validate a VGGT-SLAM geometry export and its provenance files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import json

import numpy as np

from adapters.geometry import (
    GEOMETRY_SCHEMA_VERSION,
    LEGACY_GEOMETRY_SCHEMA_VERSION,
    load_geometry_npz,
)


def _load_json(path: Path, failures: list[str]) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        failures.append(f"missing companion file: {path}")
    except (OSError, json.JSONDecodeError) as error:
        failures.append(f"invalid companion file {path}: {error}")
    return None


def validate_geometry(
    geometry_path: str | Path,
    *,
    min_frames: int = 2,
    require_run_manifest: bool = True,
) -> dict[str, Any]:
    path = Path(geometry_path)
    failures: list[str] = []
    try:
        geometry = load_geometry_npz(path)
    except (OSError, TypeError, ValueError) as error:
        return {
            "status": "FAIL",
            "geometry_path": str(path.resolve()),
            "schema_version": None,
            "frame_count": 0,
            "point_map_shape": [],
            "raw_confidence_available": False,
            "raw_confidence_min": None,
            "raw_confidence_max": None,
            "failures": [f"invalid geometry archive: {error}"],
        }

    finite_points = np.all(
        np.isfinite(geometry.point_maps),
        axis=-1,
    )
    finite_confidence = np.isfinite(geometry.confidence_maps)
    valid_mask = np.asarray(geometry.valid_masks, dtype=bool)
    valid_points = finite_points & valid_mask
    valid_ratios = valid_points.reshape(
        len(geometry.frame_ids),
        -1,
    ).mean(axis=1)

    if geometry.schema_version not in {
        LEGACY_GEOMETRY_SCHEMA_VERSION,
        GEOMETRY_SCHEMA_VERSION,
    }:
        failures.append(
            f"unsupported geometry schema: {geometry.schema_version!r}"
        )
    if len(geometry.frame_ids) < min_frames:
        failures.append(
            f"expected at least {min_frames} frames, "
            f"got {len(geometry.frame_ids)}"
        )
    if not np.all(np.isfinite(geometry.world_from_camera)):
        failures.append("world_from_camera contains NaN or Inf")
    if not np.all(finite_confidence):
        failures.append("confidence_maps contains NaN or Inf")
    if finite_confidence.any():
        confidence = geometry.confidence_maps[finite_confidence]
        if confidence.min() < 0 or confidence.max() > 1:
            failures.append(
                "confidence_maps values must be in [0, 1]"
            )
    if not np.array_equal(
        geometry.confidence_maps > 0,
        valid_mask,
    ):
        failures.append(
            "confidence_maps binary alias differs from valid_masks"
        )
    if not np.any(valid_points):
        failures.append("no finite point is marked valid")

    raw_confidence_available = (
        geometry.raw_confidence_maps is not None
    )
    raw_confidence_min: float | None = None
    raw_confidence_max: float | None = None
    if geometry.schema_version == GEOMETRY_SCHEMA_VERSION:
        if not raw_confidence_available:
            failures.append(
                "geometry 0.2 lacks raw_confidence_maps"
            )
        if geometry.valid_masks.dtype != np.bool_:
            failures.append("geometry 0.2 valid_masks are not boolean")
    if raw_confidence_available:
        raw_confidence = geometry.raw_confidence_maps
        finite_raw = np.isfinite(raw_confidence)
        if not np.all(finite_raw):
            failures.append(
                "raw_confidence_maps contains NaN or Inf"
            )
        if finite_raw.any():
            finite_values = raw_confidence[finite_raw]
            raw_confidence_min = float(finite_values.min())
            raw_confidence_max = float(finite_values.max())

    export_manifest = _load_json(
        path.with_suffix(".manifest.json"),
        failures,
    )
    anchor_poses = _load_json(
        path.with_suffix(".anchor_poses.json"),
        failures,
    )
    run_manifest_path = path.parent / "run_manifest.json"
    run_manifest = (
        _load_json(run_manifest_path, failures)
        if require_run_manifest
        else None
    )

    frame_ids = set(geometry.frame_ids)
    if isinstance(export_manifest, dict):
        manifest_ids = {
            str(item.get("frame_id"))
            for item in export_manifest.get("frames", [])
            if isinstance(item, dict)
        }
        if manifest_ids != frame_ids:
            failures.append(
                "export manifest frame IDs do not match geometry.npz"
            )
        if (
            export_manifest.get("schema_version")
            != geometry.schema_version
        ):
            failures.append(
                "export manifest schema differs from geometry.npz"
            )
        if geometry.schema_version == GEOMETRY_SCHEMA_VERSION:
            encoding = export_manifest.get("confidence_encoding")
            required_encodings = {
                "raw_confidence_maps",
                "valid_masks",
                "confidence_maps",
            }
            if (
                not isinstance(encoding, dict)
                or set(encoding) != required_encodings
            ):
                failures.append(
                    "export manifest lacks geometry 0.2 "
                    "confidence semantics"
                )
    if (
        isinstance(anchor_poses, dict)
        and set(anchor_poses) != frame_ids
    ):
        failures.append(
            "anchor pose frame IDs do not match geometry.npz"
        )
    if require_run_manifest and isinstance(run_manifest, dict):
        source_commit = (
            run_manifest.get("config", {}).get("upstream_commit")
        )
        if (
            not isinstance(source_commit, str)
            or len(source_commit) != 40
        ):
            failures.append(
                "run manifest lacks a pinned "
                "40-character upstream commit"
            )

    return {
        "status": "PASS" if not failures else "FAIL",
        "geometry_path": str(path.resolve()),
        "schema_version": geometry.schema_version,
        "frame_count": len(geometry.frame_ids),
        "point_map_shape": list(geometry.point_maps.shape),
        "valid_point_ratio_min": float(valid_ratios.min()),
        "valid_point_ratio_mean": float(valid_ratios.mean()),
        "raw_confidence_available": raw_confidence_available,
        "raw_confidence_min": raw_confidence_min,
        "raw_confidence_max": raw_confidence_max,
        "companion_files": {
            "export_manifest": str(
                path.with_suffix(".manifest.json")
            ),
            "anchor_poses": str(
                path.with_suffix(".anchor_poses.json")
            ),
            "run_manifest": str(run_manifest_path),
        },
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("geometry")
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--skip-run-manifest", action="store_true")
    args = parser.parse_args()

    report = validate_geometry(
        args.geometry,
        min_frames=args.min_frames,
        require_run_manifest=not args.skip_run_manifest,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
