"""Validate that a VGGT geometry export contains separated viewpoints."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from adapters.geometry import load_anchor_poses, load_geometry_npz
from scripts.validate_geometry import validate_geometry


DEFAULT_MIN_TRANSLATION = 0.5
DEFAULT_MIN_ROTATION_DEGREES = 3.0


def _rotation_degrees(left: np.ndarray, right: np.ndarray) -> float:
    relative = left[:3, :3].T @ right[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def measure_viewpoint_spread(
    poses: dict[str, np.ndarray],
    frame_ids: Sequence[str],
) -> dict[str, Any]:
    """Return the maximum pairwise translation and rotation baselines."""

    ordered = list(frame_ids)
    if len(ordered) < 2:
        raise ValueError("at least two frames are required")
    missing = [frame_id for frame_id in ordered if frame_id not in poses]
    if missing:
        raise ValueError(f"missing anchor poses: {missing}")

    max_translation = -1.0
    max_rotation = -1.0
    translation_pair: tuple[str, str] | None = None
    rotation_pair: tuple[str, str] | None = None
    for left_id, right_id in combinations(ordered, 2):
        left = poses[left_id]
        right = poses[right_id]
        translation = float(
            np.linalg.norm(left[:3, 3] - right[:3, 3])
        )
        rotation = _rotation_degrees(left, right)
        if translation > max_translation:
            max_translation = translation
            translation_pair = (left_id, right_id)
        if rotation > max_rotation:
            max_rotation = rotation
            rotation_pair = (left_id, right_id)

    return {
        "max_translation": max_translation,
        "max_translation_pair": list(translation_pair or ()),
        "max_rotation_degrees": max_rotation,
        "max_rotation_pair": list(rotation_pair or ()),
        "translation_unit": "unscaled_reconstruction_unit",
    }


def validate_multiview_geometry(
    geometry_path: str | Path,
    *,
    min_frames: int = 8,
    min_translation: float = DEFAULT_MIN_TRANSLATION,
    min_rotation_degrees: float = DEFAULT_MIN_ROTATION_DEGREES,
    require_run_manifest: bool = True,
) -> dict[str, Any]:
    if min_translation < 0 or min_rotation_degrees < 0:
        raise ValueError("viewpoint thresholds must be non-negative")

    report = validate_geometry(
        geometry_path,
        min_frames=min_frames,
        require_run_manifest=require_run_manifest,
    )
    failures = list(report["failures"])
    path = Path(geometry_path)
    spread: dict[str, Any] | None = None
    try:
        geometry = load_geometry_npz(path)
        poses = load_anchor_poses(
            path.with_suffix(".anchor_poses.json"),
            required_frame_ids=geometry.frame_ids,
        )
        spread = measure_viewpoint_spread(poses, geometry.frame_ids)
    except (OSError, TypeError, ValueError) as error:
        failures.append(f"cannot measure viewpoint spread: {error}")

    if spread is not None:
        if spread["max_translation"] < min_translation:
            failures.append(
                "translation baseline is too small: "
                f"{spread['max_translation']:.6f} < {min_translation:.6f}"
            )
        if spread["max_rotation_degrees"] < min_rotation_degrees:
            failures.append(
                "rotation baseline is too small: "
                f"{spread['max_rotation_degrees']:.6f} "
                f"< {min_rotation_degrees:.6f}"
            )

    report["multiview_gate"] = {
        "min_translation": min_translation,
        "min_rotation_degrees": min_rotation_degrees,
        "spread": spread,
    }
    report["failures"] = failures
    report["status"] = "PASS" if not failures else "FAIL"
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("geometry")
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument(
        "--min-translation",
        type=float,
        default=DEFAULT_MIN_TRANSLATION,
    )
    parser.add_argument(
        "--min-rotation-degrees",
        type=float,
        default=DEFAULT_MIN_ROTATION_DEGREES,
    )
    parser.add_argument("--skip-run-manifest", action="store_true")
    args = parser.parse_args()

    report = validate_multiview_geometry(
        args.geometry,
        min_frames=args.min_frames,
        min_translation=args.min_translation,
        min_rotation_degrees=args.min_rotation_degrees,
        require_run_manifest=not args.skip_run_manifest,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
