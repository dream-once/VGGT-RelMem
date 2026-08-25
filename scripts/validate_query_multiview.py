"""Validate query-specific multi-view evidence using the same frame pair."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from adapters.geometry import load_anchor_poses
from scripts.validate_d6 import validate_output as validate_d6_output
from scripts.validate_multiview_geometry import (
    DEFAULT_MIN_ROTATION_DEGREES,
    DEFAULT_MIN_TRANSLATION,
    _rotation_degrees,
)


NEGATIVE_STATUS = "INSUFFICIENT_MULTIFRAME_3D_EVIDENCE"


def measure_observation_pairs(
    poses: dict[str, np.ndarray],
    frame_ids: Sequence[str],
    *,
    min_translation: float,
    min_rotation_degrees: float,
) -> list[dict[str, Any]]:
    """Measure every observation-frame pair against both thresholds."""

    ordered = list(dict.fromkeys(frame_ids))
    missing = [frame_id for frame_id in ordered if frame_id not in poses]
    if missing:
        raise ValueError(f"missing anchor poses: {missing}")

    rows: list[dict[str, Any]] = []
    for left_id, right_id in combinations(ordered, 2):
        left = poses[left_id]
        right = poses[right_id]
        translation = float(
            np.linalg.norm(left[:3, 3] - right[:3, 3])
        )
        rotation = _rotation_degrees(left, right)
        rows.append(
            {
                "frame_pair": [left_id, right_id],
                "translation": translation,
                "rotation_degrees": rotation,
                "passes_translation": translation >= min_translation,
                "passes_rotation": rotation >= min_rotation_degrees,
                "passes_same_pair_gate": (
                    translation >= min_translation
                    and rotation >= min_rotation_degrees
                ),
            }
        )
    return rows


def _read_result(root: Path) -> dict[str, Any]:
    payload = json.loads(
        (root / "d6_result.json").read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise ValueError("d6_result.json root must be an object")
    return payload


def validate_query_multiview(
    d6_dir: str | Path,
    anchor_poses: str | Path,
    *,
    min_translation: float = DEFAULT_MIN_TRANSLATION,
    min_rotation_degrees: float = DEFAULT_MIN_ROTATION_DEGREES,
    expect_negative: bool = False,
) -> dict[str, Any]:
    """Validate positive true-multiview evidence or a zero-evidence negative."""

    if min_translation < 0 or min_rotation_degrees < 0:
        raise ValueError("viewpoint thresholds must be non-negative")

    root = Path(d6_dir)
    failures: list[str] = []
    try:
        result = _read_result(root)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {
            "status": "FAIL",
            "failures": [f"invalid D6 result: {error}"],
        }

    query = str(result.get("query", "")).strip()
    observation_frames = result.get(
        "frames_with_lifted_observations", []
    )
    if not isinstance(observation_frames, list):
        failures.append(
            "frames_with_lifted_observations must be a list"
        )
        observation_frames = []
    frame_ids = [str(frame_id) for frame_id in observation_frames]
    if len(frame_ids) != len(set(frame_ids)):
        failures.append("observation frame ids are not unique")

    pair_measurements: list[dict[str, Any]] = []
    passing_pairs: list[dict[str, Any]] = []
    if expect_negative:
        if result.get("stage") != "D6":
            failures.append("negative result stage is not D6")
        if result.get("status") != NEGATIVE_STATUS:
            failures.append(
                f"negative result status is {result.get('status')!r}"
            )
        if int(result.get("sam_instances", -1)) != 0:
            failures.append("negative control has SAM instances")
        if int(result.get("lifted_instances", -1)) != 0:
            failures.append("negative control has lifted instances")
        if result.get("frames_with_masks") not in ([], None):
            failures.append("negative control has mask-bearing frames")
        if frame_ids:
            failures.append("negative control has lifted observation frames")
        evidence_class = "NEGATIVE_CONTROL"
    else:
        d6_report = validate_d6_output(root)
        if d6_report.get("status") != "PASS":
            details = d6_report.get("errors", ["unknown D6 failure"])
            failures.append(f"base D6 validation failed: {details}")
        if len(frame_ids) < 2:
            failures.append(
                "query has fewer than two lifted-observation frames"
            )
        else:
            try:
                poses = load_anchor_poses(
                    anchor_poses,
                    required_frame_ids=frame_ids,
                )
                pair_measurements = measure_observation_pairs(
                    poses,
                    frame_ids,
                    min_translation=min_translation,
                    min_rotation_degrees=min_rotation_degrees,
                )
                passing_pairs = [
                    row
                    for row in pair_measurements
                    if row["passes_same_pair_gate"]
                ]
            except (OSError, TypeError, ValueError) as error:
                failures.append(
                    f"cannot measure query observation viewpoints: {error}"
                )
        if not passing_pairs:
            failures.append(
                "no single lifted-observation frame pair satisfies both "
                "viewpoint thresholds"
            )
        evidence_class = (
            "TRUE_MULTIVIEW"
            if passing_pairs
            else "MULTIFRAME_ONLY"
        )

    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "query": query,
        "expect_negative": expect_negative,
        "evidence_class": evidence_class,
        "frames_with_lifted_observations": frame_ids,
        "thresholds": {
            "min_translation": min_translation,
            "translation_unit": "unscaled_reconstruction_unit",
            "min_rotation_degrees": min_rotation_degrees,
            "same_pair_required": True,
        },
        "pair_measurements": pair_measurements,
        "passing_pairs": passing_pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("d6_dir")
    parser.add_argument("--anchor-poses", required=True)
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
    parser.add_argument("--expect-negative", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args()

    report = validate_query_multiview(
        args.d6_dir,
        args.anchor_poses,
        min_translation=args.min_translation,
        min_rotation_degrees=args.min_rotation_degrees,
        expect_negative=args.expect_negative,
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
