"""Validate a self-contained D7 ObjectObservation scene cache without models."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from relground.observation_cache import (
    SCENE_OBSERVATION_CACHE_VERSION,
    file_inventory,
    load_observation_cache,
)
from relground.schemas import OBJECT_OBSERVATION_SCHEMA_VERSION
from relground.stage_video import VIDEO_MODE, VIDEO_SEGMENT_RATIOS


MIN_VIDEO_MOTION_RATIO = 0.25
MOTION_DIFFERENCE_THRESHOLD = 0.65


def safe_cache_artifact(root: Path, reference: str) -> Path:
    relative = Path(reference)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"artifact escapes scene cache: {reference}")
    path = (root.resolve() / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"artifact escapes scene cache: {reference}")
    return path


def probe_stage_video(path: Path) -> dict[str, float | int]:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is required to independently probe the D7 stage video"
        ) from error
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open stage video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if fps <= 0.0 or frame_count <= 0:
            raise ValueError("stage video has invalid FPS or frame count")
        sample_stride = max(1, int(round(fps / 2.0)))
        sampled = 0
        dynamic = 0
        differences: list[float] = []
        previous: np.ndarray | None = None
        frame_index = 0
        while True:
            readable, frame = capture.read()
            if not readable:
                break
            if frame_index % sample_stride == 0:
                gray = cv2.cvtColor(
                    cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2GRAY,
                )
                if previous is not None:
                    difference = float(
                        np.mean(
                            np.abs(
                                gray.astype(np.float32)
                                - previous.astype(np.float32)
                            )
                        )
                    )
                    differences.append(difference)
                    dynamic += difference >= MOTION_DIFFERENCE_THRESHOLD
                previous = gray
                sampled += 1
            frame_index += 1
        if sampled < 2 or not differences:
            raise ValueError("stage video has too few decodable samples")
        return {
            "duration_seconds": frame_count / fps,
            "fps": fps,
            "frame_count": frame_count,
            "motion_ratio": dynamic / len(differences),
            "mean_sample_difference": float(np.mean(differences)),
        }
    finally:
        capture.release()


def validate_output(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    errors: list[str] = []
    manifest_path = root / "scene_cache.json"
    observations_path = root / "observations.json"
    run_manifest_path = root / "run_manifest.json"
    for artifact in (manifest_path, observations_path, run_manifest_path):
        if not artifact.is_file() or artifact.stat().st_size == 0:
            errors.append(f"missing or empty artifact: {artifact.name}")
    if errors:
        return {"status": "FAIL", "errors": errors}

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cache = load_observation_cache(observations_path)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {"status": "FAIL", "errors": [f"invalid D7 cache: {error}"]}

    if manifest.get("status") != "PASS":
        errors.append(f"scene cache status is {manifest.get('status')!r}")
    if manifest.get("stage") != "D7":
        errors.append("scene cache stage is not D7")
    if manifest.get("schema_version") != SCENE_OBSERVATION_CACHE_VERSION:
        errors.append("scene cache manifest schema version is not frozen D7")
    if (
        manifest.get("observation_schema_version")
        != OBJECT_OBSERVATION_SCHEMA_VERSION
    ):
        errors.append("ObjectObservation schema version is not frozen D7")
    if manifest.get("scene_id") != cache.scene_id:
        errors.append("scene_id differs between cache files")
    if manifest.get("query") != cache.query:
        errors.append("query differs between cache files")
    if manifest.get("frame_ids") != cache.frame_ids:
        errors.append("frame_ids differ between cache files")
    if int(manifest.get("observation_count", -1)) != len(cache.observations):
        errors.append("observation_count is inconsistent")

    observation_counts = Counter(
        observation.frame_id for observation in cache.observations
    )
    expected_counts = {
        frame_id: observation_counts[frame_id] for frame_id in cache.frame_ids
    }
    if manifest.get("frame_observation_counts") != expected_counts:
        errors.append("frame_observation_counts is inconsistent")
    if len([value for value in expected_counts.values() if value > 0]) < 2:
        errors.append("cache does not contain multi-frame observations")

    expected_references = {"observations.json"}
    mask_shapes: set[tuple[int, ...]] = set()
    point_count = 0
    for observation in cache.observations:
        if not observation.mask_ref or not observation.points_ref:
            errors.append(f"observation is not self-contained: {observation.obs_id}")
            continue
        expected_references.update(
            (observation.mask_ref, observation.points_ref)
        )
        try:
            mask_path = safe_cache_artifact(root, observation.mask_ref)
            mask = np.asarray(np.load(mask_path, allow_pickle=False), dtype=bool)
            mask_shapes.add(mask.shape)
            if mask.ndim != 2 or not mask.any():
                errors.append(f"invalid or empty mask: {observation.obs_id}")
        except (OSError, ValueError) as error:
            errors.append(f"cannot load mask {observation.obs_id}: {error}")
        try:
            points_path = safe_cache_artifact(root, observation.points_ref)
            with np.load(points_path, allow_pickle=False) as archive:
                points = np.asarray(archive["points"])
            if (
                points.ndim != 2
                or points.shape[1] != 3
                or len(points) < 3
                or not np.all(np.isfinite(points))
            ):
                errors.append(f"invalid points: {observation.obs_id}")
            else:
                point_count += len(points)
        except (OSError, KeyError, ValueError) as error:
            errors.append(f"cannot load points {observation.obs_id}: {error}")

    previews = manifest.get("previews", [])
    if not isinstance(previews, list) or len(previews) != len(cache.frame_ids):
        errors.append("preview list does not match cached frames")
        previews = []
    for reference in previews:
        expected_references.add(str(reference))
        try:
            preview_path = safe_cache_artifact(root, str(reference))
            if not preview_path.is_file() or preview_path.stat().st_size == 0:
                errors.append(f"missing or empty preview: {reference}")
        except (OSError, ValueError) as error:
            errors.append(f"invalid preview {reference}: {error}")

    video = manifest.get("stage_video", {})
    video_reference = str(video.get("path", ""))
    if video_reference:
        expected_references.add(video_reference)
    if video.get("mode") != VIDEO_MODE:
        errors.append("stage video is not the dynamic pipeline format")
    segments = video.get("segments", {})
    expected_segment_names = {
        name for name, _ in VIDEO_SEGMENT_RATIOS
    }
    if (
        not isinstance(segments, dict)
        or set(segments) != expected_segment_names
    ):
        errors.append("stage video segment manifest is incomplete")
        segments = {}
    else:
        try:
            segment_duration = sum(float(value) for value in segments.values())
            if any(float(value) <= 0.0 for value in segments.values()):
                errors.append("stage video segment durations must be positive")
            if not np.isclose(
                segment_duration,
                float(video.get("duration_seconds", -1.0)),
                atol=0.25,
            ):
                errors.append("stage video segments do not cover its duration")
        except (TypeError, ValueError):
            errors.append("stage video segment durations are invalid")
    try:
        video_path = safe_cache_artifact(root, video_reference)
        probe = probe_stage_video(video_path)
        duration = float(probe["duration_seconds"])
        if not 30.0 <= duration <= 60.0:
            errors.append(f"stage video duration is outside D7 gate: {duration}")
        recorded_duration = float(video.get("duration_seconds", -1.0))
        if not np.isclose(duration, recorded_duration, atol=0.25):
            errors.append("stage video duration differs from manifest")
        motion_ratio = float(probe["motion_ratio"])
        if motion_ratio < MIN_VIDEO_MOTION_RATIO:
            errors.append(
                "stage video is too static: "
                f"motion ratio {motion_ratio:.3f} < {MIN_VIDEO_MOTION_RATIO:.3f}"
            )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        errors.append(f"invalid stage video: {error}")
        duration = None
        motion_ratio = None

    raw_inventory = manifest.get("files", [])
    if not isinstance(raw_inventory, list):
        errors.append("file inventory must be a list")
        raw_inventory = []
    inventory_references = [
        str(record.get("path", ""))
        for record in raw_inventory
        if isinstance(record, dict)
    ]
    if len(inventory_references) != len(raw_inventory):
        errors.append("file inventory entries must be objects")
    if set(inventory_references) != expected_references:
        errors.append("file inventory does not cover exactly the cache artifacts")
    try:
        actual_inventory = {
            record["path"]: record
            for record in file_inventory(root, inventory_references)
        }
        for expected in raw_inventory:
            reference = str(expected["path"])
            actual = actual_inventory[reference]
            if int(expected.get("size_bytes", -1)) != actual["size_bytes"]:
                errors.append(f"size mismatch: {reference}")
            if str(expected.get("sha256", "")) != actual["sha256"]:
                errors.append(f"SHA-256 mismatch: {reference}")
    except (OSError, KeyError, TypeError, ValueError) as error:
        errors.append(f"invalid file inventory: {error}")

    source_hash = str(manifest.get("source", {}).get("d6_result_sha256", ""))
    if cache.metadata.get("source_d6_result_sha256") != source_hash:
        errors.append("D6 source hash differs between cache files")

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "scene_id": cache.scene_id,
        "query": cache.query,
        "frames": len(cache.frame_ids),
        "observations": len(cache.observations),
        "points": point_count,
        "mask_shapes": [list(shape) for shape in sorted(mask_shapes)],
        "video_duration_seconds": duration,
        "video_motion_ratio": motion_ratio,
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
