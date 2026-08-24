"""Build a self-contained D7 scene observation cache from validated D6 output."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any

from relground.observation_cache import (
    SCENE_OBSERVATION_CACHE_VERSION,
    SceneObservationCache,
    file_inventory,
    save_observation_cache,
    sha256_file,
)
from relground.schemas import (
    OBJECT_OBSERVATION_SCHEMA_VERSION,
    ObjectObservation,
    RunManifest,
)
from relground.stage_video import image_files, render_stage_video
from scripts.validate_d6 import validate_output as validate_d6_output


def git_commit(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def safe_source(root: Path, reference: str) -> Path:
    relative = Path(reference)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"source artifact escapes D6 directory: {reference}")
    source = (root / relative).resolve()
    if root.resolve() not in source.parents:
        raise ValueError(f"source artifact escapes D6 directory: {reference}")
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty D6 artifact: {reference}")
    return source


def copy_artifact(source_root: Path, output_root: Path, reference: str) -> str:
    source = safe_source(source_root, reference)
    destination = output_root / reference
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return reference


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    project_root = Path(args.project_root).resolve()
    d6_dir = Path(args.d6_dir).resolve()
    image_folder = Path(args.image_folder).resolve()
    output_dir = Path(args.output_dir).resolve()

    validation = validate_d6_output(d6_dir)
    if validation["status"] != "PASS":
        raise ValueError(f"D6 input failed validation: {validation['errors']}")
    d6_result_path = d6_dir / "d6_result.json"
    d6_result = json.loads(d6_result_path.read_text(encoding="utf-8"))
    d6_observation_payload = json.loads(
        (d6_dir / "observations.json").read_text(encoding="utf-8")
    )
    raw_observations = d6_observation_payload.get("observations", [])
    observations = [
        ObjectObservation.from_dict(raw) for raw in raw_observations
    ]
    frame_ids = [
        str(row["frame_id"]) for row in d6_result["selected_frames"]
    ]
    query = str(d6_result["query"]).strip()
    if len({observation.frame_id for observation in observations}) < 2:
        raise ValueError("D7 input lacks multi-frame 3D observations")
    if not 30.0 <= args.video_duration <= 60.0:
        raise ValueError(
            "stage video target duration must be between 30 and 60 seconds"
        )
    available_images = image_files(image_folder)
    max_geometry_index = max(
        int(row["geometry_index"])
        for row in d6_result["selected_frames"]
    )
    if max_geometry_index >= len(available_images):
        raise ValueError(
            "image folder does not cover all selected geometry frames"
        )
    input_paths = available_images[: max_geometry_index + 1]
    for row in d6_result["selected_frames"]:
        geometry_index = int(row["geometry_index"])
        if input_paths[geometry_index].stem != str(row["frame_id"]):
            raise ValueError(
                "image folder order does not match D6 geometry frame IDs"
            )
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; choose a new directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_references: list[str] = []
    for observation in observations:
        if not observation.mask_ref or not observation.points_ref:
            raise ValueError(
                f"observation is not self-contained: {observation.obs_id}"
            )
        copied_references.append(
            copy_artifact(d6_dir, output_dir, observation.mask_ref)
        )
        copied_references.append(
            copy_artifact(d6_dir, output_dir, observation.points_ref)
        )

    preview_references: list[str] = []
    observation_counts = Counter(
        observation.frame_id for observation in observations
    )
    for row in d6_result["processed_frames"]:
        preview_reference = copy_artifact(
            d6_dir,
            output_dir,
            str(row["preview"]),
        )
        preview_references.append(preview_reference)

    cache = SceneObservationCache(
        scene_id=args.scene_id,
        query=query,
        source_stage="D6",
        frame_ids=frame_ids,
        observations=observations,
        metadata={
            "source_d6_result_sha256": sha256_file(d6_result_path),
            "source_d6_observation_schema": str(
                d6_observation_payload.get("schema_version", "unknown")
            ),
            "lifter_config": d6_result["lifter_config"],
        },
    )
    observations_reference = "observations.json"
    observations_path = output_dir / observations_reference
    save_observation_cache(observations_path, cache)

    video_reference = "stage_video.mp4"
    video_path = output_dir / video_reference
    video_info = render_stage_video(
        input_paths,
        d6_result["selected_frames"],
        [output_dir / reference for reference in preview_references],
        observations,
        output_dir,
        video_path,
        fps=args.video_fps,
        duration_seconds=args.video_duration,
        codec=args.video_codec,
    )
    duration_seconds = float(video_info["duration_seconds"])
    if not 30.0 <= duration_seconds <= 60.0:
        raise RuntimeError(
            f"created video duration is outside D7 gate: {duration_seconds}"
        )

    inventory_references = list(
        dict.fromkeys(
            [
                observations_reference,
                *copied_references,
                *preview_references,
                video_reference,
            ]
        )
    )
    files = file_inventory(output_dir, inventory_references)
    manifest = {
        "schema_version": SCENE_OBSERVATION_CACHE_VERSION,
        "status": "PASS",
        "stage": "D7",
        "scene_id": args.scene_id,
        "query": query,
        "observation_schema_version": OBJECT_OBSERVATION_SCHEMA_VERSION,
        "source": {
            "stage": "D6",
            "directory": str(d6_dir),
            "d6_result_sha256": sha256_file(d6_result_path),
        },
        "frame_ids": frame_ids,
        "observation_count": len(observations),
        "frame_observation_counts": {
            frame_id: observation_counts[frame_id] for frame_id in frame_ids
        },
        "previews": preview_references,
        "stage_video": {
            "path": video_reference,
            **video_info,
        },
        "files": files,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_json(output_dir / "scene_cache.json", manifest)

    RunManifest(
        git_sha=git_commit(project_root),
        env_lock="D7 cache is model-free; OpenCV comes from vggt_geom",
        dataset_split=args.scene_id,
        seed=0,
        config={
            "pipeline": "D7 frozen ObjectObservation scene cache",
            "d6_dir": str(d6_dir),
            "scene_id": args.scene_id,
            "query": query,
            "observation_schema_version": OBJECT_OBSERVATION_SCHEMA_VERSION,
            "video_fps": args.video_fps,
            "video_duration": args.video_duration,
            "video_codec": args.video_codec,
            "image_folder": str(image_folder),
        },
        command=shlex.join(sys.argv),
        runtime_seconds=time.perf_counter() - started,
        peak_vram_mb=None,
    ).save(output_dir / "run_manifest.json")
    summary = {
        "status": "PASS",
        "stage": "D7",
        "scene_id": args.scene_id,
        "query": query,
        "frames": len(frame_ids),
        "observations": len(observations),
        "video_mode": video_info["mode"],
        "video_duration_seconds": duration_seconds,
        "output_dir": str(output_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--d6-dir",
        default="runs/office-loop-d6-trash-can",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/office-loop-d7-trash-can",
    )
    parser.add_argument("--scene-id", default="office-loop-trash-can")
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--image-folder", default="data/office_loop")
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--video-duration", type=float, default=40.0)
    parser.add_argument("--video-codec", default="mp4v")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.scene_id.strip():
        raise ValueError("scene-id must not be empty")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
