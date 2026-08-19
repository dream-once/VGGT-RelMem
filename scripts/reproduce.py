"""Build ObjectObservation caches and ObjectMemory from standard adapter files."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
import json
import shlex
import subprocess
import sys
import time

import numpy as np
import yaml

from adapters import load_geometry_npz, load_mask, load_mask_manifest
from relground.association import AssociationConfig, ObjectMemory
from relground.observations import LifterConfig, Robust3DLifter
from relground.schemas import RunManifest


def _filtered_config(config_type: type, values: dict) -> dict:
    allowed = {item.name for item in fields(config_type)}
    return {key: value for key, value in values.items() if key in allowed}


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "not-a-git-repository"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", required=True, help="VGGT geometry NPZ")
    parser.add_argument("--masks", required=True, help="Open-vocabulary mask manifest JSON")
    parser.add_argument("--output", required=True, help="Run output directory")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="development")
    args = parser.parse_args()

    started = time.perf_counter()
    config = yaml.safe_load(Path(args.config).read_text())
    geometry = load_geometry_npz(args.geometry)
    mask_records = load_mask_manifest(args.masks)
    lifter = Robust3DLifter(LifterConfig(**_filtered_config(LifterConfig, config.get("lifter", {}))))
    memory = ObjectMemory(
        AssociationConfig(**_filtered_config(AssociationConfig, config.get("association", {})))
    )

    output = Path(args.output)
    point_directory = output / "points"
    point_directory.mkdir(parents=True, exist_ok=True)
    observations = []
    failures = []
    for record in mask_records:
        try:
            frame = geometry.get(record.frame_id)
            point_path = point_directory / f"{record.obs_id}.npz"
            observation, points = lifter.make_observation(
                obs_id=record.obs_id,
                class_text=record.class_text,
                frame_id=record.frame_id,
                mask=load_mask(record, args.masks),
                point_map=frame.point_map,
                retrieval_score=record.retrieval_score,
                sam_score=record.sam_score,
                confidence_map=frame.confidence_map,
                world_from_camera=frame.world_from_camera,
                mask_ref=record.mask_ref,
                points_ref=str(point_path.relative_to(output)),
                semantic_embedding=None
                if record.semantic_embedding is None
                else np.asarray(record.semantic_embedding),
            )
            np.savez_compressed(point_path, points=points)
            observations.append(observation)
            memory.add_observation(observation)
        except (KeyError, ValueError) as error:
            failures.append({"obs_id": record.obs_id, "error": str(error)})

    output.mkdir(parents=True, exist_ok=True)
    (output / "observations.json").write_text(
        json.dumps([item.to_dict() for item in observations], ensure_ascii=False, indent=2) + "\n"
    )
    (output / "lifting_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n"
    )
    memory.save(output / "object_memory.json")
    manifest = RunManifest(
        git_sha=_git_sha(),
        env_lock="see env/ snapshots",
        dataset_split=args.split,
        seed=int(config.get("seed", 7)),
        config=config,
        command=shlex.join(sys.argv),
        runtime_seconds=time.perf_counter() - started,
    )
    manifest.save(output / "manifest.json")
    print(json.dumps({"observations": len(observations), "objects": len(memory), "failures": len(failures)}))


if __name__ == "__main__":
    main()
