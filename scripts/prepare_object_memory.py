"""Build a frozen D8 ObjectMemory envelope from a D7 observation cache."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

from relground.association import ObjectMemory
from relground.observation_cache import load_observation_cache, sha256_file
from relground.schemas import (
    MEMORY_OBJECT_SCHEMA_VERSION,
    OBJECT_MEMORY_SCHEMA_VERSION,
    OBJECT_OBSERVATION_SCHEMA_VERSION,
    RunManifest,
)


def git_commit(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return (
        completed.stdout.strip()
        if completed.returncode == 0
        else "unknown"
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    project_root = Path(args.project_root).resolve()
    cache_path = Path(args.cache).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = load_observation_cache(cache_path)
    source_hash = sha256_file(cache_path)
    memory = ObjectMemory(
        metadata={
            "scene_id": cache.scene_id,
            "query": cache.query,
            "source_stage": "D7",
            "source_cache": str(cache_path),
            "source_cache_sha256": source_hash,
        }
    )
    staged_ids = memory.stage_many(cache.observations)
    memory_path = output_dir / "object_memory.json"
    memory.save(memory_path)

    restored = ObjectMemory.load(memory_path)
    round_trip_equal = memory.to_dict() == restored.to_dict()
    if not round_trip_equal:
        raise RuntimeError("ObjectMemory changed after save/reload")
    if len(restored) != 0 or restored.decisions:
        raise RuntimeError("D8 must not perform D9 association")
    if list(restored.pending_observations) != staged_ids:
        raise RuntimeError("pending observation order changed")

    evidence = restored.to_dict()["evidence"]
    result = {
        "schema_version": OBJECT_MEMORY_SCHEMA_VERSION,
        "status": "PASS",
        "stage": "D8",
        "scene_id": cache.scene_id,
        "query": cache.query,
        "version_fields": {
            "object_memory": OBJECT_MEMORY_SCHEMA_VERSION,
            "memory_object": MEMORY_OBJECT_SCHEMA_VERSION,
            "object_observation": OBJECT_OBSERVATION_SCHEMA_VERSION,
        },
        "source": {
            "stage": "D7",
            "cache_path": str(cache_path),
            "cache_sha256": source_hash,
        },
        "pending_observation_count": len(
            restored.pending_observations
        ),
        "permanent_object_count": len(restored),
        "association_decision_count": len(restored.decisions),
        "frame_ids": evidence["frame_ids"],
        "round_trip_equal": round_trip_equal,
        "artifacts": {"object_memory": memory_path.name},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    result_path = output_dir / "d8_result.json"
    write_json(result_path, result)

    RunManifest(
        git_sha=git_commit(project_root),
        env_lock="D8 schema/round-trip is model-free",
        dataset_split=cache.scene_id,
        seed=0,
        config={
            "pipeline": "D8 frozen ObjectMemory envelope",
            "source_cache": str(cache_path),
            "source_cache_sha256": source_hash,
            "object_memory_schema": OBJECT_MEMORY_SCHEMA_VERSION,
            "memory_object_schema": MEMORY_OBJECT_SCHEMA_VERSION,
            "observation_schema": OBJECT_OBSERVATION_SCHEMA_VERSION,
            "association_executed": False,
        },
        command=shlex.join(sys.argv),
        runtime_seconds=time.perf_counter() - started,
        peak_vram_mb=None,
    ).save(output_dir / "run_manifest.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        default="runs/office-loop-d7-trash-can/observations.json",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/office-loop-d8-trash-can",
    )
    parser.add_argument("--project-root", default=str(root))
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
