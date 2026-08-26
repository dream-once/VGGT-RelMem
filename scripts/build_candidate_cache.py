"""Build a CPU-only D11 candidate cache from retained D5/D6/D7 JSON."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

from relground.candidate_cache import (
    CandidateOutcomeCache,
    VisualMemoryManifest,
    build_d11_payloads,
)
from relground.observation_cache import sha256_file
from relground.schemas import RunManifest


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} root must be an object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _git_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _prepare_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _project_image_reference(project_root: Path, image_path: str) -> tuple[str, str]:
    resolved = Path(image_path).resolve()
    if resolved != project_root and project_root not in resolved.parents:
        raise ValueError(f"image path escapes project root: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"missing ranked image: {resolved}")
    return (
        Path(os.path.relpath(resolved, project_root)).as_posix(),
        sha256_file(resolved),
    )


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    _prepare_output(output_dir)

    retrieval_path = Path(args.retrieval).resolve()
    d6_path = Path(args.d6_result).resolve()
    observations_path = Path(args.observations).resolve()
    retrieval = _read_json(retrieval_path)
    d6_result = _read_json(d6_path)
    observations = _read_json(observations_path)

    snapshots = {
        "d5_retrieval": output_dir / "source_d5_retrieval.json",
        "d6_result": output_dir / "source_d6_result.json",
        "d7_observations": output_dir / "source_d7_observations.json",
    }
    for key, payload in (
        ("d5_retrieval", retrieval),
        ("d6_result", d6_result),
        ("d7_observations", observations),
    ):
        _write_json(snapshots[key], payload)

    image_refs: dict[str, str] = {}
    image_hashes: dict[str, str] = {}
    for row in retrieval["raw_ranking"]:
        frame_id = str(row["frame_id"])
        image_ref, image_hash = _project_image_reference(
            project_root, str(row["image_path"])
        )
        image_refs[frame_id] = image_ref
        image_hashes[frame_id] = image_hash

    visual_path = output_dir / "visual_memory_manifest.json"
    cache_path = output_dir / "candidate_cache.json"
    created_at = datetime.now(timezone.utc).isoformat()
    artifact_refs = {
        "visual_memory_manifest": visual_path.name,
        "d5_retrieval": snapshots["d5_retrieval"].name,
        "d6_result": snapshots["d6_result"].name,
        "d7_observations": snapshots["d7_observations"].name,
    }
    source_hashes = {
        "d5_retrieval_sha256": sha256_file(snapshots["d5_retrieval"]),
        "d6_result_sha256": sha256_file(snapshots["d6_result"]),
        "d7_observations_sha256": sha256_file(
            snapshots["d7_observations"]
        ),
    }
    visual_payload, cache_payload = build_d11_payloads(
        retrieval=retrieval,
        d6_result=d6_result,
        observations_payload=observations,
        scene_id=args.scene_id,
        query_id=args.query_id,
        image_refs=image_refs,
        image_hashes=image_hashes,
        source_hashes=source_hashes,
        artifact_refs=artifact_refs,
        created_at=created_at,
    )
    VisualMemoryManifest.from_dict(visual_payload).save(visual_path)
    CandidateOutcomeCache.from_dict(cache_payload).save(cache_path)

    materialization = cache_payload["materialization_status"]
    status = (
        "PASS"
        if materialization == "complete"
        else "PASS_WITH_UNMATERIALIZED_OUTCOMES"
    )
    RunManifest(
        git_sha=_git_commit(project_root),
        env_lock="D11 is deterministic and model-free",
        dataset_split=args.scene_id,
        seed=0,
        config={
            "stage": "D11",
            "status": "CPU_COMPLETE",
            "gpu_acceptance": "PENDING",
            "candidate_cache": cache_path.name,
            "materialization_status": materialization,
            "source_hashes": source_hashes,
        },
        command=shlex.join(
            [sys.executable, "-m", "scripts.build_candidate_cache", *sys.argv[1:]]
        ),
        runtime_seconds=time.perf_counter() - started,
        peak_vram_mb=None,
    ).save(output_dir / "run_manifest.json")
    report = {
        "status": status,
        "stage": "D11",
        "cpu_completion": "COMPLETE",
        "gpu_acceptance": "PENDING",
        "scene_id": cache_payload["scene_id"],
        "query_id": cache_payload["query_id"],
        **cache_payload["counts"],
        "embedding_status": visual_payload["embedding_status"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrieval",
        default="evidence/week1/d5-multiview/trash-can/retrieval.json",
    )
    parser.add_argument(
        "--d6-result",
        default="evidence/week1/d6-multiview/trash-can/d6_result.json",
    )
    parser.add_argument(
        "--observations",
        default=(
            "evidence/week1/runs/office-loop-mv-d7-trash-can/"
            "observations.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="evidence/week2/d11-candidate-cache",
    )
    parser.add_argument(
        "--scene-id",
        default="office-loop-multiview-s10-trash-can",
    )
    parser.add_argument("--query-id", default="office-loop-trash-can")
    parser.add_argument("--project-root", default=str(root))
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
