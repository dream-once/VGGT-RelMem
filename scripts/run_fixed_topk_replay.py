"""Replay the D14 fixed-Top-K policy from a CandidateOutcomeCache."""

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

from relground.candidate_cache import CandidateOutcomeCache
from relground.observation_cache import sha256_file
from relground.q1_fixed_topk import replay_fixed_topk
from relground.schemas import RunManifest


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def git_commit(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def relative_source_reference(source: Path, output_dir: Path) -> str:
    boundary = output_dir.resolve().parent
    resolved = source.resolve()
    if resolved != boundary and boundary not in resolved.parents:
        raise ValueError("cache must be contained by the output parent")
    return Path(os.path.relpath(resolved, output_dir.resolve())).as_posix()


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    project_root = Path(args.project_root).resolve()
    cache_path = Path(args.cache).resolve()
    output_dir = Path(args.output_dir).resolve()
    prefix = str(args.prefix).strip()
    if not prefix or "/" in prefix or ".." in prefix:
        raise ValueError("prefix must be a simple file stem")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = CandidateOutcomeCache.load(cache_path).to_dict()
    created_at = datetime.now(timezone.utc).isoformat()
    result = replay_fixed_topk(
        cache,
        cache_ref=relative_source_reference(cache_path, output_dir),
        cache_sha256=sha256_file(cache_path),
        created_at=created_at,
    )
    result_path = output_dir / f"{prefix}_prediction.json"
    write_json(result_path, result)
    RunManifest(
        git_sha=git_commit(project_root),
        env_lock="D14 deterministic CPU cache replay; no model inference",
        dataset_split=str(cache["scene_id"]),
        seed=0,
        config={
            "stage": "D14-prediction",
            "policy_id": result["policy_id"],
            "candidate_cache": result["source"]["candidate_cache"],
            "candidate_cache_sha256": result["source"][
                "candidate_cache_sha256"
            ],
            "development_replay": True,
            "gpu_acceptance": "GPU_ACCEPTANCE_PENDING",
        },
        command=shlex.join(sys.argv),
        runtime_seconds=time.perf_counter() - started,
        peak_vram_mb=None,
    ).save(output_dir / f"{prefix}_run_manifest.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 3


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="real")
    parser.add_argument("--project-root", default=str(root))
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
