"""Replay D15 gain-based sequential search from a candidate cache."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
import time

from relground.candidate_cache import CandidateOutcomeCache
from relground.observation_cache import sha256_file
from relground.q2_sequential import run_sequential_search
from relground.schemas import RunManifest
from scripts.run_fixed_topk_replay import (
    git_commit,
    relative_source_reference,
    write_json,
)


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
    result = run_sequential_search(
        cache,
        cache_ref=relative_source_reference(cache_path, output_dir),
        cache_sha256=sha256_file(cache_path),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    write_json(output_dir / f"{prefix}_trace.json", result)
    RunManifest(
        git_sha=git_commit(project_root),
        env_lock="D15 deterministic CPU cache replay; no model inference",
        dataset_split=str(cache["scene_id"]),
        seed=0,
        config={
            "stage": "D15-policy-trace",
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
    if result["status"] == "PASS" or args.allow_blocked:
        return 0
    return 3


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="real")
    parser.add_argument("--allow-blocked", action="store_true")
    parser.add_argument("--project-root", default=str(root))
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
