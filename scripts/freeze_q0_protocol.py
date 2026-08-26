"""Freeze the D13 Q0 upstream-aligned protocol without model inference."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
import time

from relground.q0_protocol import build_q0_protocol
from relground.schemas import RunManifest
from scripts.run_d9_association import git_commit, prepare_output_dir, write_json


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    prepare_output_dir(output_dir)
    protocol = build_q0_protocol(
        project_root,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    write_json(output_dir / "q0_protocol.json", protocol)
    RunManifest(
        git_sha=git_commit(project_root),
        env_lock="D13 is a deterministic static source/evidence audit",
        dataset_split="office-loop-development",
        seed=0,
        config={
            "stage": "D13",
            "status": "CPU_COMPLETE",
            "gpu_acceptance": "PENDING",
            "protocol_id": protocol["protocol_id"],
            "protocol_status": protocol["status"],
            "source_file_count": len(protocol["source_files"]),
            "source_check_count": len(protocol["source_checks"]),
            "limitation_count": len(protocol["limitations"]),
        },
        command=shlex.join(sys.argv),
        runtime_seconds=time.perf_counter() - started,
        peak_vram_mb=None,
    ).save(output_dir / "run_manifest.json")
    print(json.dumps({
        "status": "PASS",
        "stage": "D13",
        "cpu_completion": "COMPLETE",
        "gpu_acceptance": "PENDING",
        "protocol_id": protocol["protocol_id"],
        "protocol_status": protocol["status"],
        "top1_matches_d5_first": protocol["development_selection"][
            "top1_matches_d5_first"
        ],
        "source_checks": protocol["source_checks"],
        "limitations": protocol["limitations"],
    }, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="evidence/week2/d13-q0-protocol"
    )
    parser.add_argument("--project-root", default=str(root))
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
