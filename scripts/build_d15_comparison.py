"""Build the D15 synthetic Q0/Q1/Q2 engineering-count comparison."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from relground.observation_cache import sha256_file
from relground.q2_sequential import build_engineering_comparison
from scripts.run_fixed_topk_replay import write_json


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} root must be an object")
    return payload


def contained_reference(target: Path, output_dir: Path) -> str:
    boundary = output_dir.resolve().parent
    resolved = target.resolve()
    if resolved != boundary and boundary not in resolved.parents:
        raise ValueError("comparison input must be contained by output parent")
    return Path(os.path.relpath(resolved, output_dir.resolve())).as_posix()


def run(args: argparse.Namespace) -> int:
    q1_path = Path(args.q1_prediction).resolve()
    q2_path = Path(args.q2_trace).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = build_engineering_comparison(
        read_json(q1_path),
        read_json(q2_path),
        q1_ref=contained_reference(q1_path, output_dir),
        q1_sha256=sha256_file(q1_path),
        q2_ref=contained_reference(q2_path, output_dir),
        q2_sha256=sha256_file(q2_path),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    write_json(output_dir / "synthetic_comparison.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q1-prediction", required=True)
    parser.add_argument("--q2-trace", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
