"""Write the complete D14 synthetic cache and evaluator-only labels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from relground.q1_synthetic import build_synthetic_complete_fixture
from scripts.run_fixed_topk_replay import write_json


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache, labels = build_synthetic_complete_fixture(
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    write_json(output_dir / "synthetic_cache.json", cache)
    write_json(output_dir / "synthetic_source_labels.json", labels)
    print(json.dumps({
        "status": "PASS",
        "materialization_status": cache["materialization_status"],
        "candidate_count": cache["counts"]["candidate_count"],
        "labels_are_evaluator_only": True,
    }, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
