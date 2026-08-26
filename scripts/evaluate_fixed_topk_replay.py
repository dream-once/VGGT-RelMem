"""Evaluate a D14 prediction with labels kept outside policy execution."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from relground.candidate_cache import CandidateOutcomeCache
from relground.d9_association import ManualInstanceLabels
from relground.observation_cache import sha256_file
from relground.q1_fixed_topk import (
    evaluate_budget_curve,
    validate_prediction_payload,
)
from scripts.run_fixed_topk_replay import write_json


def contained_reference(target: Path, output_dir: Path) -> str:
    boundary = output_dir.resolve().parent
    resolved = target.resolve()
    if resolved != boundary and boundary not in resolved.parents:
        raise ValueError("evaluation input must be contained by output parent")
    return Path(os.path.relpath(resolved, output_dir.resolve())).as_posix()


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} root must be an object")
    return payload


def run(args: argparse.Namespace) -> int:
    prediction_path = Path(args.prediction).resolve()
    cache_path = Path(args.cache).resolve()
    labels_path = Path(args.labels).resolve()
    output_dir = Path(args.output_dir).resolve()
    prefix = str(args.prefix).strip()
    if not prefix or "/" in prefix or ".." in prefix:
        raise ValueError("prefix must be a simple file stem")
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction = read_json(prediction_path)
    cache = CandidateOutcomeCache.load(cache_path).to_dict()
    validate_prediction_payload(prediction, cache)
    labels = ManualInstanceLabels.load(labels_path)
    retained_labels_path = output_dir / f"{prefix}_labels.json"
    write_json(retained_labels_path, labels.to_dict())
    result = evaluate_budget_curve(
        prediction,
        cache,
        labels,
        prediction_ref=contained_reference(prediction_path, output_dir),
        prediction_sha256=sha256_file(prediction_path),
        labels_ref=retained_labels_path.name,
        labels_sha256=sha256_file(retained_labels_path),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    write_json(output_dir / f"{prefix}_evaluation.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="real")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
