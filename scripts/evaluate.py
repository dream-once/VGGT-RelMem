"""Evaluate structured JSONL queries against a frozen ObjectMemory."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np

from evaluation.metrics import brier_score, grounding_metrics, risk_coverage_curve
from relground.association import ObjectMemory
from relground.relations import RelationGrounder
from relground.schemas import GroundingQuery


def _load_anchor_poses(path: str | None) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text())
    return {frame_id: np.asarray(pose, dtype=np.float64) for frame_id, pose in payload.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", required=True)
    parser.add_argument("--queries", required=True, help="JSONL with query fields and answer_object_id")
    parser.add_argument("--anchor-poses", help="JSON mapping frame_id to world_from_anchor")
    parser.add_argument("--output", help="Optional detailed JSON result")
    args = parser.parse_args()

    memory = ObjectMemory.load(args.memory)
    grounder = RelationGrounder(memory, _load_anchor_poses(args.anchor_poses))
    records = [json.loads(line) for line in Path(args.queries).read_text().splitlines() if line.strip()]
    results = [grounder.ground(GroundingQuery.from_dict(record)) for record in records]
    answers = [record.get("answer_object_id") for record in records]
    rankings = [result.ranked_ids for result in results]
    correct = [answer is not None and bool(ranking) and ranking[0] == answer for answer, ranking in zip(answers, rankings)]
    metrics = grounding_metrics(answers, rankings, [result.abstain for result in results])
    if results:
        metrics["brier"] = brier_score([item.confidence for item in results], correct)
        risk_coverage = risk_coverage_curve([item.confidence for item in results], correct)
    else:
        metrics["brier"] = None
        risk_coverage = []
    payload = {
        "metrics": metrics,
        "risk_coverage": risk_coverage,
        "results": [item.to_dict() for item in results],
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
