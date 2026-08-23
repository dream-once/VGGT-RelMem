"""Validate D5 PE top-K retrieval artifacts without loading PE or a GPU."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from relground.retrieval import FrameCandidate, RetrievalConfig, TopKFrameRetriever


def _candidate(row: dict[str, Any]) -> FrameCandidate:
    return FrameCandidate(
        frame_id=str(row["frame_id"]),
        score=float(row["retrieval_score"]),
        index=int(row["geometry_index"]),
        camera_center=np.asarray(row["camera_center"], dtype=np.float64),
        view_direction=np.asarray(row["view_direction"], dtype=np.float64),
        metadata={"cosine": float(row["retrieval_cosine"])},
    )


def validate_output(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    errors: list[str] = []
    result_path = root / "retrieval.json"
    manifest_path = root / "run_manifest.json"
    preview_path = root / "topk_preview.png"
    for item in (result_path, manifest_path, preview_path):
        if not item.is_file() or item.stat().st_size == 0:
            errors.append(f"missing or empty artifact: {item.name}")
    if errors:
        return {"status": "FAIL", "errors": errors}

    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "PASS":
            errors.append(f"retrieval status is {result.get('status')!r}")
        if result.get("stage") != "D5":
            errors.append("retrieval stage is not D5")
        raw_rows = result["raw_ranking"]
        if not raw_rows:
            raise ValueError("raw ranking is empty")
        candidates = [_candidate(row) for row in raw_rows]
        expected_raw = TopKFrameRetriever(
            RetrievalConfig(top_k=len(candidates), redundancy="none")
        ).retrieve(candidates)
        raw_ids = [row["frame_id"] for row in raw_rows]
        if raw_ids != [candidate.frame_id for candidate in expected_raw]:
            errors.append("raw ranking is not sorted deterministically")
        if [int(row["rank"]) for row in raw_rows] != list(
            range(1, len(raw_rows) + 1)
        ):
            errors.append("raw ranking has invalid rank values")

        settings = dict(result["retrieval_config"])
        k_values = [int(value) for value in result["k_values"]]
        if k_values != sorted(set(k_values)) or any(value < 1 for value in k_values):
            errors.append("k_values must be sorted, unique and positive")
        if 1 not in k_values:
            errors.append("k_values must include 1")
        selections = result["selections"]
        previous: list[str] = []
        selected_counts: dict[str, int] = {}

        for k in k_values:
            config = RetrievalConfig(top_k=k, **settings)
            expected = TopKFrameRetriever(config).retrieve(candidates)
            expected_ids = [candidate.frame_id for candidate in expected]
            summary = selections[str(k)]
            artifact_path = root / str(summary["artifact"])
            if not artifact_path.is_file() or artifact_path.stat().st_size == 0:
                errors.append(f"missing top-{k} artifact")
                continue
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            rows = artifact.get("frames", [])
            actual_ids = [str(row["frame_id"]) for row in rows]
            if actual_ids != expected_ids:
                errors.append(f"top-{k} frame ids do not match deterministic selection")
            if actual_ids != [str(value) for value in summary["frame_ids"]]:
                errors.append(f"top-{k} summary and artifact disagree")
            if int(artifact.get("requested_k", -1)) != k:
                errors.append(f"top-{k} artifact has wrong requested_k")
            if int(artifact.get("selected_count", -1)) != len(actual_ids):
                errors.append(f"top-{k} artifact has wrong selected_count")
            if int(summary.get("selected_count", -1)) != len(actual_ids):
                errors.append(f"top-{k} summary has wrong selected_count")
            if artifact.get("retrieval_config") != asdict(config):
                errors.append(f"top-{k} retrieval config mismatch")
            exhausted = len(expected) < min(k, len(candidates))
            if bool(artifact.get("exhausted_nonredundant_candidates")) != exhausted:
                errors.append(f"top-{k} exhausted flag is incorrect")
            for rank, (row, candidate) in enumerate(zip(rows, expected), start=1):
                if int(row.get("rank", -1)) != rank:
                    errors.append(f"top-{k} has invalid rank values")
                if not np.isclose(
                    float(row["retrieval_score"]),
                    candidate.score,
                    atol=1e-12,
                ):
                    errors.append(f"top-{k} score mismatch for {candidate.frame_id}")
            if previous != actual_ids[: len(previous)]:
                errors.append(f"top-{k} is not prefix-consistent")
            previous = actual_ids
            selected_counts[str(k)] = len(actual_ids)

        top1 = result["upstream_top1"]
        if not raw_rows or str(top1["frame_id"]) != str(raw_rows[0]["frame_id"]):
            errors.append("upstream top-1 does not match raw ranking")
        if selections.get("1", {}).get("frame_ids", [None])[0] != top1["frame_id"]:
            errors.append("D5 top-1 does not reproduce upstream B0")
        if not bool(result.get("top1_compatible")):
            errors.append("top1_compatible is false")
        if not bool(result.get("prefix_consistent")):
            errors.append("prefix_consistent is false")
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"invalid retrieval artifact: {error}")
        selected_counts = {}

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "query": result.get("query") if "result" in locals() else None,
        "searched_frames": len(raw_rows) if "raw_rows" in locals() else 0,
        "selected_counts": selected_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    args = parser.parse_args()
    result = validate_output(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
