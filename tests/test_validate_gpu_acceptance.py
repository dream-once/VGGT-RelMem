import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from relground.candidate_cache import CandidateOutcomeCache
from scripts.validate_gpu_acceptance import (
    _forbidden_paths,
    _positive_vram,
    compare_retained_partial,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETAINED_ROOT = PROJECT_ROOT / "evidence/week2/d11-candidate-cache"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _complete_fixture(root: Path) -> Path:
    old = _read(RETAINED_ROOT / "candidate_cache.json")
    complete = copy.deepcopy(old)
    complete["materialization_status"] = "complete"
    for candidate in complete["candidates"]:
        rank = int(candidate["rank"])
        if candidate["outcome_status"] == "available":
            if candidate["frame_id"] == "frame_0021":
                for collection in ("observations", "rejections"):
                    for row in candidate["outcome"][collection]:
                        if "selected_rank" in row:
                            row["selected_rank"] = rank
                        metadata = row.get("metadata", {})
                        if "selected_rank" in metadata:
                            metadata["selected_rank"] = rank
            continue
        candidate["outcome_status"] = "available"
        candidate["failure_reason"] = None
        candidate["outcome"] = {
            "sam_instances": 0,
            "lifted_instances": 0,
            "rejected_instances": 0,
            "observations": [],
            "rejections": [],
        }
        candidate["cost"] = {
            "sam_calls": 1,
            "runtime_seconds": None,
            "peak_vram_mb": None,
        }
    complete["counts"].update(
        available_candidates=8,
        unmaterialized_candidates=0,
    )
    complete["costs"]["sam_calls"] = 8
    CandidateOutcomeCache.from_dict(complete)

    cache_path = root / "candidate_cache.json"
    _write(cache_path, complete)
    _write(
        root / "source_d5_retrieval.json",
        _read(RETAINED_ROOT / "source_d5_retrieval.json"),
    )
    _write(
        root / "source_d6_result.json",
        {
            "selected_frames": [
                {"frame_id": row["frame_id"], "rank": row["rank"]}
                for row in complete["candidates"]
            ]
        },
    )
    return cache_path


class GPUAcceptanceValidatorTests(unittest.TestCase):
    def test_partial_completion_preserves_old_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            complete_path = _complete_fixture(Path(directory))
            report = compare_retained_partial(
                RETAINED_ROOT / "candidate_cache.json",
                complete_path,
            )

        self.assertEqual(len(report["retained_available_unchanged"]), 4)
        self.assertEqual(len(report["newly_materialized"]), 4)
        self.assertEqual(
            report["rank_provenance_changes"],
            [
                {
                    "frame_id": "frame_0021",
                    "retained_selection_rank": 4,
                    "complete_source_rank": 8,
                }
            ],
        )
        self.assertEqual(
            report["normalized_fields"],
            [
                "observations[].metadata.selected_rank",
                "rejections[].selected_rank",
            ],
        )

    def test_non_rank_outcome_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete_path = _complete_fixture(root)
            payload = _read(complete_path)
            payload["candidates"][0]["outcome"]["observations"][0][
                "sam_score"
            ] -= 0.01
            CandidateOutcomeCache.from_dict(payload)
            _write(complete_path, payload)

            with self.assertRaisesRegex(ValueError, "outcome drifted"):
                compare_retained_partial(
                    RETAINED_ROOT / "candidate_cache.json",
                    complete_path,
                )

    def test_gpu_provenance_and_evaluation_leak_guards(self) -> None:
        self.assertEqual(_positive_vram({"peak_vram_mb": 12.5}, "D6"), 12.5)
        for value in (None, 0, -1, math.inf, math.nan):
            with self.assertRaisesRegex(ValueError, "peak_vram_mb"):
                _positive_vram({"peak_vram_mb": value}, "D6")
        self.assertEqual(
            _forbidden_paths({"prediction": {"metrics": {"f1": 1.0}}}),
            ["prediction.metrics", "prediction.metrics.f1"],
        )


if __name__ == "__main__":
    unittest.main()
