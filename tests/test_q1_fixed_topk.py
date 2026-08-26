import copy
import json
import tempfile
import unittest
from pathlib import Path

from relground.candidate_cache import CandidateOutcomeCache
from relground.d9_association import ManualInstanceLabels
from relground.q1_fixed_topk import (
    FixedTopKPolicy,
    candidate_metadata,
    evaluate_budget_curve,
    replay_fixed_topk,
    validate_evaluation_payload,
    validate_prediction_payload,
)
from relground.q1_synthetic import build_synthetic_complete_fixture
from scripts.validate_d14 import validate_output


CREATED_AT = "2026-08-27T00:00:00+00:00"


def fixture() -> tuple[dict, dict]:
    return build_synthetic_complete_fixture(created_at=CREATED_AT)


def prediction(cache: dict) -> dict:
    return replay_fixed_topk(
        cache,
        cache_ref="synthetic_cache.json",
        cache_sha256="e" * 64,
        created_at=CREATED_AT,
    )


class FixedTopKTests(unittest.TestCase):
    def test_budget_prefixes_are_deterministic_and_k1_matches_q0(self) -> None:
        cache, _ = fixture()
        first = prediction(cache)
        second = prediction(cache)

        self.assertEqual(first, second)
        self.assertEqual(first["status"], "PASS")
        self.assertTrue(first["acceptance"]["k1_matches_q0"])
        selected = [
            [row["frame_id"] for row in curve["selected_frames"]]
            for curve in first["curves"]
        ]
        self.assertEqual(selected[0], [cache["candidate_universe"][0]])
        self.assertEqual(selected[0], selected[1][:1])
        self.assertEqual(selected[1], selected[2][:3])

    def test_selection_does_not_read_unselected_outcome(self) -> None:
        cache, _ = fixture()
        metadata_before = candidate_metadata(cache)
        selected_before = FixedTopKPolicy().select(metadata_before, 3)
        cache["candidates"][-1]["outcome"]["observations"][0][
            "sam_score"
        ] = 0.1
        CandidateOutcomeCache.from_dict(cache)

        self.assertEqual(
            selected_before,
            FixedTopKPolicy().select(candidate_metadata(cache), 3),
        )

    def test_redundancy_exhaustion_is_explicit(self) -> None:
        cache, _ = fixture()
        for index, candidate in enumerate(cache["candidates"]):
            candidate["geometry_index"] = index
            candidate["camera_center"] = [0.0, 0.0, 0.0]
        result = prediction(cache)

        self.assertEqual(result["curves"][-1]["selected_count"], 1)
        self.assertEqual(
            result["curves"][-1]["exhaustion_reason"],
            "nonredundant_candidates_exhausted",
        )

    def test_selected_unmaterialized_outcome_blocks_without_skip(self) -> None:
        cache, _ = fixture()
        candidate = cache["candidates"][1]
        candidate["outcome_status"] = "unmaterialized"
        candidate["failure_reason"] = "gpu_outcome_not_materialized"
        candidate["outcome"] = None
        candidate["cost"] = None
        cache["materialization_status"] = "partial"
        cache["counts"]["available_candidates"] = 5
        cache["counts"]["unmaterialized_candidates"] = 1
        cache["counts"]["total_observations"] = 5
        cache["costs"]["sam_calls"] = 5
        CandidateOutcomeCache.from_dict(cache)
        result = prediction(cache)

        self.assertEqual(result["status"], "BLOCKED_MISSING_OUTCOME")
        blocked = result["curves"][1]
        self.assertEqual(blocked["status"], "BLOCKED_MISSING_OUTCOME")
        self.assertEqual(blocked["selected_frames"][-1]["frame_id"], "frame_0003")
        self.assertEqual(blocked["selected_count"], 2)

    def test_evaluator_is_separate_and_round_trip_validates(self) -> None:
        cache, raw_labels = fixture()
        result = prediction(cache)
        labels = ManualInstanceLabels.from_dict(raw_labels)
        evaluation = evaluate_budget_curve(
            result,
            cache,
            labels,
            prediction_ref="synthetic_prediction.json",
            prediction_sha256="f" * 64,
            labels_ref="synthetic_labels.json",
            labels_sha256="a" * 64,
            created_at=CREATED_AT,
        )

        validate_prediction_payload(result, cache)
        validate_evaluation_payload(evaluation, result, cache, labels)
        self.assertEqual(evaluation["status"], "PASS")
        self.assertNotIn("metrics", json.dumps(result))
        self.assertNotIn("instance_id", json.dumps(result))

    def test_prediction_tamper_is_rejected(self) -> None:
        cache, _ = fixture()
        result = prediction(cache)
        result["curves"][1]["sam_calls"] += 1

        with self.assertRaisesRegex(ValueError, "deterministic replay"):
            validate_prediction_payload(result, cache)

    def test_bundle_validator_rejects_missing_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = validate_output(Path(directory))

        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(len(report["failures"]), 2)


if __name__ == "__main__":
    unittest.main()
