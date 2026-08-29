import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
import numpy as np


from relground.candidate_cache import CandidateOutcomeCache
from relground.q1_fixed_topk import replay_fixed_topk
from relground.q1_synthetic import build_synthetic_complete_fixture
from relground.q2_sequential import (
    GainBasedSequentialPolicy,
    Q2_METHOD_NAME,
    Q2_OBSERVATION_METRIC,
    Q2_OBSERVATION_SEMANTICS,
    SequentialSearchConfig,
    build_engineering_comparison,
    run_sequential_search,
    sequential_metadata,
    validate_comparison_payload,
    validate_trace_payload,
)
from scripts.validate_d15 import validate_output


CREATED_AT = "2026-08-27T00:00:00+00:00"


def fixture() -> dict:
    cache, _ = build_synthetic_complete_fixture(created_at=CREATED_AT)
    return cache


def trace(cache: dict, config: SequentialSearchConfig | None = None) -> dict:
    return run_sequential_search(
        cache,
        cache_ref="synthetic_cache.json",
        cache_sha256="a" * 64,
        created_at=CREATED_AT,
        config=config,
    )


def set_unmaterialized(cache: dict, index: int) -> None:
    candidate = cache["candidates"][index]
    observations = candidate["outcome"]["lifted_instances"]
    rejections = candidate["outcome"]["rejected_instances"]
    candidate["outcome_status"] = "unmaterialized"
    candidate["failure_reason"] = "gpu_outcome_not_materialized"
    candidate["outcome"] = None
    candidate["cost"] = None
    cache["materialization_status"] = "partial"
    cache["counts"]["available_candidates"] -= 1
    cache["counts"]["unmaterialized_candidates"] += 1
    cache["counts"]["total_observations"] -= observations
    cache["counts"]["total_rejections"] -= rejections
    cache["costs"]["sam_calls"] -= 1


class SequentialSearchTests(unittest.TestCase):
    def test_budget_one_strictly_matches_q0(self) -> None:
        cache = fixture()
        result = trace(cache, SequentialSearchConfig(max_budget=1))

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(
            result["summary"]["selected_frames"],
            [cache["candidate_universe"][0]],
        )
        self.assertTrue(result["summary"]["budget1_matches_q0"])
        self.assertEqual(
            result["steps"][0]["selected"]["selection_reason"],
            "retrieval_score_only",
        )

    def test_pose_novelty_and_weighted_score_are_frozen(self) -> None:
        metadata = [
            {
                "rank": 1,
                "frame_id": "a",
                "geometry_index": 0,
                "camera_center": [0.0, 0.0, 0.0],
                "view_direction": [0.0, 0.0, 1.0],
                "retrieval_score": 1.0,
                "retrieval_cosine": 1.0,
            },
            {
                "rank": 2,
                "frame_id": "b",
                "geometry_index": 1,
                "camera_center": [0.075, 0.0, 0.0],
                "view_direction": [0.02617695, 0.0, 0.99965732],
                "retrieval_score": 0.5,
                "retrieval_cosine": 0.5,
            },
            {
                "rank": 3,
                "frame_id": "c",
                "geometry_index": 2,
                "camera_center": [0.0, 0.0, 0.0],
                "view_direction": [0.0, 0.0, 1.0],
                "retrieval_score": 0.0,
                "retrieval_cosine": 0.0,
            },
        ]
        scores = GainBasedSequentialPolicy().score_candidates(metadata, ["a"])
        row = next(item for item in scores if item["frame_id"] == "b")

        self.assertAlmostEqual(row["translation_novelty"], 0.5)
        self.assertAlmostEqual(row["view_angle_novelty"], 0.5, places=5)
        self.assertAlmostEqual(row["pose_novelty"], 0.5, places=5)
        self.assertAlmostEqual(row["policy_score"], 0.5, places=5)

    def test_method_name_does_not_claim_object_coverage(self) -> None:
        self.assertEqual(
            Q2_METHOD_NAME, "retrieval-pose-novelty-sequential-search"
        )
        self.assertEqual(Q2_OBSERVATION_METRIC, "new_observation_count")
        self.assertIn(
            "not_object_or_spatial_coverage", Q2_OBSERVATION_SEMANTICS
        )

    def test_ties_break_by_source_rank_then_frame_id(self) -> None:
        cache = fixture()
        for candidate in cache["candidates"]:
            candidate["retrieval_score"] = 0.5
            candidate["retrieval_cosine"] = 0.5
            candidate["camera_center"] = [0.0, 0.0, 0.0]
            candidate["view_direction"] = [0.0, 0.0, 1.0]
        CandidateOutcomeCache.from_dict(cache)
        result = trace(cache)

        self.assertEqual(
            result["summary"]["selected_frames"],
            cache["candidate_universe"][:5],
        )

    def test_two_zero_gain_steps_stop_search(self) -> None:
        cache = fixture()
        for candidate in cache["candidates"]:
            candidate["outcome"]["observations"] = []
            candidate["outcome"]["sam_instances"] = 1
            candidate["outcome"]["lifted_instances"] = 0
            candidate["outcome"]["rejected_instances"] = 1
            candidate["outcome"]["rejections"] = [{"reason": "synthetic"}]
        cache["counts"]["total_observations"] = 0
        cache["counts"]["total_rejections"] = 6
        CandidateOutcomeCache.from_dict(cache)
        result = trace(cache)

        self.assertEqual(result["summary"]["selected_count"], 2)
        self.assertEqual(result["summary"]["stop_reason"], "two_consecutive_low_gain")

    def test_candidate_exhaustion_is_explicit(self) -> None:
        cache = fixture()
        cache["candidates"] = cache["candidates"][:2]
        cache["candidate_universe"] = cache["candidate_universe"][:2]
        cache["counts"]["candidate_count"] = 2
        cache["counts"]["available_candidates"] = 2
        cache["counts"]["total_observations"] = 2
        cache["costs"]["sam_calls"] = 2
        CandidateOutcomeCache.from_dict(cache)
        result = trace(cache)

        self.assertEqual(result["summary"]["selected_count"], 2)
        self.assertEqual(result["summary"]["stop_reason"], "candidate_exhausted")

    def test_missing_selected_outcome_blocks_without_skip(self) -> None:
        cache = fixture()
        set_unmaterialized(cache, 1)
        CandidateOutcomeCache.from_dict(cache)
        result = trace(cache)

        self.assertEqual(result["status"], "BLOCKED_MISSING_OUTCOME")
        self.assertEqual(result["summary"]["selected_frames"][-1], "frame_0003")
        self.assertEqual(result["summary"]["selected_count"], 2)
        self.assertIsNone(result["steps"][-1]["revealed_outcome"]["observed_gain"])

    def test_future_outcome_does_not_change_selection_prefix(self) -> None:
        cache = fixture()
        before = trace(cache)
        mutated = copy.deepcopy(cache)
        mutated["candidates"][-1]["outcome"]["observations"][0][
            "sam_score"
        ] = 0.1
        CandidateOutcomeCache.from_dict(mutated)
        after = trace(mutated)

        self.assertEqual(
            before["summary"]["selected_frames"],
            after["summary"]["selected_frames"],
        )

    def test_trace_and_comparison_recompute_deterministically(self) -> None:
        cache = fixture()
        q2 = trace(cache)
        q1 = replay_fixed_topk(
            cache,
            cache_ref="synthetic_cache.json",
            cache_sha256="a" * 64,
            created_at=CREATED_AT,
        )
        comparison = build_engineering_comparison(
            q1,
            q2,
            q1_ref="q1.json",
            q1_sha256="b" * 64,
            q2_ref="q2.json",
            q2_sha256="c" * 64,
            created_at=CREATED_AT,
        )

        validate_trace_payload(q2, cache)
        validate_comparison_payload(comparison, q1, q2)
        self.assertEqual(comparison["comparison_scope"], "synthetic_engineering_counts_not_performance")
        self.assertIsNone(q2["summary"]["performance_claim"])
        self.assertNotIn("instance_id", json.dumps(q2))

    def test_one_ulp_replay_drift_passes_but_tamper_fails(self) -> None:
        cache = fixture()
        result = trace(cache)
        score = result["steps"][1]["selected"]["policy_score"]
        result["steps"][1]["selected"]["policy_score"] = float(
            np.nextafter(score, math.inf)
        )
        validate_trace_payload(result, cache)

        result["steps"][1]["selected"]["policy_score"] += 1e-6
        with self.assertRaisesRegex(ValueError, "float differs"):
            validate_trace_payload(result, cache)

    def test_nonfinite_replay_float_fails_closed(self) -> None:
        cache = fixture()
        result = trace(cache)
        result["steps"][1]["selected"]["policy_score"] = math.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_trace_payload(result, cache)

    def test_trace_tamper_and_missing_bundle_fail(self) -> None:
        cache = fixture()
        result = trace(cache)
        result["steps"][1]["selected"]["policy_score"] += 0.01
        with self.assertRaisesRegex(ValueError, "deterministic replay"):
            validate_trace_payload(result, cache)
        with tempfile.TemporaryDirectory() as directory:
            report = validate_output(Path(directory))
        self.assertEqual(report["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
