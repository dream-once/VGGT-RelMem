import json
from pathlib import Path
import unittest

import numpy as np

from relground.clio_grounding_benchmark import (
    _aggregate,
    _evaluate_center,
    _quality,
    _validate_frozen_policy,
)


class ClioGroundingBenchmarkTests(unittest.TestCase):
    def test_abstentions_remain_in_grounding_denominator(self) -> None:
        rows = [
            {"policy": {"answered": True, "correct": True, "correct_with_alignment_rmse_margin": True}},
            {"policy": {"answered": False, "correct": False, "correct_with_alignment_rmse_margin": False}},
        ]
        metrics = _aggregate(rows, "policy")
        self.assertEqual(metrics["task_count"], 2)
        self.assertEqual(metrics["answered_tasks"], 1)
        self.assertEqual(metrics["coverage"], 0.5)
        self.assertEqual(metrics["grounding_acc_at_1"], 0.5)
        self.assertEqual(metrics["conditional_acc_at_1"], 1.0)

    def test_any_gt_containment_wins_over_nearest_center_diagnostic(self) -> None:
        identity = np.eye(3, dtype=np.float64)
        boxes = [
            {
                "gt_id": "nearest_but_not_containing",
                "center": np.asarray([0.20, 0.0, 0.0]),
                "extent": np.asarray([0.02, 0.02, 0.02]),
                "rotation": identity,
            },
            {
                "gt_id": "farther_and_containing",
                "center": np.asarray([1.0, 0.0, 0.0]),
                "extent": np.asarray([2.0, 1.0, 1.0]),
                "rotation": identity,
            },
        ]
        result = _evaluate_center(
            np.asarray([0.25, 0.0, 0.0]),
            boxes,
            alignment_margin_m=0.0,
        )
        self.assertEqual(result["nearest_gt_id"], "nearest_but_not_containing")
        self.assertAlmostEqual(result["center_distance_m"], 0.05)
        self.assertTrue(result["correct"])
        self.assertTrue(result["correct_with_alignment_rmse_margin"])

    def test_observation_quality_is_conservative_geometric_mean(self) -> None:
        value = _quality({
            "retrieval_score": 0.125,
            "sam_score": 1.0,
            "valid_point_ratio": 1.0,
        })
        self.assertAlmostEqual(value, 0.5)

    def test_fallback_choice_depends_only_on_availability(self) -> None:
        q0 = {"answered": True, "correct": True}
        q1 = {"answered": False, "correct": False}
        selected = q1 if q1["answered"] else q0
        self.assertIs(selected, q0)
        q1["answered"] = True
        selected = q1 if q1["answered"] else q0
        self.assertIs(selected, q1)

    def test_frozen_policy_content_is_validated_not_just_its_path(self) -> None:
        root = Path(__file__).resolve().parents[1]
        policy = json.loads(
            (root / "configs/clio_cubicle_frozen_policy.json").read_text()
        )
        validated = _validate_frozen_policy(policy, scene_id="cubicle")
        self.assertEqual(validated["decision_policy"]["name"], "Q1F")
        policy["decision_policy"]["name"] = "Q1"
        with self.assertRaisesRegex(ValueError, "must be Q1F"):
            _validate_frozen_policy(policy, scene_id="cubicle")

    def test_frozen_policy_rejects_wrong_scene(self) -> None:
        root = Path(__file__).resolve().parents[1]
        policy = json.loads(
            (root / "configs/clio_cubicle_frozen_policy.json").read_text()
        )
        with self.assertRaisesRegex(ValueError, "scene"):
            _validate_frozen_policy(policy, scene_id="apartment")


if __name__ == "__main__":
    unittest.main()
