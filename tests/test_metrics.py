import unittest

from evaluation.metrics import (
    area_under_risk_coverage,
    expected_calibration_error,
    brier_score,
    frame_recall_at_k,
    grounding_metrics,
    pairwise_f1,
    risk_coverage_curve,
    selective_answer_risk_coverage,
)


class MetricTests(unittest.TestCase):
    def test_core_metrics(self) -> None:
        self.assertEqual(frame_recall_at_k({"q": {"f2"}}, {"q": ["f1", "f2"]}, 2), 1.0)
        self.assertEqual(pairwise_f1([1, 1, 2], ["a", "a", "b"])["f1"], 1.0)
        self.assertEqual(grounding_metrics(["a"], [["a", "b"]])["acc_at_1"], 1.0)
        negative = grounding_metrics([None], [[]], [True])
        self.assertEqual(negative["negative_rejection_accuracy"], 1.0)
        self.assertEqual(negative["task_accuracy"], 1.0)
        self.assertAlmostEqual(brier_score([0.8, 0.2], [1, 0]), 0.04)
        curve = risk_coverage_curve([0.9, 0.1], [True, False])
        self.assertEqual(curve[0]["risk"], 0.0)
        self.assertEqual(curve[-1]["coverage"], 1.0)
        self.assertAlmostEqual(expected_calibration_error([0.8, 0.2], [1, 0]), 0.2)
        self.assertAlmostEqual(area_under_risk_coverage(curve), 0.25)
        selective = selective_answer_risk_coverage(
            [0.9, 0.1, 1.0],
            [True, True, False],
            [True, False, True],
        )
        self.assertEqual(len(selective), 2)
        self.assertAlmostEqual(selective[-1]["coverage"], 2.0 / 3.0)
        self.assertEqual(selective[0]["answered_count"], 1)
        self.assertAlmostEqual(
            area_under_risk_coverage(selective), 0.25
        )

    def test_selective_answer_curve_rejects_bad_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite probabilities"):
            selective_answer_risk_coverage(
                [float("nan")], [True], [True]
            )



if __name__ == "__main__":
    unittest.main()
