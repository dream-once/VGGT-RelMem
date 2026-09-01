from types import SimpleNamespace
import unittest

from relground.clio_association_benchmark import (
    _a1_component_prediction_map,
    _accumulate,
    _metrics,
    label_pair,
)


class ClioAssociationBenchmarkTests(unittest.TestCase):
    def test_background_background_pair_is_unknown(self) -> None:
        self.assertIsNone(label_pair({"assigned_gt_id": None}, {"assigned_gt_id": None}))

    def test_target_background_pair_is_negative(self) -> None:
        self.assertFalse(label_pair({"assigned_gt_id": "gt_0000"}, {"assigned_gt_id": None}))

    def test_same_gt_pair_is_positive(self) -> None:
        self.assertTrue(label_pair({"assigned_gt_id": "gt_0000"}, {"assigned_gt_id": "gt_0000"}))

    def test_pair_metrics(self) -> None:
        counts = {
            "pair_count": 0, "positive_pairs": 0, "negative_pairs": 0,
            "true_positive": 0, "false_positive": 0,
            "true_negative": 0, "false_negative": 0,
        }
        _accumulate(counts, True, True)
        _accumulate(counts, False, True)
        metrics = _metrics(counts)
        self.assertEqual(metrics["pair_count"], 2)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 1.0)

    def test_a1_metrics_use_production_component_closure(self) -> None:
        observations = [SimpleNamespace(obs_id=value) for value in ("a", "b", "c")]
        pairs = [
            SimpleNamespace(obs_id_a="a", obs_id_b="b", predicted_same=True),
            SimpleNamespace(obs_id_a="a", obs_id_b="c", predicted_same=False),
            SimpleNamespace(obs_id_a="b", obs_id_b="c", predicted_same=True),
        ]
        self.assertEqual(
            _a1_component_prediction_map(observations, pairs),
            {
                ("a", "b"): True,
                ("a", "c"): True,
                ("b", "c"): True,
            },
        )


if __name__ == "__main__":
    unittest.main()
