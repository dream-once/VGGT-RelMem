import unittest

from relground.clio_association_benchmark import _accumulate, _metrics, label_pair


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


if __name__ == "__main__":
    unittest.main()
