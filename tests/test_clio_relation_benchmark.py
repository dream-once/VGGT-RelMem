import unittest

from relground.clio_relation_benchmark import OPPOSITE, _pair_gt_matches


class ClioRelationBenchmarkTests(unittest.TestCase):
    def test_directional_opposites_are_involutions(self) -> None:
        for relation, opposite in OPPOSITE.items():
            self.assertEqual(OPPOSITE[opposite], relation)

    def test_protocol_has_all_four_relations(self) -> None:
        self.assertEqual(set(OPPOSITE), {"left_of", "right_of", "front_of", "behind"})

    def test_correct_target_with_wrong_reference_is_not_correct(self) -> None:
        label = {
            "acceptable_target_object_ids_strict": ["target-good"],
            "acceptable_target_object_ids_alignment_rmse_padded": ["target-good"],
            "acceptable_reference_object_ids_strict": ["reference-good"],
            "acceptable_reference_object_ids_alignment_rmse_padded": ["reference-good"],
        }
        result = _pair_gt_matches(
            label,
            predicted_target_id="target-good",
            predicted_reference_id="reference-wrong",
        )
        self.assertTrue(result["target_strict"])
        self.assertFalse(result["reference_strict"])
        self.assertFalse(result["pair_strict"])
        self.assertFalse(result["pair_padded"])

    def test_relation_pair_requires_both_semantic_roles(self) -> None:
        label = {
            "acceptable_target_object_ids_strict": ["target-good"],
            "acceptable_target_object_ids_alignment_rmse_padded": ["target-good"],
            "acceptable_reference_object_ids_strict": ["reference-good"],
            "acceptable_reference_object_ids_alignment_rmse_padded": ["reference-good"],
        }
        result = _pair_gt_matches(
            label,
            predicted_target_id="target-good",
            predicted_reference_id="reference-good",
        )
        self.assertTrue(result["pair_strict"])
        self.assertTrue(result["pair_padded"])


if __name__ == "__main__":
    unittest.main()
