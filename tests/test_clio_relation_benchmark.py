import unittest

from relground.clio_relation_benchmark import OPPOSITE


class ClioRelationBenchmarkTests(unittest.TestCase):
    def test_directional_opposites_are_involutions(self) -> None:
        for relation, opposite in OPPOSITE.items():
            self.assertEqual(OPPOSITE[opposite], relation)

    def test_protocol_has_all_four_relations(self) -> None:
        self.assertEqual(set(OPPOSITE), {"left_of", "right_of", "front_of", "behind"})


if __name__ == "__main__":
    unittest.main()
