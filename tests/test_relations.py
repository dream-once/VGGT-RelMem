import unittest

import numpy as np

from relground.association import ObjectMemory
from relground.relations import RelationGrounder
from relground.schemas import GroundingQuery, ObjectObservation, OrientedBoundingBox


def observation(obs_id: str, class_text: str, center: list[float]) -> ObjectObservation:
    center_array = np.asarray(center, dtype=float)
    return ObjectObservation(
        obs_id=obs_id,
        class_text=class_text,
        frame_id=obs_id,
        mask_ref=None,
        retrieval_score=0.95,
        sam_score=0.95,
        valid_point_ratio=0.95,
        points_ref=None,
        center=center_array,
        obb=OrientedBoundingBox(center_array, np.array([0.2, 0.2, 0.2])),
    )


def query(target: str = "chair") -> GroundingQuery:
    return GroundingQuery("q", target, "left_of", "desk", "anchor")


class RelationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = ObjectMemory()
        self.decisions = self.memory.add_many(
            [
                observation("left", "chair", [-1, 0, 0]),
                observation("right", "chair", [1, 0, 0]),
                observation("desk", "desk", [0, 0, 0]),
            ]
        )

    def test_left_right_changes_with_anchor_rotation(self) -> None:
        identity_result = RelationGrounder(self.memory, {"anchor": np.eye(4)}).ground(query())
        self.assertFalse(identity_result.abstain)
        self.assertEqual(identity_result.ranked_ids[0], self.decisions[0].object_id)

        rotated = np.eye(4)
        rotated[:3, :3] = np.diag([-1.0, 1.0, -1.0])
        rotated_result = RelationGrounder(self.memory, {"anchor": rotated}).ground(query())
        self.assertFalse(rotated_result.abstain)
        self.assertEqual(rotated_result.ranked_ids[0], self.decisions[1].object_id)

    def test_missing_target_and_anchor_are_explicit_abstentions(self) -> None:
        missing_target = RelationGrounder(self.memory, {"anchor": np.eye(4)}).ground(query("lamp"))
        self.assertTrue(missing_target.abstain)
        self.assertEqual(missing_target.reason, "target_not_found")

        missing_anchor = RelationGrounder(self.memory).ground(query())
        self.assertTrue(missing_anchor.abstain)
        self.assertEqual(missing_anchor.reason, "anchor_pose_not_found")

    def test_relation_boundary_abstains(self) -> None:
        memory = ObjectMemory()
        memory.add_many(
            [observation("chair", "chair", [0, 0, 0]), observation("desk", "desk", [0, 0, 0])]
        )
        result = RelationGrounder(memory, {"anchor": np.eye(4)}).ground(query())
        self.assertTrue(result.abstain)
        self.assertEqual(result.reason, "relation_conflict_or_boundary")


if __name__ == "__main__":
    unittest.main()
