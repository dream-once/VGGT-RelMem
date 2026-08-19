import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from relground.association import ObjectMemory
from relground.schemas import ObjectObservation, OrientedBoundingBox


def make_observation(obs_id: str, center: list[float], class_text: str = "chair") -> ObjectObservation:
    center_array = np.asarray(center, dtype=float)
    return ObjectObservation(
        obs_id=obs_id,
        class_text=class_text,
        frame_id=f"frame_{obs_id}",
        mask_ref=f"masks/{obs_id}.npy",
        retrieval_score=0.9,
        sam_score=0.9,
        valid_point_ratio=0.9,
        points_ref=f"points/{obs_id}.npz",
        center=center_array,
        obb=OrientedBoundingBox(center_array, np.array([0.4, 0.8, 0.4])),
        semantic_embedding=np.array([1.0, 0.0]),
    )


class AssociationTests(unittest.TestCase):
    def test_same_object_merges_and_adjacent_instance_stays_separate(self) -> None:
        memory = ObjectMemory()
        first, second, other = memory.add_many(
            [
                make_observation("a", [0.0, 0.0, 0.0]),
                make_observation("b", [0.1, 0.0, 0.0]),
                make_observation("c", [1.2, 0.0, 0.0]),
            ]
        )
        self.assertTrue(first.created)
        self.assertFalse(second.created)
        self.assertEqual(first.object_id, second.object_id)
        self.assertNotEqual(first.object_id, other.object_id)
        self.assertEqual(len(memory), 2)
        self.assertEqual(len(memory.get(first.object_id).observations), 2)

    def test_memory_json_round_trip_preserves_ids_and_values(self) -> None:
        memory = ObjectMemory()
        memory.add_many([make_observation("a", [0, 0, 0]), make_observation("b", [0.1, 0, 0])])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            memory.save(path)
            restored = ObjectMemory.load(path)
            self.assertEqual(list(memory.objects), list(restored.objects))
            original = next(iter(memory))
            recovered = next(iter(restored))
            self.assertEqual([item.obs_id for item in original.observations], ["a", "b"])
            np.testing.assert_allclose(original.fused_center, recovered.fused_center)
            self.assertEqual(original.evidence_frames, recovered.evidence_frames)
            json.loads(path.read_text())


if __name__ == "__main__":
    unittest.main()
