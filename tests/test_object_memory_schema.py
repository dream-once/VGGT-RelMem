import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from relground.association import ObjectMemory
from relground.schemas import (
    OBJECT_MEMORY_SCHEMA_VERSION,
    ObjectObservation,
    OrientedBoundingBox,
)


def make_observation(obs_id: str, frame_id: str) -> ObjectObservation:
    center = np.array([float(len(obs_id)), 0.0, 1.0])
    return ObjectObservation(
        obs_id=obs_id,
        class_text="trash can",
        frame_id=frame_id,
        mask_ref=f"masks/{obs_id}.npy",
        retrieval_score=0.8,
        sam_score=0.9,
        valid_point_ratio=0.7,
        points_ref=f"points/{obs_id}.npz",
        center=center,
        obb=OrientedBoundingBox(center, np.ones(3)),
    )


class ObjectMemorySchemaTests(unittest.TestCase):
    def test_pending_observations_round_trip_without_objects(self) -> None:
        memory = ObjectMemory(metadata={"scene_id": "scene"})
        observations = [
            make_observation("obs_a", "frame_0001"),
            make_observation("obs_b", "frame_0011"),
        ]
        memory.stage_many(observations)
        self.assertEqual(len(memory), 0)
        self.assertEqual(memory.decisions, [])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            memory.save(path)
            restored = ObjectMemory.load(path)

        self.assertEqual(
            restored.schema_version,
            OBJECT_MEMORY_SCHEMA_VERSION,
        )
        self.assertEqual(
            list(restored.pending_observations),
            ["obs_a", "obs_b"],
        )
        self.assertEqual(memory.to_dict(), restored.to_dict())
        self.assertEqual(len(restored), 0)

    def test_observation_becomes_object_only_after_explicit_association(self) -> None:
        observation = make_observation("obs_a", "frame_0001")
        memory = ObjectMemory()
        memory.stage_many([observation])

        memory.add_observation(observation)

        self.assertEqual(memory.pending_observations, {})
        self.assertEqual(len(memory), 1)
        self.assertEqual(len(memory.decisions), 1)

    def test_tampered_evidence_is_rejected(self) -> None:
        memory = ObjectMemory()
        memory.stage_many([make_observation("obs_a", "frame_0001")])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            memory.save(path)
            payload = json.loads(path.read_text())
            payload["evidence"]["pending_observation_ids"].append("fake")
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "evidence is inconsistent"):
                ObjectMemory.load(path)


if __name__ == "__main__":
    unittest.main()
