import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from relground.association import ObjectMemory
from relground.observation_cache import sha256_file
from relground.schemas import (
    MEMORY_OBJECT_SCHEMA_VERSION,
    OBJECT_MEMORY_SCHEMA_VERSION,
    OBJECT_OBSERVATION_SCHEMA_VERSION,
    ObjectObservation,
    OrientedBoundingBox,
)
from scripts.validate_d8_memory import validate_output


def observation(obs_id: str, frame_id: str) -> ObjectObservation:
    center = np.array([0.0, 0.0, 1.0])
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


class D8ValidationTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> None:
        source = root / "source_observations.json"
        source.write_text("{}\n")
        source_reference = source.name
        memory = ObjectMemory(
            metadata={
                "source_cache": source_reference,
                "source_cache_sha256": sha256_file(source),
            }
        )
        memory.stage_many(
            [
                observation("obs_a", "frame_0001"),
                observation("obs_b", "frame_0011"),
            ]
        )
        memory.save(root / "object_memory.json")
        result = {
            "schema_version": OBJECT_MEMORY_SCHEMA_VERSION,
            "status": "PASS",
            "stage": "D8",
            "scene_id": "scene",
            "query": "trash can",
            "version_fields": {
                "object_memory": OBJECT_MEMORY_SCHEMA_VERSION,
                "memory_object": MEMORY_OBJECT_SCHEMA_VERSION,
                "object_observation": OBJECT_OBSERVATION_SCHEMA_VERSION,
            },
            "source": {
                "stage": "D7",
                "cache_path": source_reference,
                "cache_sha256": sha256_file(source),
            },
            "pending_observation_count": 2,
            "permanent_object_count": 0,
            "association_decision_count": 0,
            "frame_ids": ["frame_0001", "frame_0011"],
            "round_trip_equal": True,
            "artifacts": {"object_memory": "object_memory.json"},
            "created_at": "test",
        }
        (root / "d8_result.json").write_text(json.dumps(result))
        manifest = {
            "config": {
                "association_executed": False,
                "object_memory_schema": OBJECT_MEMORY_SCHEMA_VERSION,
                "source_cache": source_reference,
                "source_cache_sha256": sha256_file(source),
            }
        }
        (root / "run_manifest.json").write_text(json.dumps(manifest))

    def test_complete_schema_only_d8_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_fixture(root)
            report = validate_output(root)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["pending_observations"], 2)
        self.assertEqual(report["permanent_objects"], 0)

    def test_tampered_memory_evidence_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_fixture(root)
            path = root / "object_memory.json"
            payload = json.loads(path.read_text())
            payload["evidence"]["frame_ids"] = ["frame_fake"]
            path.write_text(json.dumps(payload))
            report = validate_output(root)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(
            any("evidence" in item for item in report["failures"])
        )

    def test_absolute_source_path_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_fixture(root)
            path = root / "d8_result.json"
            payload = json.loads(path.read_text())
            payload["source"]["cache_path"] = str(
                root / "source_observations.json"
            )
            path.write_text(json.dumps(payload))
            report = validate_output(root)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(
            any(
                "must be relative" in item
                for item in report["failures"]
            )
        )

    def test_bundle_remains_valid_after_move(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "bundle"
            root.mkdir()
            self.make_fixture(root)
            moved = parent / "moved_bundle"
            root.rename(moved)


if __name__ == "__main__":
    unittest.main()
