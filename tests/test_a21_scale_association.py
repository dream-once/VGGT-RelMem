import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from relground.a21_scale_association import (
    ScaleAwareAssociationConfig,
    associate_pending_a21,
    compute_scale_aware_pair,
)
from relground.a2_association import EvidenceAssociationConfig, compute_pair
from relground.association import ObjectMemory
from relground.schemas import ObjectObservation, OrientedBoundingBox
from scripts.run_a21_scale_association import build_result
from scripts.validate_a21_scale_association import validate_output


def observation(obs_id, frame_id, center, *, extent=0.1, quality=0.9):
    center = np.asarray(center, dtype=float)
    return ObjectObservation(
        obs_id=obs_id,
        class_text="pillow",
        frame_id=frame_id,
        mask_ref=f"masks/{obs_id}.npy",
        retrieval_score=quality,
        sam_score=quality,
        valid_point_ratio=quality,
        points_ref=f"points/{obs_id}.npy",
        center=center,
        obb=OrientedBoundingBox(center=center, extent=np.full(3, extent)),
    )


class A21ScaleAssociationTests(unittest.TestCase):
    def test_scale_normalization_handles_same_relative_geometry(self):
        small_a = observation("a", "f1", [0, 0, 0], extent=0.1)
        small_b = observation("b", "f2", [0.15, 0, 0], extent=0.1)
        large_a = observation("c", "f1", [0, 0, 0], extent=1.0)
        large_b = observation("d", "f2", [1.5, 0, 0], extent=1.0)
        _, first = compute_scale_aware_pair(small_a, small_b, ScaleAwareAssociationConfig())
        _, second = compute_scale_aware_pair(large_a, large_b, ScaleAwareAssociationConfig())
        self.assertAlmostEqual(first["normalized_center_distance"], second["normalized_center_distance"])
        self.assertEqual(first["scale_center_pass"], second["scale_center_pass"])

    def test_low_quality_is_not_overridden_by_scale(self):
        good = observation("a", "f1", [0, 0, 0], quality=0.9)
        weak = observation("b", "f2", [0.01, 0, 0], quality=0.1)
        pair, _ = compute_scale_aware_pair(good, weak, ScaleAwareAssociationConfig())
        self.assertTrue(pair.center_pass)
        self.assertFalse(pair.quality_pass)
        self.assertFalse(pair.gate_pass)

    def test_far_same_class_objects_remain_separate(self):
        memory = ObjectMemory(metadata={"scene_id": "s", "query": "pillow"})
        memory.stage_many([
            observation("a", "f1", [0, 0, 0], extent=0.1),
            observation("b", "f2", [1, 0, 0], extent=0.1),
        ])
        outcome = associate_pending_a21(memory, ScaleAwareAssociationConfig())
        self.assertFalse(outcome["pairs"][0]["predicted_same"])
        self.assertEqual(len(memory.objects), 0)

    def test_cross_frame_scaled_pair_promotes_object(self):
        memory = ObjectMemory(metadata={"scene_id": "s", "query": "pillow"})
        memory.stage_many([
            observation("a", "f1", [0, 0, 0], extent=0.2),
            observation("b", "f2", [0.2, 0, 0], extent=0.2),
        ])
        outcome = associate_pending_a21(memory, ScaleAwareAssociationConfig())
        self.assertTrue(outcome["pairs"][0]["predicted_same"])
        self.assertEqual(len(memory.objects), 1)

    def test_frozen_a2_absolute_gate_is_unchanged(self):
        first = observation("a", "f1", [0, 0, 0], extent=1.0)
        second = observation("b", "f2", [0.2, 0, 0], extent=1.0)
        a2 = compute_pair(first, second, EvidenceAssociationConfig())
        a21, diagnostics = compute_scale_aware_pair(first, second, ScaleAwareAssociationConfig())
        self.assertFalse(a2.center_pass)
        self.assertTrue(diagnostics["scale_center_pass"])
        self.assertTrue(a21.gate_pass)

    def test_bundle_replays_and_source_tampering_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            memory = ObjectMemory(
                metadata={"scene_id": "s", "query": "pillow"}
            )
            memory.stage_many([
                observation("a", "f1", [0, 0, 0], extent=0.2),
                observation("b", "f2", [0.2, 0, 0], extent=0.2),
            ])
            memory.save(source)
            output = root / "prediction"
            build_result(
                source_memory_path=source,
                output_dir=output,
                config=ScaleAwareAssociationConfig(),
            )
            self.assertEqual(validate_output(output)["status"], "PASS")
            source_copy = output / "source_memory.json"
            source_copy.write_text(source_copy.read_text() + "\n")
            self.assertEqual(validate_output(output)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
