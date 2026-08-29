"""Boundary tests for the D15.5 object-centric visualization contract."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from relground.scene_visualization import (
    audit_object_viewpoints,
    load_observation_points,
    obb_edges,
)
from scripts.validate_d15_5_visualization import _recompute_evidence


class D155VisualizationTests(unittest.TestCase):
    @staticmethod
    def _pose(
        center: tuple[float, float, float],
        rotation: np.ndarray | None = None,
    ) -> list[list[float]]:
        pose = np.eye(4, dtype=np.float64)
        if rotation is not None:
            pose[:3, :3] = rotation
        pose[:3, 3] = np.asarray(center, dtype=np.float64)
        return pose.tolist()

    def _audit(
        self,
        cameras: dict[str, list[list[float]]],
        *,
        observation_frames: list[str] | None = None,
        strict_angle_deg: float = 15.0,
        strict_ratio: float = 0.2,
    ) -> dict:
        frames = list(cameras)
        member_frames = frames if observation_frames is None else observation_frames
        observations = [
            {
                "obs_id": f"obs_{index:04d}",
                "frame_id": frame_id,
                "class_text": "trash can",
                "center": [0.0, 0.0, 0.0],
                "retrieval_score": 0.9,
                "sam_score": 0.8,
                "valid_point_ratio": 0.7,
            }
            for index, frame_id in enumerate(member_frames, start=1)
        ]
        association = {
            "metadata": {"scene_id": "synthetic", "query": "trash can"},
            "objects": [
                {
                    "object_id": "obj_0001",
                    "observation_ids": [
                        observation["obs_id"] for observation in observations
                    ],
                }
            ],
        }
        cache = {
            "scene_id": "synthetic",
            "query": "trash can",
            "frame_ids": frames,
            "observations": observations,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_path = root / "observations.json"
            association_path = root / "object_memory.json"
            anchor_path = root / "anchors.json"
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
            association_path.write_text(json.dumps(association), encoding="utf-8")
            anchor_path.write_text(json.dumps(cameras), encoding="utf-8")
            return audit_object_viewpoints(
                cache_path,
                association_path,
                anchor_path,
                strict_angle_deg=strict_angle_deg,
                strict_ratio=strict_ratio,
            )

    def test_three_object_centric_views_pass_strict_gate(self) -> None:
        audit = self._audit(
            {
                "frame_0001": self._pose((0.0, 0.0, -1.0)),
                "frame_0002": self._pose((1.0, 0.0, 0.0)),
                "frame_0003": self._pose((0.0, 1.0, 0.0)),
            }
        )
        self.assertEqual(audit["evidence_status"], "STRONG_OBJECT_CENTRIC_MULTIVIEW")
        record = audit["objects"][0]
        self.assertEqual(record["evidence_status"], "STRICT_MULTIVIEW")
        self.assertEqual(record["distinct_frame_count"], 3)
        self.assertGreaterEqual(len(record["strict_qualifying_pairs"]), 2)
        self.assertEqual(len(record["strict_covered_frames"]), 3)

    def test_in_place_rotation_is_not_parallax(self) -> None:
        angle = math.radians(45.0)
        rotation = np.asarray(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        audit = self._audit(
            {
                "frame_0001": self._pose((0.0, 0.0, -1.0)),
                "frame_0002": self._pose((0.0, 0.0, -1.0), rotation),
                "frame_0003": self._pose((0.0, 0.0, -1.0), rotation.T),
            }
        )
        record = audit["objects"][0]
        self.assertEqual(record["evidence_status"], "WEAK_OR_SINGLE_VIEW")
        self.assertTrue(all(pair["angle_deg"] == 0.0 for pair in record["pair_metrics"]))
        self.assertTrue(all(pair["baseline"] == 0.0 for pair in record["pair_metrics"]))

    def test_translation_along_same_object_ray_is_not_parallax(self) -> None:
        audit = self._audit(
            {
                "frame_0001": self._pose((0.0, 0.0, -1.0)),
                "frame_0002": self._pose((0.0, 0.0, -2.0)),
                "frame_0003": self._pose((0.0, 0.0, -3.0)),
            }
        )
        record = audit["objects"][0]
        self.assertEqual(record["evidence_status"], "WEAK_OR_SINGLE_VIEW")
        self.assertTrue(all(pair["angle_deg"] == 0.0 for pair in record["pair_metrics"]))
        self.assertTrue(
            any(pair["baseline_depth_ratio"] > 0.2 for pair in record["pair_metrics"])
        )

    def test_angle_and_baseline_must_pass_on_the_same_pair(self) -> None:
        ray_b = math.radians(20.0)
        ray_c = math.radians(11.0)
        audit = self._audit(
            {
                "frame_a": self._pose((0.0, 0.0, -100.0)),
                "frame_b": self._pose(
                    (
                        -100.0 * math.sin(ray_b),
                        0.0,
                        -100.0 * math.cos(ray_b),
                    )
                ),
                "frame_c": self._pose(
                    (-math.sin(ray_c), 0.0, -math.cos(ray_c))
                ),
            },
            strict_angle_deg=15.0,
            strict_ratio=1.0,
        )
        pairs = audit["objects"][0]["pair_metrics"]
        self.assertTrue(any(pair["angle_deg"] >= 15.0 for pair in pairs))
        self.assertTrue(any(pair["baseline_depth_ratio"] >= 1.0 for pair in pairs))
        self.assertFalse(
            any(
                pair["angle_deg"] >= 15.0
                and pair["baseline_depth_ratio"] >= 1.0
                for pair in pairs
            )
        )
        self.assertNotEqual(
            audit["objects"][0]["evidence_status"], "STRICT_MULTIVIEW"
        )

    def test_duplicate_masks_in_one_frame_do_not_add_views(self) -> None:
        audit = self._audit(
            {"frame_0001": self._pose((0.0, 0.0, -1.0))},
            observation_frames=["frame_0001", "frame_0001"],
        )
        record = audit["objects"][0]
        self.assertEqual(record["distinct_frame_count"], 1)
        self.assertEqual(len(record["frame_evidence"][0]["observation_ids"]), 2)
        self.assertEqual(record["evidence_status"], "WEAK_OR_SINGLE_VIEW")

    def test_point_reference_cannot_escape_cache_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "unsafe artifact reference"):
                load_observation_points(root, "../outside.npy")

    def test_obb_wireframe_is_canonical_and_deterministic(self) -> None:
        first_corners, first_edges = obb_edges(
            [1.0, 2.0, 3.0], np.eye(3), [2.0, 4.0, 6.0]
        )
        second_corners, second_edges = obb_edges(
            [1.0, 2.0, 3.0], np.eye(3), [2.0, 4.0, 6.0]
        )
        self.assertEqual(first_corners.shape, (8, 3))
        self.assertEqual(first_edges.shape, (12, 2))
        np.testing.assert_array_equal(first_corners, second_corners)
        np.testing.assert_array_equal(first_edges, second_edges)
        np.testing.assert_array_equal(first_corners.min(axis=0), [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(first_corners.max(axis=0), [2.0, 4.0, 6.0])

    def test_validator_recompute_retains_zero_status_counts(self) -> None:
        audit = self._audit(
            {
                "frame_0001": self._pose((0.0, 0.0, -1.0)),
                "frame_0002": self._pose((1.0, 0.0, 0.0)),
                "frame_0003": self._pose((0.0, 1.0, 0.0)),
            }
        )
        recomputed = _recompute_evidence(audit)
        self.assertEqual(
            recomputed["status_counts"],
            {
                "STRICT_MULTIVIEW": 1,
                "DIAGNOSTIC_PARALLAX": 0,
                "WEAK_OR_SINGLE_VIEW": 0,
            },
        )


if __name__ == "__main__":
    unittest.main()
