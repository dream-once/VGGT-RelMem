import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.validate_query_multiview import (
    measure_observation_pairs,
    validate_query_multiview,
)


def yaw_pose(
    degrees: float,
    translation: tuple[float, float, float],
) -> np.ndarray:
    radians = np.radians(degrees)
    cosine = np.cos(radians)
    sine = np.sin(radians)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = [
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ]
    pose[:3, 3] = translation
    return pose


class QueryMultiviewValidationTests(unittest.TestCase):
    def test_same_pair_must_pass_both_thresholds(self) -> None:
        poses = {
            "frame_a": yaw_pose(0.0, (0.0, 0.0, 0.0)),
            "frame_b": yaw_pose(0.0, (1.0, 0.0, 0.0)),
            "frame_c": yaw_pose(4.0, (0.1, 0.0, 0.0)),
        }

        rows = measure_observation_pairs(
            poses,
            list(poses),
            min_translation=1.0,
            min_rotation_degrees=3.0,
        )

        self.assertTrue(any(row["passes_translation"] for row in rows))
        self.assertTrue(any(row["passes_rotation"] for row in rows))
        self.assertFalse(
            any(row["passes_same_pair_gate"] for row in rows)
        )

    def test_true_multiview_pair_is_reported(self) -> None:
        poses = {
            "frame_a": yaw_pose(0.0, (0.0, 0.0, 0.0)),
            "frame_b": yaw_pose(5.0, (0.8, 0.0, 0.0)),
        }

        rows = measure_observation_pairs(
            poses,
            list(poses),
            min_translation=0.5,
            min_rotation_degrees=3.0,
        )

        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["passes_same_pair_gate"])

    def test_zero_evidence_negative_control_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = {
                "stage": "D6",
                "status": "INSUFFICIENT_MULTIFRAME_3D_EVIDENCE",
                "query": "dog",
                "sam_instances": 0,
                "lifted_instances": 0,
                "frames_with_masks": [],
                "frames_with_lifted_observations": [],
            }
            (root / "d6_result.json").write_text(json.dumps(result))
            report = validate_query_multiview(
                root,
                root / "unused_poses.json",
                expect_negative=True,
            )

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["evidence_class"], "NEGATIVE_CONTROL")


if __name__ == "__main__":
    unittest.main()
