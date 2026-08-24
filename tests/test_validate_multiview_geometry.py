import unittest

import numpy as np

from scripts.validate_multiview_geometry import measure_viewpoint_spread


def yaw_pose(degrees: float, translation: tuple[float, float, float]) -> np.ndarray:
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


class MultiviewGeometryValidationTests(unittest.TestCase):
    def test_measure_viewpoint_spread_reports_pairwise_maxima(self) -> None:
        poses = {
            "frame_0001": yaw_pose(0.0, (0.0, 0.0, 0.0)),
            "frame_0011": yaw_pose(5.0, (1.0, 0.0, 0.0)),
            "frame_0021": yaw_pose(10.0, (0.0, 2.0, 0.0)),
        }

        spread = measure_viewpoint_spread(poses, list(poses))

        self.assertAlmostEqual(spread["max_translation"], np.sqrt(5.0))
        self.assertEqual(
            spread["max_translation_pair"],
            ["frame_0011", "frame_0021"],
        )
        self.assertAlmostEqual(spread["max_rotation_degrees"], 10.0)
        self.assertEqual(
            spread["max_rotation_pair"],
            ["frame_0001", "frame_0021"],
        )

    def test_missing_required_pose_is_rejected(self) -> None:
        poses = {"frame_0001": np.eye(4)}

        with self.assertRaisesRegex(ValueError, "missing anchor poses"):
            measure_viewpoint_spread(
                poses,
                ["frame_0001", "frame_0011"],
            )


if __name__ == "__main__":
    unittest.main()
