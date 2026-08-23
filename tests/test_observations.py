import unittest

import numpy as np

from relground.observations import LifterConfig, LiftingError, Robust3DLifter


class ObservationTests(unittest.TestCase):
    def setUp(self) -> None:
        x, y = np.meshgrid(np.arange(6, dtype=float), np.arange(6, dtype=float))
        self.point_map = np.stack([x, y, np.zeros_like(x)], axis=-1)
        self.mask = np.ones((6, 6), dtype=bool)

    def test_lift_applies_camera_to_world_transform(self) -> None:
        transform = np.eye(4)
        transform[:3, 3] = [1.0, 2.0, 3.0]
        lifter = Robust3DLifter(LifterConfig(min_points=10))
        result = lifter.lift(self.mask, self.point_map, np.ones((6, 6)), transform)
        np.testing.assert_allclose(result.center, [3.5, 4.5, 3.0], atol=1e-8)
        self.assertEqual(result.points.shape, (36, 3))
        self.assertAlmostEqual(result.valid_point_ratio, 1.0)

    def test_low_confidence_evidence_is_rejected(self) -> None:
        lifter = Robust3DLifter(LifterConfig(confidence_threshold=0.5, min_points=10))
        with self.assertRaisesRegex(LiftingError, "too few valid points"):
            lifter.lift(self.mask, self.point_map, np.zeros((6, 6)))

    def test_empty_mask_is_rejected(self) -> None:
        with self.assertRaisesRegex(LiftingError, "empty mask"):
            Robust3DLifter(LifterConfig(min_points=3)).lift(
                np.zeros((6, 6), dtype=bool), self.point_map
            )

    def test_radial_mad_filter_removes_far_outlier(self) -> None:
        point_map = self.point_map.copy()
        point_map[0, 0] = [1000.0, -1000.0, 500.0]
        lifter = Robust3DLifter(
            LifterConfig(min_points=10, outlier_mad_scale=3.5)
        )
        result = lifter.lift(self.mask, point_map, np.ones((6, 6)))
        self.assertEqual(result.points.shape, (35, 3))
        self.assertLess(float(np.max(np.linalg.norm(result.points, axis=1))), 10.0)
        self.assertAlmostEqual(result.valid_point_ratio, 35.0 / 36.0)


if __name__ == "__main__":
    unittest.main()
