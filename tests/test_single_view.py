import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from relground.observations import LifterConfig, Robust3DLifter
from relground.single_view import (
    B0_OFFICIAL,
    VGGTImageTransform,
    compute_vggt_crop_transform,
    load_vggt_sam_image,
    make_official_observation,
    official_pca_lift,
)


class SingleViewBaselineTests(unittest.TestCase):
    def test_landscape_transform_matches_office_loop_grid(self) -> None:
        transform = compute_vggt_crop_transform(
            (1920, 1080),
            (294, 518),
        )
        self.assertEqual(transform.resized_size, (518, 294))
        self.assertEqual(transform.crop_xyxy, (0, 0, 518, 294))
        self.assertEqual(transform.padding_ltrb, (0, 0, 0, 0))
        self.assertEqual(transform.output_shape, (294, 518))
        self.assertEqual(
            VGGTImageTransform.from_dict(transform.to_dict()),
            transform,
        )

    def test_portrait_transform_center_crops_like_upstream(self) -> None:
        transform = compute_vggt_crop_transform(
            (600, 1200),
            (518, 518),
        )
        self.assertEqual(transform.resized_size, (518, 1036))
        self.assertEqual(transform.crop_xyxy, (0, 259, 518, 777))
        self.assertEqual(transform.padding_ltrb, (0, 0, 0, 0))

    def test_batch_padding_and_rgba_white_background_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transparent.png"
            pixels = np.zeros((500, 1000, 4), dtype=np.uint8)
            pixels[..., :3] = [255, 0, 0]
            pixels[..., 3] = 0
            Image.fromarray(pixels, mode="RGBA").save(path)
            image, transform = load_vggt_sam_image(path, (294, 518))
            self.assertEqual(transform.resized_size, (518, 252))
            self.assertEqual(transform.padding_ltrb, (0, 21, 0, 21))
            self.assertEqual(image.size, (518, 294))
            self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
            self.assertEqual(image.getpixel((259, 147)), (255, 255, 255))

    def test_official_lift_keeps_finite_outlier_while_robust_removes_it(self) -> None:
        x, y = np.meshgrid(np.arange(6, dtype=float), np.arange(6, dtype=float))
        point_map = np.stack([x, y, np.zeros_like(x)], axis=-1)
        point_map[0, 0] = [1000.0, -1000.0, 500.0]
        mask = np.ones((6, 6), dtype=bool)
        official = official_pca_lift(mask, point_map)
        robust = Robust3DLifter(
            LifterConfig(min_points=10, outlier_mad_scale=3.5)
        ).lift(mask, point_map, np.ones((6, 6)))
        self.assertEqual(len(official.points), 36)
        self.assertEqual(len(robust.points), 35)
        self.assertGreater(float(official.obb.extent.max()), 100.0)
        self.assertLess(float(robust.obb.extent.max()), 10.0)

    def test_official_observation_marks_baseline_and_applies_transform(self) -> None:
        point_map = np.zeros((2, 2, 3), dtype=float)
        point_map[..., 0] = [[0.0, 1.0], [0.0, 1.0]]
        point_map[..., 1] = [[0.0, 0.0], [1.0, 1.0]]
        transform = np.eye(4)
        transform[:3, 3] = [2.0, 3.0, 4.0]
        observation, points = make_official_observation(
            obs_id="b0_f0_000",
            class_text="chair",
            frame_id="f0",
            mask=np.ones((2, 2), dtype=bool),
            point_map=point_map,
            global_from_submap=transform,
            retrieval_score=0.5,
            sam_score=0.8,
            mask_ref="masks/sam_f0_000.npy",
            points_ref="b0_official/points/b0_f0_000.npz",
        )
        self.assertEqual(observation.metadata["baseline_id"], B0_OFFICIAL)
        np.testing.assert_allclose(points.min(axis=0), [2.0, 3.0, 4.0])
        np.testing.assert_allclose(points.max(axis=0), [3.0, 4.0, 4.0])

    def test_preprocess_schema_rejects_unexpected_fields(self) -> None:
        payload = compute_vggt_crop_transform(
            (1920, 1080),
            (294, 518),
        ).to_dict()
        payload["untracked"] = True
        with self.assertRaisesRegex(ValueError, "unexpected"):
            VGGTImageTransform.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
