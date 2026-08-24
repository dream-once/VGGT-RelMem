import unittest

from scripts.run_vggt_geometry import _select_image_names


class GeometryFrameSelectionTests(unittest.TestCase):
    def test_start_stride_and_limit_are_applied_after_sorting(self) -> None:
        names = [f"frame_{index:04d}.jpg" for index in range(1, 13)]

        selected = _select_image_names(
            names,
            frame_start=1,
            frame_stride=3,
            max_frames=3,
        )

        self.assertEqual(
            selected,
            ["frame_0002.jpg", "frame_0005.jpg", "frame_0008.jpg"],
        )

    def test_invalid_selection_parameters_are_rejected(self) -> None:
        names = ["frame_0001.jpg", "frame_0002.jpg"]
        for kwargs in (
            {"frame_start": -1},
            {"frame_stride": 0},
            {"max_frames": 0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    _select_image_names(names, **kwargs)


if __name__ == "__main__":
    unittest.main()
