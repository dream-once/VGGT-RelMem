from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from relground.clio_retrieval_evaluation import (
    box_intersects_frame,
    read_colmap_cameras,
    read_colmap_image_records,
    slugify_task,
)


class ClioRetrievalEvaluationTests(unittest.TestCase):
    def test_reads_colmap_camera_and_image_records(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cameras_path = root / "cameras.bin"
            cameras_path.write_bytes(
                struct.pack("<QiiQQ4d", 1, 7, 2, 640, 480, 500.0, 320.0, 240.0, 0.01)
            )
            images_path = root / "images.bin"
            images_path.write_bytes(
                struct.pack("<Qi4d3di", 1, 9, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7)
                + b"rgb_12.jpg\0"
                + struct.pack("<Q", 0)
            )
            camera = read_colmap_cameras(cameras_path)[7]
            image = read_colmap_image_records(images_path)["rgb_12"]
            self.assertEqual(camera["model_id"], 2)
            self.assertEqual((camera["width"], camera["height"]), (640, 480))
            self.assertEqual(image["camera_id"], 7)
            np.testing.assert_allclose(image["camera_from_colmap_rotation"], np.eye(3))

    def test_oriented_box_frustum_intersection(self) -> None:
        box = {
            "center": [0.0, 0.0, 3.0],
            "extents": [1.0, 1.0, 1.0],
            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        }
        image = {
            "camera_id": 1,
            "camera_from_colmap_rotation": np.eye(3),
            "camera_from_colmap_translation": np.zeros(3),
        }
        camera = {
            "model_id": 2,
            "width": 640,
            "height": 480,
            "params": np.asarray([500.0, 320.0, 240.0, 0.0]),
        }
        self.assertTrue(box_intersects_frame(
            box,
            image=image,
            camera=camera,
            world_from_colmap_scale=1.0,
            world_from_colmap_rotation=np.eye(3),
            world_from_colmap_translation=np.zeros(3),
        ))
        box["center"] = [10.0, 0.0, 3.0]
        self.assertFalse(box_intersects_frame(
            box,
            image=image,
            camera=camera,
            world_from_colmap_scale=1.0,
            world_from_colmap_rotation=np.eye(3),
            world_from_colmap_translation=np.zeros(3),
        ))

    def test_task_slug_matches_artifact_names(self) -> None:
        self.assertEqual(slugify_task("get can of WD-40"), "get-can-of-wd-40")
        self.assertEqual(slugify_task("bring me a pillow"), "bring-me-a-pillow")


if __name__ == "__main__":
    unittest.main()
