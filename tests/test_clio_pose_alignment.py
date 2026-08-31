import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from relground.clio_pose_alignment import (
    build_vggt_to_colmap_alignment,
    estimate_sim3,
    read_colmap_world_from_camera,
    validate_vggt_to_colmap_alignment,
)


def write_images_bin(path: Path, poses: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(poses)))
        for image_id, (name, world_from_camera) in enumerate(poses.items(), start=1):
            rotation = world_from_camera[:3, :3].T
            trace = float(np.trace(rotation))
            qw = np.sqrt(max(0.0, 1.0 + trace)) / 2.0
            qx = (rotation[2, 1] - rotation[1, 2]) / (4.0 * qw)
            qy = (rotation[0, 2] - rotation[2, 0]) / (4.0 * qw)
            qz = (rotation[1, 0] - rotation[0, 1]) / (4.0 * qw)
            tvec = -rotation @ world_from_camera[:3, 3]
            handle.write(struct.pack("<i4d3di", image_id, qw, qx, qy, qz, *tvec, image_id))
            handle.write(f"{name}.jpg".encode() + b"\0")
            handle.write(struct.pack("<Q", 0))


class ClioPoseAlignmentTests(unittest.TestCase):
    def test_colmap_reader_returns_world_from_camera(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "images.bin"
            pose = np.eye(4)
            pose[:3, 3] = [1.0, 2.0, 3.0]
            write_images_bin(path, {"rgb_0": pose})
            actual = read_colmap_world_from_camera(path)["rgb_0"]
            np.testing.assert_allclose(actual, pose, atol=1e-12)

    def test_umeyama_recovers_known_sim3(self) -> None:
        source = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        rotation = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        target = (2.5 * (rotation @ source.T)).T + np.asarray([3.0, -2.0, 1.0])
        scale, actual_rotation, translation = estimate_sim3(source, target)
        self.assertAlmostEqual(scale, 2.5)
        np.testing.assert_allclose(actual_rotation, rotation, atol=1e-12)
        np.testing.assert_allclose(translation, [3.0, -2.0, 1.0], atol=1e-12)

    def test_build_and_validate_alignment(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            anchors: dict[str, list[list[float]]] = {}
            colmap: dict[str, np.ndarray] = {}
            rotation = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
            for index, center in enumerate(([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [2, 1, 0])):
                anchor = np.eye(4)
                anchor[:3, 3] = center
                anchors[f"rgb_{index}"] = anchor.tolist()
                target = np.eye(4)
                target[:3, 3] = 2.5 * (rotation @ np.asarray(center, dtype=float)) + [3.0, -2.0, 1.0]
                colmap[f"rgb_{index}"] = target
            anchor_path = root / "runs/anchors.json"
            anchor_path.parent.mkdir(parents=True)
            anchor_path.write_text(json.dumps(anchors))
            images_path = root / "data/images.bin"
            write_images_bin(images_path, colmap)
            payload = build_vggt_to_colmap_alignment(project_root=root, anchor_poses_path=anchor_path, colmap_images_path=images_path, created_at="fixed", max_rmse_colmap_units=0.15)
            self.assertEqual(payload["status"], "PASS")
            self.assertEqual(payload["matches"]["count"], 6)
            self.assertEqual(payload["contract"]["task_gt_coordinate_alignment"], "ROS_BAG_ALIGNMENT_PENDING")
            self.assertEqual(validate_vggt_to_colmap_alignment(payload, project_root=root)["status"], "PASS")
            cubicle_payload = build_vggt_to_colmap_alignment(
                project_root=root,
                anchor_poses_path=anchor_path,
                colmap_images_path=images_path,
                created_at="fixed",
                max_rmse_colmap_units=0.15,
                scene_id="cubicle",
                split_role="held-out",
            )
            self.assertEqual(cubicle_payload["scene_id"], "cubicle")
            self.assertEqual(cubicle_payload["split_role"], "held-out")
            self.assertEqual(
                validate_vggt_to_colmap_alignment(cubicle_payload, project_root=root)["status"],
                "PASS",
            )
            payload["sim3"]["scale"] = 99.0
            self.assertEqual(validate_vggt_to_colmap_alignment(payload, project_root=root)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
