import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from relground.clio_world_alignment import (
    build_vggt_to_clio_world_alignment,
    validate_vggt_to_clio_world_alignment,
)


class ClioWorldAlignmentTests(unittest.TestCase):
    def write_colmap_alignment(self, root: Path) -> Path:
        path = root / "runs/colmap.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            "status": "PASS",
            "contract": {"main_inference_may_read_alignment": False},
            "sim3": {
                "scale": 2.0,
                "rotation": np.eye(3).tolist(),
                "translation": [1.0, 2.0, 3.0],
            },
            "error_colmap_units": {"rmse": 0.1, "median": 0.05, "max": 0.2, "threshold_rmse": 0.3},
        }))
        return path

    def write_scene_transform(self, root: Path) -> Path:
        path = root / "configs/transforms.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            "schema_version": "0.1",
            "source": {
                "repository": "https://example.test/clio",
                "commit": "frozen",
                "path": "scene_transforms.yaml",
                "sha256": "upstream-hash",
            },
            "scenes": {
                "apartment": {
                    "split_role": "development",
                    "scale": 0.5,
                    "T_world_from_scaled_colmap": [
                        [0, -1, 0, 10],
                        [1, 0, 0, 20],
                        [0, 0, 1, 30],
                        [0, 0, 0, 1],
                    ],
                },
                "cubicle": {
                    "split_role": "held-out",
                    "scale": 0.25,
                    "T_world_from_scaled_colmap": [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ],
                },
            },
        }))
        return path

    def test_compose_and_deterministically_replay(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            colmap_path = self.write_colmap_alignment(root)
            transform_path = self.write_scene_transform(root)
            payload = build_vggt_to_clio_world_alignment(
                project_root=root,
                colmap_alignment_path=colmap_path,
                scene_transform_path=transform_path,
                created_at="fixed",
            )
            self.assertEqual(payload["status"], "PASS")
            self.assertAlmostEqual(payload["sim3"]["scale"], 1.0)
            np.testing.assert_allclose(payload["sim3"]["translation"], [9.0, 20.5, 31.5])
            self.assertAlmostEqual(payload["error_m"]["rmse"], 0.05)
            self.assertEqual(
                validate_vggt_to_clio_world_alignment(payload, project_root=root)["status"],
                "PASS",
            )
            cubicle_payload = build_vggt_to_clio_world_alignment(
                project_root=root,
                colmap_alignment_path=colmap_path,
                scene_transform_path=transform_path,
                scene_id="cubicle",
                created_at="fixed",
            )
            self.assertEqual(cubicle_payload["scene_id"], "cubicle")
            self.assertEqual(cubicle_payload["split_role"], "held-out")
            self.assertAlmostEqual(cubicle_payload["sim3"]["scale"], 0.5)
            self.assertEqual(
                validate_vggt_to_clio_world_alignment(cubicle_payload, project_root=root)["status"],
                "PASS",
            )
            payload["sim3"]["translation"][0] += 1.0
            self.assertEqual(
                validate_vggt_to_clio_world_alignment(payload, project_root=root)["status"],
                "FAIL",
            )

    def test_rejects_non_rigid_official_transform(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            colmap_path = self.write_colmap_alignment(root)
            transform_path = self.write_scene_transform(root)
            config = json.loads(transform_path.read_text())
            config["scenes"]["apartment"]["T_world_from_scaled_colmap"][0][0] = 2.0
            transform_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "not orthonormal"):
                build_vggt_to_clio_world_alignment(
                    project_root=root,
                    colmap_alignment_path=colmap_path,
                    scene_transform_path=transform_path,
                )

    def test_rejects_prediction_visible_alignment(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            colmap_path = self.write_colmap_alignment(root)
            payload = json.loads(colmap_path.read_text())
            payload["contract"]["main_inference_may_read_alignment"] = True
            colmap_path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "not evaluator-only"):
                build_vggt_to_clio_world_alignment(
                    project_root=root,
                    colmap_alignment_path=colmap_path,
                    scene_transform_path=self.write_scene_transform(root),
                )


if __name__ == "__main__":
    unittest.main()
