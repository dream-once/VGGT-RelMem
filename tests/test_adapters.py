import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from adapters.geometry import (
    GEOMETRY_SCHEMA_VERSION,
    LEGACY_GEOMETRY_SCHEMA_VERSION,
    load_anchor_poses,
    load_geometry_npz,
    save_geometry_npz,
)
from adapters.masks import (
    MaskRecord,
    load_mask,
    load_mask_manifest,
    save_mask_manifest,
)


class AdapterTests(unittest.TestCase):
    def test_legacy_geometry_and_mask_contract_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            geometry_path = root / "geometry.npz"
            save_geometry_npz(
                geometry_path,
                frame_ids=["f0"],
                point_maps=np.zeros((1, 3, 4, 3)),
                confidence_maps=np.ones((1, 3, 4)),
                world_from_camera=np.eye(4)[None],
            )
            geometry = load_geometry_npz(geometry_path)
            self.assertEqual(
                geometry.schema_version,
                LEGACY_GEOMETRY_SCHEMA_VERSION,
            )
            self.assertIsNone(geometry.raw_confidence_maps)
            self.assertTrue(np.all(geometry.valid_masks))
            self.assertEqual(
                geometry.get("f0").point_map.shape,
                (3, 4, 3),
            )

            np.save(
                root / "mask.npy",
                np.ones((3, 4), dtype=bool),
            )
            manifest_path = root / "masks.json"
            record = MaskRecord(
                "o0",
                "f0",
                "chair",
                "mask.npy",
                0.8,
                0.9,
            )
            save_mask_manifest(manifest_path, [record])
            restored = load_mask_manifest(manifest_path)[0]
            self.assertTrue(
                np.all(load_mask(restored, manifest_path))
            )

    def test_geometry_02_preserves_raw_confidence_and_valid_mask(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "geometry.npz"
            raw = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
            valid = raw > 5
            save_geometry_npz(
                path,
                frame_ids=["f0"],
                point_maps=np.zeros((1, 3, 4, 3)),
                confidence_maps=valid.astype(np.float32),
                world_from_camera=np.eye(4)[None],
                raw_confidence_maps=raw,
                valid_masks=valid,
            )

            geometry = load_geometry_npz(path)
            frame = geometry.get("f0")

            self.assertEqual(
                geometry.schema_version,
                GEOMETRY_SCHEMA_VERSION,
            )
            np.testing.assert_array_equal(
                geometry.raw_confidence_maps,
                raw,
            )
            np.testing.assert_array_equal(
                geometry.valid_masks,
                valid,
            )
            np.testing.assert_array_equal(
                frame.raw_confidence_map,
                raw[0],
            )
            np.testing.assert_array_equal(
                frame.valid_mask,
                valid[0],
            )

    def test_anchor_pose_contract_requires_rigid_complete_poses(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anchor_poses.json"
            pose = np.eye(4)
            pose[:3, 3] = [1.0, 2.0, 3.0]
            path.write_text(json.dumps({"f0": pose.tolist()}))
            restored = load_anchor_poses(
                path,
                required_frame_ids=["f0"],
            )
            np.testing.assert_allclose(restored["f0"], pose)

            with self.assertRaisesRegex(ValueError, "missing frames"):
                load_anchor_poses(
                    path,
                    required_frame_ids=["f0", "f1"],
                )
            pose[0, 0] = 2.0
            path.write_text(json.dumps({"f0": pose.tolist()}))
            with self.assertRaisesRegex(ValueError, "not rigid"):
                load_anchor_poses(path)


if __name__ == "__main__":
    unittest.main()
