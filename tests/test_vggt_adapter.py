import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from adapters.geometry import load_geometry_npz
from adapters.vggt_slam import export_solver_geometry
from scripts.validate_geometry import validate_geometry


class FakeGraph:
    def get_homography(self, node_id):
        result = np.eye(4)
        result[0, 3] = node_id + 1
        return result


class FakeSubmap:
    def __init__(self, submap_id, names, loop=False):
        self._id = submap_id
        self._loop = loop
        self.img_names = names
        self.pointclouds = np.zeros((len(names), 2, 3, 3), dtype=np.float32)
        self.conf = np.stack([np.arange(6).reshape(2, 3) for _ in names])

    def get_id(self):
        return self._id

    def get_lc_status(self):
        return self._loop

    def get_conf_threshold(self):
        return 2.5

    def get_all_poses_world(self, graph):
        return np.stack([np.eye(4) for _ in self.img_names])


class FakeMap:
    def ordered_submaps_by_key(self):
        return iter([
            FakeSubmap(0, ["images/frame_0001.png", "images/frame_0002.png"]),
            FakeSubmap(2, ["images/frame_0002.png", "images/frame_0003.png"]),
            FakeSubmap(4, ["images/frame_0099.png"], loop=True),
        ])


class FakeSolver:
    graph = FakeGraph()
    map = FakeMap()


class VGGTAdapterTests(unittest.TestCase):
    def test_export_deduplicates_and_preserves_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "geometry.npz"
            summary = export_solver_geometry(FakeSolver(), path, source_commit="test")
            geometry = load_geometry_npz(path)
            self.assertEqual(summary.frame_count, 3)
            self.assertEqual(summary.skipped_duplicate_frames, 1)
            self.assertEqual(summary.skipped_loop_closure_submaps, 1)
            self.assertEqual(geometry.frame_ids, ["frame_0001", "frame_0002", "frame_0003"])
            np.testing.assert_allclose(geometry.world_from_camera[2, :3, 3], [4, 0, 0])
            self.assertEqual(set(np.unique(geometry.confidence_maps)), {0.0, 1.0})
            self.assertEqual(geometry.schema_version, "0.2")
            np.testing.assert_array_equal(
                geometry.raw_confidence_maps[0],
                np.arange(6).reshape(2, 3),
            )
            np.testing.assert_array_equal(
                geometry.valid_masks,
                geometry.confidence_maps.astype(bool),
            )
            manifest = json.loads(Path(summary.manifest_path).read_text())
            self.assertEqual(manifest["source_commit"], "test")
            self.assertEqual(manifest["schema_version"], "0.2")
            self.assertEqual(
                set(manifest["confidence_encoding"]),
                {"raw_confidence_maps", "valid_masks", "confidence_maps"},
            )
            poses = json.loads(Path(summary.anchor_poses_path).read_text())
            self.assertEqual(set(poses), set(geometry.frame_ids))
            (path.parent / "run_manifest.json").write_text(json.dumps({
                "config": {"upstream_commit": "a" * 40}
            }))
            report = validate_geometry(path)
            self.assertEqual(report["status"], "PASS")

            self.assertTrue(report["raw_confidence_available"])

if __name__ == "__main__":
    unittest.main()
