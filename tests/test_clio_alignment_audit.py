import sqlite3
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from relground.clio_alignment_audit import (
    build_alignment_readiness,
    validate_alignment_readiness,
)


class ClioAlignmentAuditTests(unittest.TestCase):
    def make_sparse(self, root: Path, names: list[str]) -> Path:
        sparse = root / "data/sparse/0"
        sparse.mkdir(parents=True)
        (sparse / "cameras.bin").write_bytes(b"camera")
        (sparse / "points3D.bin").write_bytes(b"points")
        with (sparse / "images.bin").open("wb") as handle:
            handle.write(struct.pack("<Q", len(names)))
            for image_id, name in enumerate(names, start=1):
                handle.write(struct.pack(
                    "<i4d3di",
                    image_id,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    float(image_id),
                    0.0,
                    0.0,
                    image_id,
                ))
                handle.write(name.encode() + b".jpg\0")
                handle.write(struct.pack("<Q", 0))
        return sparse

    def make_database(self, path: Path) -> None:
        path.parent.mkdir(parents=True)
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY, model INTEGER, width INTEGER, height INTEGER)")
            connection.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, prior_qw REAL, prior_qx REAL, prior_qy REAL, prior_qz REAL, prior_tx REAL, prior_ty REAL, prior_tz REAL)")
            connection.executemany("INSERT INTO cameras VALUES (?, 2, 640, 480)", [(1,), (2,)])
            connection.executemany("INSERT INTO images VALUES (?, ?, NULL, NULL, NULL, NULL, NULL, NULL, NULL)", [(1, "rgb_0.jpg"), (2, "rgb_1.jpg")])

    def test_missing_pose_sources_are_explicit(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "data/database.db"
            self.make_database(database)
            rgb = root / "data/images"
            rgb.mkdir()
            (rgb / "rgb_0.jpg").write_bytes(b"rgb")
            payload = build_alignment_readiness(project_root=root, database_path=database, rgb_root=rgb, sparse_root=root / "data/sparse/0", rosbag_path=root / "data/apartment.bag", created_at="fixed")
            self.assertEqual(payload["database"]["image_count"], 2)
            self.assertEqual(payload["local_rgb"]["missing_count"], 1)
            self.assertEqual(payload["alignment"]["readiness"], "BLOCKED_MISSING_SPARSE_OR_ROSBAG_POSES")
            self.assertEqual(validate_alignment_readiness(payload, project_root=root)["status"], "PASS")

    def test_complete_sparse_model_makes_alignment_ready(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "data/database.db"
            self.make_database(database)
            rgb = root / "data/images"
            rgb.mkdir()
            for index in range(2):
                (rgb / f"rgb_{index}.jpg").write_bytes(b"rgb")
            sparse = self.make_sparse(root, ["rgb_0", "rgb_1"])
            payload = build_alignment_readiness(project_root=root, database_path=database, rgb_root=rgb, sparse_root=sparse, rosbag_path=root / "data/apartment.bag", created_at="fixed")
            self.assertEqual(payload["status"], "PASS")
            self.assertEqual(payload["alignment"]["readiness"], "READY_FOR_EVALUATOR_ONLY_SIM3")

    def test_sparse_pose_readiness_does_not_require_every_database_image(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "data/database.db"
            self.make_database(database)
            rgb = root / "data/images"
            rgb.mkdir()
            (rgb / "rgb_0.jpg").write_bytes(b"rgb")
            sparse = self.make_sparse(root, ["rgb_0", "rgb_1"])
            payload = build_alignment_readiness(project_root=root, database_path=database, rgb_root=rgb, sparse_root=sparse, rosbag_path=root / "data/apartment.bag", created_at="fixed")
            self.assertEqual(payload["status"], "PASS")
            self.assertEqual(payload["local_rgb"]["missing_count"], 1)
            self.assertEqual(payload["local_rgb"]["comparison_universe"], "colmap_sparse_registered_images")

    def test_tampered_count_fails_replay(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "data/database.db"
            self.make_database(database)
            rgb = root / "data/images"
            rgb.mkdir()
            payload = build_alignment_readiness(project_root=root, database_path=database, rgb_root=rgb, sparse_root=root / "data/sparse/0", rosbag_path=root / "data/apartment.bag", created_at="fixed")
            payload["database"]["image_count"] = 99
            self.assertEqual(validate_alignment_readiness(payload, project_root=root)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
