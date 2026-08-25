import json
import tempfile
import unittest
from pathlib import Path

from scripts.cache_scene_observations import resolve_video_input_paths


class D7VideoInputResolutionTests(unittest.TestCase):
    def test_strided_geometry_uses_manifest_frame_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            scene = project_root / "data" / "scene"
            scene.mkdir(parents=True)
            frame_ids = ["frame_0001", "frame_0011", "frame_0071"]
            for frame_id in frame_ids:
                (scene / f"{frame_id}.jpg").write_bytes(b"image")

            decoys = project_root / "data" / "continuous"
            decoys.mkdir(parents=True)
            for index in range(1, 9):
                (decoys / f"frame_{index:04d}.jpg").write_bytes(b"decoy")

            geometry_manifest = project_root / "runs" / "geometry.manifest.json"
            geometry_manifest.parent.mkdir(parents=True)
            geometry_manifest.write_text(
                json.dumps(
                    {
                        "frames": [
                            {
                                "frame_id": frame_id,
                                "image_path": f"data/scene/{frame_id}.jpg",
                            }
                            for frame_id in frame_ids
                        ]
                    }
                ),
                encoding="utf-8",
            )

            d6_dir = project_root / "runs" / "d6"
            d6_dir.mkdir()
            (d6_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "config": {
                            "geometry_manifest": (
                                "runs/geometry.manifest.json"
                            )
                        }
                    }
                ),
                encoding="utf-8",
            )
            selected = [
                {"geometry_index": 0, "frame_id": "frame_0001"},
                {"geometry_index": 2, "frame_id": "frame_0071"},
            ]

            paths, resolved_manifest = resolve_video_input_paths(
                project_root,
                d6_dir,
                decoys,
                selected,
                None,
            )

            self.assertEqual([path.stem for path in paths], frame_ids)
            self.assertEqual(resolved_manifest, geometry_manifest.resolve())

    def test_manifest_image_stem_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            image = project_root / "frame_0011.jpg"
            image.write_bytes(b"image")
            manifest = project_root / "geometry.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "frames": [
                            {
                                "frame_id": "frame_0011",
                                "image_path": str(image),
                            },
                            {
                                "frame_id": "frame_0021",
                                "image_path": str(image),
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "image stem does not match frame_id",
            ):
                resolve_video_input_paths(
                    project_root,
                    project_root,
                    project_root,
                    [{"geometry_index": 0, "frame_id": "frame_0011"}],
                    str(manifest),
                )


if __name__ == "__main__":
    unittest.main()
