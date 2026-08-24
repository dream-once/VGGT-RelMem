import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from relground.observation_cache import (
    SCENE_OBSERVATION_CACHE_VERSION,
    SceneObservationCache,
    file_inventory,
    save_observation_cache,
)
from relground.schemas import (
    OBJECT_OBSERVATION_SCHEMA_VERSION,
    ObjectObservation,
    OrientedBoundingBox,
)
from scripts.validate_d7_cache import validate_output


DYNAMIC_PROBE = {
    "duration_seconds": 40.0,
    "fps": 10.0,
    "frame_count": 400,
    "motion_ratio": 0.80,
    "mean_sample_difference": 3.0,
}


def make_observation(obs_id: str, frame_id: str) -> ObjectObservation:
    return ObjectObservation(
        obs_id=obs_id,
        class_text="trash can",
        frame_id=frame_id,
        mask_ref=f"masks/{obs_id}.npy",
        retrieval_score=0.8,
        sam_score=0.7,
        valid_point_ratio=0.75,
        points_ref=f"points/{obs_id}.npz",
        center=np.array([1.0, 0.5, 0.5]),
        obb=OrientedBoundingBox(
            center=np.array([1.0, 0.5, 0.5]),
            extent=np.array([0.0, 1.0, 1.0]),
            rotation=np.eye(3),
        ),
        metadata={"selected_rank": 1},
    )


def write_valid_fixture(root: Path) -> None:
    observations = [
        make_observation("obs0", "f0"),
        make_observation("obs1", "f2"),
    ]
    cache = SceneObservationCache(
        scene_id="office-loop",
        query="trash can",
        frame_ids=["f0", "f2"],
        observations=observations,
        metadata={"source_d6_result_sha256": "source-hash"},
    )
    save_observation_cache(root / "observations.json", cache)
    references = ["observations.json"]
    for item in observations:
        mask_path = root / str(item.mask_ref)
        point_path = root / str(item.points_ref)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        point_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(mask_path, np.ones((2, 2), dtype=bool))
        np.savez_compressed(
            point_path,
            points=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
        )
        references.extend((str(item.mask_ref), str(item.points_ref)))

    previews = ["previews/f0.png", "previews/f2.png"]
    for reference in previews:
        path = root / reference
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"preview")
    video_reference = "stage_video.mp4"
    (root / video_reference).write_bytes(b"video")
    references.extend((*previews, video_reference))
    manifest = {
        "schema_version": SCENE_OBSERVATION_CACHE_VERSION,
        "status": "PASS",
        "stage": "D7",
        "scene_id": "office-loop",
        "query": "trash can",
        "observation_schema_version": OBJECT_OBSERVATION_SCHEMA_VERSION,
        "source": {
            "stage": "D6",
            "directory": "/ignored/source",
            "d6_result_sha256": "source-hash",
        },
        "frame_ids": ["f0", "f2"],
        "observation_count": 2,
        "frame_observation_counts": {"f0": 1, "f2": 1},
        "previews": previews,
        "stage_video": {
            "path": video_reference,
            "mode": "dynamic_pipeline",
            "duration_seconds": 40.0,
            "fps": 10.0,
            "codec": "mp4v",
            "segments": {
                "input_stream": 10.0,
                "topk_retrieval": 8.0,
                "sam_masks": 10.0,
                "observations_3d": 12.0,
            },
        },
        "files": file_inventory(root, references),
        "created_at": "2026-08-23T00:00:00+00:00",
    }
    (root / "scene_cache.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    (root / "run_manifest.json").write_text("{}", encoding="utf-8")


class D7ValidationTests(unittest.TestCase):
    @patch(
        "scripts.validate_d7_cache.probe_stage_video",
        return_value=DYNAMIC_PROBE,
    )
    def test_complete_self_contained_cache_passes(self, _probe) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_valid_fixture(root)
            result = validate_output(root)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["frames"], 2)
            self.assertEqual(result["observations"], 2)
            self.assertEqual(result["video_duration_seconds"], 40.0)
            self.assertEqual(result["video_motion_ratio"], 0.80)

    @patch(
        "scripts.validate_d7_cache.probe_stage_video",
        return_value=DYNAMIC_PROBE,
    )
    def test_tampered_inventory_hash_fails(self, _probe) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_valid_fixture(root)
            path = root / "scene_cache.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["files"][0]["sha256"] = "0" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = validate_output(root)
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(
                any("SHA-256 mismatch" in error for error in result["errors"])
            )

    @patch(
        "scripts.validate_d7_cache.probe_stage_video",
        return_value={
            **DYNAMIC_PROBE,
            "motion_ratio": 0.05,
            "mean_sample_difference": 0.1,
        },
    )
    def test_static_slideshow_fails(self, _probe) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_valid_fixture(root)
            result = validate_output(root)
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(
                any("too static" in error for error in result["errors"])
            )


if __name__ == "__main__":
    unittest.main()
