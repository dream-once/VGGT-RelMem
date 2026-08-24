import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from relground.observation_cache import (
    SCENE_OBSERVATION_CACHE_FIELDS,
    SCENE_OBSERVATION_CACHE_VERSION,
    SceneObservationCache,
    load_observation_cache,
    save_observation_cache,
)
from relground.schemas import (
    OBJECT_OBSERVATION_FIELDS,
    OBJECT_OBSERVATION_SCHEMA_VERSION,
    ObjectObservation,
    OrientedBoundingBox,
)


def observation(obs_id: str, frame_id: str) -> ObjectObservation:
    return ObjectObservation(
        obs_id=obs_id,
        class_text="trash can",
        frame_id=frame_id,
        mask_ref=f"masks/{obs_id}.npy",
        retrieval_score=0.8,
        sam_score=0.7,
        valid_point_ratio=0.6,
        points_ref=f"points/{obs_id}.npz",
        center=np.array([1.0, 2.0, 3.0]),
        obb=OrientedBoundingBox(
            center=np.array([1.0, 2.0, 3.0]),
            extent=np.array([0.5, 0.6, 0.7]),
            rotation=np.eye(3),
        ),
        metadata={"selected_rank": 1},
    )


class ObservationCacheTests(unittest.TestCase):
    def test_multiframe_round_trip_uses_exact_frozen_fields(self) -> None:
        cache = SceneObservationCache(
            scene_id="office-loop",
            query="trash can",
            frame_ids=["f0", "f2"],
            observations=[
                observation("obs0", "f0"),
                observation("obs1", "f2"),
            ],
            metadata={"source_d6_result_sha256": "abc"},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations.json"
            save_observation_cache(path, cache)
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(tuple(raw), SCENE_OBSERVATION_CACHE_FIELDS)
            self.assertEqual(
                tuple(raw["observations"][0]),
                OBJECT_OBSERVATION_FIELDS,
            )
            self.assertEqual(
                raw["schema_version"],
                SCENE_OBSERVATION_CACHE_VERSION,
            )
            self.assertEqual(
                raw["observations"][0]["schema_version"],
                OBJECT_OBSERVATION_SCHEMA_VERSION,
            )
            loaded = load_observation_cache(path)
            self.assertEqual(loaded.frame_ids, ["f0", "f2"])
            self.assertEqual(
                [item.obs_id for item in loaded.observations],
                ["obs0", "obs1"],
            )
            np.testing.assert_allclose(
                loaded.observations[0].obb.extent,
                [0.5, 0.6, 0.7],
            )

    def test_unknown_observation_field_is_rejected(self) -> None:
        cache = SceneObservationCache(
            scene_id="office-loop",
            query="trash can",
            frame_ids=["f0", "f2"],
            observations=[
                observation("obs0", "f0"),
                observation("obs1", "f2"),
            ],
        )
        payload = cache.to_dict()
        payload["observations"][0]["future_field"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen schema"):
                load_observation_cache(path)

    def test_cache_requires_observations_from_multiple_frames(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two frames"):
            SceneObservationCache(
                scene_id="office-loop",
                query="trash can",
                frame_ids=["f0", "f2"],
                observations=[observation("obs0", "f0")],
            )

    def test_legacy_d6_observation_upgrades_to_frozen_version(self) -> None:
        payload = observation("obs0", "f0").to_dict()
        payload.pop("schema_version")
        loaded = ObjectObservation.from_dict(payload)
        self.assertEqual(
            loaded.schema_version,
            OBJECT_OBSERVATION_SCHEMA_VERSION,
        )


if __name__ == "__main__":
    unittest.main()
