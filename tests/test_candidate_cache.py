import argparse
from contextlib import redirect_stdout
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from relground.candidate_cache import (
    CandidateOutcomeCache,
    VisualMemoryManifest,
    build_d11_payloads,
)
from relground.schemas import ObjectObservation, OrientedBoundingBox
from scripts.build_candidate_cache import run as run_builder
from scripts.validate_candidate_cache import validate_output


def _observation(frame_id: str, index: int) -> dict:
    center = np.asarray([index * 0.1, 0.0, 1.0], dtype=float)
    return ObjectObservation(
        obs_id=f"obs-{frame_id}",
        class_text="trash can",
        frame_id=frame_id,
        mask_ref=None,
        retrieval_score=0.9,
        sam_score=0.8,
        valid_point_ratio=0.7,
        points_ref=None,
        center=center,
        obb=OrientedBoundingBox(center=center, extent=np.full(3, 0.2)),
    ).to_dict()


def synthetic_sources(*, complete: bool) -> tuple[dict, dict, dict]:
    frame_ids = ["frame_0001", "frame_0003", "frame_0005"]
    ranking = []
    for index, frame_id in enumerate(frame_ids):
        ranking.append({
            "rank": index + 1,
            "frame_id": frame_id,
            "geometry_index": index,
            "image_path": f"images/{frame_id}.jpg",
            "retrieval_score": 0.9 - index * 0.1,
            "retrieval_cosine": 0.8 - index * 0.1,
            "camera_center": [index * 0.2, 0.0, 0.0],
            "view_direction": [0.0, 0.0, 1.0],
        })
    materialized = frame_ids if complete else frame_ids[:2]
    processed = [{
        "frame_id": frame_id,
        "sam_instances": 1,
        "lifted_instances": 1,
        "rejected_instances": 0,
    } for frame_id in materialized]
    retrieval = {
        "query": "trash can",
        "backend": "facebook/PE-Core-L14-336",
        "retrieval_config": {"strategy": "hybrid"},
        "raw_ranking": ranking,
        "source_commits": {"perception_models": "pe-revision"},
    }
    d6_result = {
        "query": "trash can",
        "sam_threshold": 0.5,
        "lifter_config": {"min_points": 16},
        "mask_resizing_after_sam": True,
        "processed_frames": processed,
        "rejected_instances": [],
        "source_commits": {"sam3": "sam3-revision"},
    }
    observations = {
        "query": "trash can",
        "observations": [
            _observation(frame_id, index)
            for index, frame_id in enumerate(materialized)
        ],
    }
    return retrieval, d6_result, observations


def synthetic_payloads(*, complete: bool) -> tuple[dict, dict]:
    retrieval, d6_result, observations = synthetic_sources(
        complete=complete
    )
    frame_ids = [item["frame_id"] for item in retrieval["raw_ranking"]]
    return build_d11_payloads(
        retrieval=retrieval,
        d6_result=d6_result,
        observations_payload=observations,
        scene_id="synthetic-scene",
        query_id="synthetic-trash-can",
        image_refs={
            frame_id: f"images/{frame_id}.jpg" for frame_id in frame_ids
        },
        image_hashes={frame_id: "a" * 64 for frame_id in frame_ids},
        source_hashes={
            "d5_retrieval_sha256": "b" * 64,
            "d6_result_sha256": "c" * 64,
            "d7_observations_sha256": "d" * 64,
        },
        artifact_refs={
            "visual_memory_manifest": "visual_memory_manifest.json",
            "d5_retrieval": "source_d5_retrieval.json",
            "d6_result": "source_d6_result.json",
            "d7_observations": "source_d7_observations.json",
        },
        created_at="2026-08-27T00:00:00+00:00",
    )


class CandidateCacheTests(unittest.TestCase):
    def test_partial_cache_round_trip_and_conservation(self) -> None:
        visual, cache = synthetic_payloads(complete=False)

        self.assertEqual(
            VisualMemoryManifest.from_dict(visual).to_dict(), visual
        )
        self.assertEqual(
            CandidateOutcomeCache.from_dict(cache).to_dict(), cache
        )
        self.assertEqual(cache["materialization_status"], "partial")
        self.assertEqual(cache["counts"]["candidate_count"], 3)
        self.assertEqual(cache["counts"]["available_candidates"], 2)
        self.assertEqual(cache["counts"]["unmaterialized_candidates"], 1)
        self.assertIsNone(cache["candidates"][2]["outcome"])

    def test_complete_synthetic_cache_is_a_pass_contract(self) -> None:
        visual, cache = synthetic_payloads(complete=True)

        VisualMemoryManifest.from_dict(visual)
        CandidateOutcomeCache.from_dict(cache)
        self.assertEqual(cache["materialization_status"], "complete")
        self.assertEqual(cache["counts"]["available_candidates"], 3)
        self.assertEqual(cache["counts"]["unmaterialized_candidates"], 0)

    def test_duplicate_candidate_frame_is_rejected(self) -> None:
        _, cache = synthetic_payloads(complete=False)
        cache["candidates"][2]["frame_id"] = cache["candidates"][0]["frame_id"]
        cache["candidate_universe"][2] = cache["candidate_universe"][0]

        with self.assertRaisesRegex(ValueError, "frame ids must be unique"):
            CandidateOutcomeCache.from_dict(cache)

    def test_unmaterialized_candidate_cannot_expose_outcome(self) -> None:
        _, cache = synthetic_payloads(complete=False)
        cache["candidates"][2]["outcome"] = copy.deepcopy(
            cache["candidates"][0]["outcome"]
        )

        with self.assertRaisesRegex(ValueError, "cannot expose outcome"):
            CandidateOutcomeCache.from_dict(cache)

    def test_illegal_status_is_rejected(self) -> None:
        _, cache = synthetic_payloads(complete=False)
        cache["candidates"][2]["outcome_status"] = "missing"

        with self.assertRaisesRegex(ValueError, "unsupported candidate"):
            CandidateOutcomeCache.from_dict(cache)

    def test_path_escape_is_rejected(self) -> None:
        visual, cache = synthetic_payloads(complete=True)
        cache["artifacts"]["d5_retrieval"] = "../../secret.json"
        visual["frames"][0]["image_ref"] = "/tmp/image.jpg"

        with self.assertRaisesRegex(ValueError, "contained relative path"):
            CandidateOutcomeCache.from_dict(cache)
        with self.assertRaisesRegex(ValueError, "contained relative path"):
            VisualMemoryManifest.from_dict(visual)

    def test_ground_truth_and_policy_trace_leakage_is_rejected(self) -> None:
        _, cache = synthetic_payloads(complete=True)
        cache["candidates"][0]["outcome"]["observations"][0][
            "ground_truth"
        ] = "instance-a"

        with self.assertRaisesRegex(ValueError, "forbidden evaluation keys"):
            CandidateOutcomeCache.from_dict(cache)

    def _build_bundle(self, root: Path) -> Path:
        retrieval, d6_result, observations = synthetic_sources(
            complete=False
        )
        image_dir = root / "images"
        image_dir.mkdir(parents=True)
        for row in retrieval["raw_ranking"]:
            image_path = image_dir / f"{row['frame_id']}.jpg"
            image_path.write_bytes(str(row["rank"]).encode("ascii"))
            row["image_path"] = str(image_path)
        inputs = root / "inputs"
        inputs.mkdir()
        paths = {}
        for name, payload in (
            ("retrieval", retrieval),
            ("d6", d6_result),
            ("observations", observations),
        ):
            path = inputs / f"{name}.json"
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            paths[name] = path
        output = root / "evidence" / "d11"
        args = argparse.Namespace(
            project_root=str(root),
            retrieval=str(paths["retrieval"]),
            d6_result=str(paths["d6"]),
            observations=str(paths["observations"]),
            output_dir=str(output),
            scene_id="synthetic-scene",
            query_id="synthetic-trash-can",
        )
        with redirect_stdout(io.StringIO()):
            self.assertEqual(run_builder(args), 0)
        return output

    def test_validator_replays_partial_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = self._build_bundle(root)
            report = validate_output(output, project_root=root)

        self.assertEqual(
            report["status"], "PASS_WITH_UNMATERIALIZED_OUTCOMES"
        )
        self.assertEqual(report["available_candidates"], 2)
        self.assertEqual(report["unmaterialized_candidates"], 1)
        self.assertEqual(report["embedding_status"], "not_retained")

    def test_validator_rejects_tampered_retained_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = self._build_bundle(root)
            source = output / "source_d5_retrieval.json"
            source.write_text("{}\n", encoding="utf-8")
            report = validate_output(output, project_root=root)

        self.assertEqual(report["status"], "FAIL")
        self.assertIn("source hash changed", report["failures"][0])


if __name__ == "__main__":
    unittest.main()
