import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

from relground.retrieval import FrameCandidate, RetrievalConfig, TopKFrameRetriever
from scripts.validate_topk_retrieval import validate_output


def record(candidate: FrameCandidate, rank: int) -> dict:
    return {
        "rank": rank,
        "frame_id": candidate.frame_id,
        "geometry_index": candidate.index,
        "image_path": f"/tmp/{candidate.frame_id}.jpg",
        "submap_id": 0,
        "submap_frame_index": candidate.index,
        "retrieval_score": candidate.score,
        "retrieval_cosine": candidate.metadata["cosine"],
        "camera_center": candidate.camera_center.tolist(),
        "view_direction": candidate.view_direction.tolist(),
    }


def write_valid_fixture(root: Path) -> None:
    candidates = [
        FrameCandidate(
            "f0",
            0.9,
            index=0,
            camera_center=np.array([0.0, 0.0, 0.0]),
            view_direction=np.array([0.0, 0.0, 1.0]),
            metadata={"cosine": 0.9},
        ),
        FrameCandidate(
            "f1",
            0.8,
            index=1,
            camera_center=np.array([0.1, 0.0, 0.0]),
            view_direction=np.array([0.0, 0.0, 1.0]),
            metadata={"cosine": 0.8},
        ),
        FrameCandidate(
            "f3",
            0.7,
            index=3,
            camera_center=np.array([0.3, 0.0, 0.0]),
            view_direction=np.array([0.0, 0.0, 1.0]),
            metadata={"cosine": 0.7},
        ),
        FrameCandidate(
            "f5",
            0.6,
            index=5,
            camera_center=np.array([0.5, 0.0, 0.0]),
            view_direction=np.array([0.0, 0.0, 1.0]),
            metadata={"cosine": 0.6},
        ),
    ]
    raw = TopKFrameRetriever(
        RetrievalConfig(top_k=len(candidates), redundancy="none")
    ).retrieve(candidates)
    settings = {
        "redundancy": "temporal",
        "min_frame_gap": 2,
        "min_camera_distance": 0.15,
        "min_view_angle_deg": 12.0,
    }
    selections = {}
    artifacts = {"preview": "topk_preview.png"}
    for k in (1, 3, 5):
        config = RetrievalConfig(top_k=k, **settings)
        selected = TopKFrameRetriever(config).retrieve(candidates)
        name = f"topk_{k}.json"
        payload = {
            "schema_version": "0.1",
            "stage": "D5",
            "query": "trash can",
            "requested_k": k,
            "selected_count": len(selected),
            "exhausted_nonredundant_candidates": len(selected)
            < min(k, len(candidates)),
            "retrieval_config": asdict(config),
            "frames": [
                record(candidate, rank)
                for rank, candidate in enumerate(selected, start=1)
            ],
        }
        (root / name).write_text(json.dumps(payload))
        selections[str(k)] = {
            "artifact": name,
            "selected_count": len(selected),
            "frame_ids": [candidate.frame_id for candidate in selected],
        }
        artifacts[f"topk_{k}"] = name

    result = {
        "schema_version": "0.1",
        "status": "PASS",
        "stage": "D5",
        "query": "trash can",
        "searched_frames": len(raw),
        "retrieval_config": settings,
        "k_values": [1, 3, 5],
        "upstream_top1": {
            "frame_id": raw[0].frame_id,
            "geometry_index": raw[0].index,
            "retrieval_score": raw[0].score,
            "retrieval_cosine": raw[0].metadata["cosine"],
        },
        "top1_compatible": True,
        "prefix_consistent": True,
        "raw_ranking": [
            record(candidate, rank)
            for rank, candidate in enumerate(raw, start=1)
        ],
        "selections": selections,
        "artifacts": artifacts,
    }
    (root / "retrieval.json").write_text(json.dumps(result))
    (root / "run_manifest.json").write_text("{}")
    (root / "topk_preview.png").write_bytes(b"preview")


class TopKValidationTests(unittest.TestCase):
    def test_complete_d5_artifact_set_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_valid_fixture(root)
            result = validate_output(root)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["selected_counts"], {"1": 1, "3": 3, "5": 3})

    def test_inconsistent_topk_artifact_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_valid_fixture(root)
            path = root / "topk_3.json"
            payload = json.loads(path.read_text())
            payload["frames"].reverse()
            path.write_text(json.dumps(payload))
            result = validate_output(root)
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(
                any("top-3 frame ids" in error for error in result["errors"])
            )


if __name__ == "__main__":
    unittest.main()
