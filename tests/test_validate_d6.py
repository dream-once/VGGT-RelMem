import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from adapters.masks import MaskRecord, save_mask_manifest
from adapters.open_vocab import SAM3_SOURCE_COMMIT, FrameSource
from relground.schemas import ObjectObservation, OrientedBoundingBox
from scripts.run_sam_topk_lifting import load_d5_selection
from scripts.validate_d6 import validate_output


def selection_rows() -> list[dict]:
    return [
        {
            "rank": 1,
            "frame_id": "f0",
            "geometry_index": 0,
            "image_path": "/stale/instance/f0.jpg",
            "submap_id": 0,
            "submap_frame_index": 0,
            "retrieval_score": 0.9,
            "retrieval_cosine": 0.9,
        },
        {
            "rank": 2,
            "frame_id": "f2",
            "geometry_index": 1,
            "image_path": "/stale/instance/f2.jpg",
            "submap_id": 0,
            "submap_frame_index": 2,
            "retrieval_score": 0.8,
            "retrieval_cosine": 0.8,
        },
    ]


def write_valid_fixture(root: Path) -> None:
    rows = selection_rows()
    selection = {
        "schema_version": "0.1",
        "stage": "D5",
        "query": "trash can",
        "requested_k": 3,
        "selected_count": 2,
        "frames": rows,
    }
    (root / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    (root / "run_manifest.json").write_text("{}", encoding="utf-8")

    records: list[MaskRecord] = []
    observations: list[dict] = []
    processed: list[dict] = []
    for rank, frame_id in enumerate(("f0", "f2"), start=1):
        obs_id = f"d6_{frame_id}_000"
        mask_ref = f"masks/{obs_id}.npy"
        points_ref = f"points/{obs_id}.npz"
        (root / "masks").mkdir(exist_ok=True)
        (root / "points").mkdir(exist_ok=True)
        (root / "previews").mkdir(exist_ok=True)
        np.save(root / mask_ref, np.ones((2, 2), dtype=bool))
        points = np.array(
            [
                [rank, 0.0, 0.0],
                [rank, 1.0, 0.0],
                [rank, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        np.savez_compressed(root / points_ref, points=points)
        (root / "previews" / f"{frame_id}.png").write_bytes(b"preview")
        records.append(
            MaskRecord(
                obs_id=obs_id,
                frame_id=frame_id,
                class_text="trash can",
                mask_ref=mask_ref,
                retrieval_score=1.0 - rank / 10.0,
                sam_score=0.75,
            )
        )
        observation = ObjectObservation(
            obs_id=obs_id,
            class_text="trash can",
            frame_id=frame_id,
            mask_ref=mask_ref,
            retrieval_score=1.0 - rank / 10.0,
            sam_score=0.75,
            valid_point_ratio=0.75,
            points_ref=points_ref,
            center=np.array([rank, 0.5, 0.5]),
            obb=OrientedBoundingBox(
                center=np.array([rank, 0.5, 0.5]),
                extent=np.array([0.0, 1.0, 1.0]),
                rotation=np.eye(3),
            ),
            metadata={"selected_rank": rank},
        )
        observations.append(observation.to_dict())
        processed.append(
            {
                "rank": rank,
                "frame_id": frame_id,
                "geometry_index": rank - 1,
                "retrieval_score": 1.0 - rank / 10.0,
                "sam_instances": 1,
                "lifted_instances": 1,
                "rejected_instances": 0,
                "preview": f"previews/{frame_id}.png",
            }
        )

    save_mask_manifest(root / "masks.json", records)
    (root / "observations.json").write_text(
        json.dumps({"schema_version": "0.1", "observations": observations}),
        encoding="utf-8",
    )
    result = {
        "schema_version": "0.1",
        "status": "PASS",
        "stage": "D6",
        "query": "trash can",
        "requested_k": 3,
        "selected_frames": rows,
        "processed_frames": processed,
        "sam_instances": 2,
        "lifted_instances": 2,
        "rejected_instances": [],
        "frames_with_masks": ["f0", "f2"],
        "frames_with_lifted_observations": ["f0", "f2"],
        "lifter_config": {
            "confidence_threshold": 0.5,
            "min_points": 3,
            "outlier_mad_scale": 3.5,
        },
        "source_commits": {"sam3": SAM3_SOURCE_COMMIT},
    }
    (root / "d6_result.json").write_text(json.dumps(result), encoding="utf-8")


class D6ValidationTests(unittest.TestCase):
    def test_complete_multiframe_artifact_set_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_valid_fixture(root)
            result = validate_output(root)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["selected_frames"], ["f0", "f2"])
            self.assertEqual(result["lifted_instances"], 2)

    def test_inconsistent_per_frame_count_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_valid_fixture(root)
            result_path = root / "d6_result.json"
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            payload["processed_frames"][0]["sam_instances"] = 2
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            result = validate_output(root)
            self.assertEqual(result["status"], "FAIL")
            self.assertIn("SAM count mismatch for f0", result["errors"])

    def test_distinct_retrieval_and_segmentation_queries_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_valid_fixture(root)
            selection_path = root / "selection.json"
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            selection["query"] = "bring me a pillow"
            selection_path.write_text(json.dumps(selection), encoding="utf-8")

            result_path = root / "d6_result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["query"] = "pillow"
            result["retrieval_query"] = "bring me a pillow"
            result_path.write_text(json.dumps(result), encoding="utf-8")

            masks_path = root / "masks.json"
            masks = json.loads(masks_path.read_text(encoding="utf-8"))
            for row in masks["records"]:
                row["class_text"] = "pillow"
            masks_path.write_text(json.dumps(masks), encoding="utf-8")

            observations_path = root / "observations.json"
            observations = json.loads(observations_path.read_text(encoding="utf-8"))
            for row in observations["observations"]:
                row["class_text"] = "pillow"
            observations_path.write_text(json.dumps(observations), encoding="utf-8")

            validation = validate_output(root)
            self.assertEqual(validation["status"], "PASS")

    def test_selection_resolves_images_from_current_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current_paths = []
            sources = []
            for index, frame_id in enumerate(("f0", "f2")):
                image_path = root / f"{frame_id}.jpg"
                image_path.write_bytes(b"image")
                current_paths.append(str(image_path.resolve()))
                sources.append(
                    FrameSource(
                        frame_id=frame_id,
                        image_path=image_path.resolve(),
                        geometry_index=index,
                        submap_id=0,
                        submap_frame_index=index * 2,
                    )
                )
            selection = {
                "stage": "D5",
                "query": "trash can",
                "requested_k": 3,
                "selected_count": 2,
                "frames": selection_rows(),
            }
            path = root / "topk.json"
            path.write_text(json.dumps(selection), encoding="utf-8")
            _payload, rows = load_d5_selection(path, sources)
            self.assertEqual(
                [row["image_path"] for row in rows],
                current_paths,
            )


if __name__ == "__main__":
    unittest.main()
