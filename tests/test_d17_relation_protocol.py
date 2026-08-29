import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from relground.association import ObjectMemory
from relground.relation_protocol import (
    evaluate_relation_prediction,
    run_relation_prediction,
    validate_calibration_manifest,
    validate_query_bundle,
)
from relground.schemas import ObjectObservation, OrientedBoundingBox
from scripts.validate_d17 import validate


ROOT = Path(__file__).resolve().parents[1]


def observation(obs_id, class_text, center):
    center = np.asarray(center, dtype=np.float64)
    return ObjectObservation(
        obs_id=obs_id,
        class_text=class_text,
        frame_id=f"frame_{obs_id}",
        mask_ref=None,
        retrieval_score=0.95,
        sam_score=0.95,
        valid_point_ratio=0.95,
        points_ref=None,
        center=center,
        obb=OrientedBoundingBox(center, np.array([0.2, 0.2, 0.2])),
    )


def fixture():
    memory = ObjectMemory()
    decisions = memory.add_many([
        observation("left", "chair", [-1, 0, 0]),
        observation("right", "chair", [1, 0, 0]),
        observation("desk", "desk", [0, 0, 0]),
    ])
    queries = {
        "schema_version": "0.1",
        "scene_id": "synthetic-relations",
        "split_role": "synthetic",
        "queries": [
            {
                "query_id": "left",
                "target": "chair",
                "relation": "left_of",
                "reference": "desk",
                "anchor_frame": "anchor",
            },
            {
                "query_id": "right",
                "target": "chair",
                "relation": "right_of",
                "reference": "desk",
                "anchor_frame": "anchor",
            },
            {
                "query_id": "missing-target",
                "target": "lamp",
                "relation": "left_of",
                "reference": "desk",
                "anchor_frame": "anchor",
            },
            {
                "query_id": "missing-reference",
                "target": "chair",
                "relation": "left_of",
                "reference": "cabinet",
                "anchor_frame": "anchor",
            },
            {
                "query_id": "missing-anchor",
                "target": "chair",
                "relation": "left_of",
                "reference": "desk",
                "anchor_frame": None,
            },
        ],
    }
    labels = {
        "schema_version": "0.1",
        "scene_id": "synthetic-relations",
        "split_role": "synthetic",
        "labels": [
            {
                "query_id": "left",
                "answerable": True,
                "answer_object_id": decisions[0].object_id,
                "expected_abstain_reason": None,
            },
            {
                "query_id": "right",
                "answerable": True,
                "answer_object_id": decisions[1].object_id,
                "expected_abstain_reason": None,
            },
            {
                "query_id": "missing-target",
                "answerable": False,
                "answer_object_id": None,
                "expected_abstain_reason": "target_not_found",
            },
            {
                "query_id": "missing-reference",
                "answerable": False,
                "answer_object_id": None,
                "expected_abstain_reason": "reference_not_found",
            },
            {
                "query_id": "missing-anchor",
                "answerable": False,
                "answer_object_id": None,
                "expected_abstain_reason": "missing_anchor_frame",
            },
        ],
    }
    calibration = json.loads(
        (ROOT / "configs" / "relation_calibration_manifest.json").read_text()
    )
    source = {
        "object_memory": "object_memory.json",
        "object_memory_sha256": "a" * 64,
        "anchor_poses": "anchor_poses.json",
        "anchor_poses_sha256": "b" * 64,
        "queries": "queries.json",
        "queries_sha256": "c" * 64,
        "calibration": "calibration_manifest.json",
        "calibration_sha256": "d" * 64,
    }
    return memory, queries, labels, calibration, source


class RelationProtocolTests(unittest.TestCase):
    def test_prediction_is_label_free_and_negative_rejection_is_correct(self):
        memory, queries, labels, calibration, source = fixture()
        prediction = run_relation_prediction(
            memory,
            {"anchor": np.eye(4)},
            queries,
            calibration,
            source=source,
            created_at="test",
        )
        serialized = json.dumps(prediction, sort_keys=True).lower()
        for forbidden in (
            '"answerable"', '"answer_object_id"',
            '"expected_abstain_reason"', '"metrics"',
        ):
            self.assertNotIn(forbidden, serialized)
        evaluation = evaluate_relation_prediction(
            prediction,
            labels,
            source={
                "prediction": "prediction.json",
                "prediction_sha256": "e" * 64,
                "labels": "labels.json",
                "labels_sha256": "f" * 64,
            },
            created_at="test",
        )
        self.assertEqual(evaluation["metrics"]["task_accuracy"], 1.0)
        self.assertEqual(
            evaluation["metrics"]["negative_rejection_accuracy"], 1.0
        )
        self.assertEqual(evaluation["metrics"]["positive_count"], 2)
        self.assertEqual(evaluation["metrics"]["negative_count"], 3)

    def test_formal_query_rejects_embedded_answer(self):
        _, queries, _, _, _ = fixture()
        queries["queries"][0]["answer_object_id"] = "obj_0001"
        with self.assertRaisesRegex(ValueError, "fields are not frozen"):
            validate_query_bundle(queries)

    def test_calibration_cannot_claim_held_out_fit(self):
        _, _, _, calibration, _ = fixture()
        calibration["fitted_on_split_role"] = "held-out"
        calibration["fit_sample_ids"] = ["test-1"]
        with self.assertRaisesRegex(ValueError, "uncalibrated"):
            validate_calibration_manifest(
                calibration, execution_split_role="held-out"
            )

    def test_anchor_rotation_swaps_left_right_ranking(self):
        memory, queries, _, calibration, source = fixture()
        identity = run_relation_prediction(
            memory, {"anchor": np.eye(4)}, queries, calibration,
            source=source, created_at="test",
        )
        rotated_pose = np.eye(4)
        rotated_pose[:3, :3] = np.diag([-1.0, 1.0, -1.0])
        rotated = run_relation_prediction(
            memory, {"anchor": rotated_pose}, queries, calibration,
            source=source, created_at="test",
        )
        self.assertNotEqual(
            identity["results"][0]["ranked_ids"][0],
            rotated["results"][0]["ranked_ids"][0],
        )

    def test_source_path_escape_is_rejected(self):
        memory, queries, _, calibration, source = fixture()
        source["queries"] = "../queries.json"
        with self.assertRaisesRegex(ValueError, "safe relative"):
            run_relation_prediction(
                memory, {"anchor": np.eye(4)}, queries, calibration,
                source=source, created_at="test",
            )

    def test_bundle_tampering_fails_validator(self):
        source = ROOT / "evidence" / "week3" / "d17-relations"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for path in source.iterdir():
                if path.is_file():
                    (target / path.name).write_bytes(path.read_bytes())
            prediction = json.loads((target / "prediction.json").read_text())
            prediction["results"][0]["ranked_ids"].reverse()
            (target / "prediction.json").write_text(json.dumps(prediction))
            self.assertEqual(validate(target)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
