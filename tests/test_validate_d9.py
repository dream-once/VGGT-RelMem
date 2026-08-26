import argparse
from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from relground.association import ObjectMemory
from relground.d9_association import (
    ManualInstanceGroup,
    ManualInstanceLabels,
)
from relground.schemas import ObjectObservation, OrientedBoundingBox
from scripts.evaluate_d9_association import run as run_evaluation
from scripts.run_d9_association import (
    build_parser as build_prediction_parser,
    run as run_prediction,
)
from scripts.validate_d9_association import (
    validate_output as validate_prediction,
)
from scripts.validate_d9_evaluation import (
    validate_output as validate_evaluation,
)


def make_observation(
    obs_id: str,
    frame_id: str,
    center: list[float],
) -> ObjectObservation:
    center_array = np.asarray(center, dtype=float)
    return ObjectObservation(
        obs_id=obs_id,
        class_text="trash can",
        frame_id=frame_id,
        mask_ref=f"masks/{obs_id}.npy",
        retrieval_score=0.8,
        sam_score=0.9,
        valid_point_ratio=0.95,
        points_ref=f"points/{obs_id}.npz",
        center=center_array,
        obb=OrientedBoundingBox(
            center=center_array,
            extent=np.full(3, 0.2),
        ),
    )


class ValidateD9Tests(unittest.TestCase):
    def build_prediction(
        self,
        root: Path,
        *,
        no_matches: bool = False,
    ) -> tuple[Path, Path]:
        runs = root / "runs"
        d8_dir = runs / "d8"
        d8_dir.mkdir(parents=True)
        if no_matches:
            observations = [
                make_observation(
                    "a", "frame_0001", [0.0, 0.0, 0.0]
                ),
                make_observation(
                    "b", "frame_0002", [10.0, 0.0, 0.0]
                ),
            ]
        else:
            observations = [
                make_observation(
                    "a", "frame_0001", [0.0, 0.0, 0.0]
                ),
                make_observation(
                    "b", "frame_0002", [0.05, 0.0, 0.0]
                ),
                make_observation(
                    "c", "frame_0001", [1.0, 0.0, 0.0]
                ),
                make_observation(
                    "d", "frame_0001", [1.04, 0.0, 0.0]
                ),
            ]
        memory = ObjectMemory(
            metadata={"scene_id": "scene", "query": "trash can"}
        )
        memory.stage_many(observations)
        memory_path = d8_dir / "object_memory.json"
        memory.save(memory_path)

        labels = ManualInstanceLabels(
            scene_id="scene",
            query="trash can",
            annotation_method="unit-test manual labels",
            notes=("synthetic",),
            instance_groups=(
                (
                    ManualInstanceGroup(
                        "instance_a",
                        ("a",),
                    )
                    if no_matches
                    else ManualInstanceGroup(
                        "instance_a",
                        ("a", "b"),
                    )
                ),
                (
                    ManualInstanceGroup(
                        "instance_b",
                        ("b",),
                    )
                    if no_matches
                    else ManualInstanceGroup(
                        "instance_b",
                        ("c", "d"),
                    )
                ),
            ),
        )
        labels_path = root / "labels.json"
        labels_path.write_text(
            json.dumps(labels.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

        prediction_dir = runs / "d9" / "prediction"
        args = argparse.Namespace(
            project_root=str(Path.cwd()),
            memory=str(memory_path),
            output_dir=str(prediction_dir),
            center_distance_threshold=0.15,
            min_overlap_iou=0.0,
            min_distinct_frames=2,
        )
        with redirect_stdout(io.StringIO()):
            self.assertEqual(run_prediction(args), 0)
        return prediction_dir, labels_path

    def build_evaluation(
        self,
        root: Path,
    ) -> tuple[Path, Path]:
        prediction_dir, labels_path = self.build_prediction(root)
        evaluation_dir = root / "runs" / "d9" / "evaluation"
        args = argparse.Namespace(
            project_root=str(Path.cwd()),
            prediction_dir=str(prediction_dir),
            labels=str(labels_path),
            output_dir=str(evaluation_dir),
            min_pairwise_f1=0.95,
        )
        with redirect_stdout(io.StringIO()):
            self.assertEqual(run_evaluation(args), 0)
        return prediction_dir, evaluation_dir

    def test_prediction_cli_has_no_labels_argument(self) -> None:
        option_strings = {
            option
            for action in build_prediction_parser()._actions
            for option in action.option_strings
        }
        self.assertNotIn("--labels", option_strings)
        self.assertNotIn("--min-pairwise-f1", option_strings)

    def test_prediction_validator_recomputes_passing_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir, _ = self.build_prediction(Path(directory))
            report = validate_prediction(output_dir)
            result = json.loads(
                (output_dir / "d9_result.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["pair_count"], 6)
        self.assertEqual(report["permanent_objects"], 1)
        self.assertEqual(report["pending_observations"], 2)
        self.assertTrue(report["observation_conservation"])
        self.assertTrue(report["deterministic_recompute"])
        self.assertTrue(report["round_trip_equal"])
        self.assertNotIn("metrics", result)
        self.assertNotIn("failure_cases", result)
        self.assertNotIn("pair_labels", json.dumps(result))

    def test_prediction_validator_rejects_tampered_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir, _ = self.build_prediction(Path(directory))
            result_path = output_dir / "d9_result.json"
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            payload["pairs"][0]["center_distance"] = 999.0
            result_path.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
            report = validate_prediction(output_dir)

        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any(
            "saved pairs differs" in failure
            for failure in report["failures"]
        ))

    def test_prediction_validator_rejects_absolute_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir, _ = self.build_prediction(Path(directory))
            result_path = output_dir / "d9_result.json"
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            payload["source"]["d8_memory"] = "/tmp/escape.json"
            result_path.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
            report = validate_prediction(output_dir)

        self.assertEqual(report["status"], "FAIL")
        self.assertIn(
            "D9 source paths must be relative",
            report["failures"][0],
        )

    def test_prediction_validator_rejects_escaping_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir, _ = self.build_prediction(Path(directory))
            result_path = output_dir / "d9_result.json"
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            payload["source"]["d8_memory"] = "../../escape.json"
            result_path.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
            report = validate_prediction(output_dir)

        self.assertEqual(report["status"], "FAIL")
        self.assertIn(
            "escapes the artifact bundle",
            report["failures"][0],
        )

    def test_zero_match_prediction_is_a_legal_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir, _ = self.build_prediction(
                Path(directory),
                no_matches=True,
            )
            report = validate_prediction(output_dir)

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["predicted_match_pairs"], 0)
        self.assertEqual(report["permanent_objects"], 0)
        self.assertEqual(report["pending_observations"], 2)
        self.assertEqual(report["association_decisions"], 0)

    def test_evaluation_validator_recomputes_passing_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, evaluation_dir = self.build_evaluation(Path(directory))
            report = validate_evaluation(evaluation_dir)

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["prediction_status"], "PASS")
        self.assertEqual(report["pair_count"], 6)
        self.assertEqual(report["pairwise_f1"], 1.0)
        self.assertEqual(report["failure_case_count"], 0)

    def test_evaluation_validator_rejects_tampered_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, evaluation_dir = self.build_evaluation(Path(directory))
            result_path = evaluation_dir / "d9_evaluation.json"
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            payload["pairs"][0]["expected_same"] = not (
                payload["pairs"][0]["expected_same"]
            )
            result_path.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
            report = validate_evaluation(evaluation_dir)

        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any(
            "saved pairs differs" in failure
            for failure in report["failures"]
        ))

    def test_evaluation_validator_rejects_escaping_prediction_path(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, evaluation_dir = self.build_evaluation(Path(directory))
            result_path = evaluation_dir / "d9_evaluation.json"
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            payload["source"]["prediction_result"] = (
                "../../../escape.json"
            )
            result_path.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
            report = validate_evaluation(evaluation_dir)

        self.assertEqual(report["status"], "FAIL")
        self.assertIn(
            "escapes the artifact bundle",
            report["failures"][0],
        )


if __name__ == "__main__":
    unittest.main()
