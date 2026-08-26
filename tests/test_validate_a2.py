import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from relground.association import ObjectMemory
from relground.d9_association import ManualInstanceGroup, ManualInstanceLabels
from scripts.evaluate_a2_association import run as run_evaluation
from scripts.run_a2_association import (
    build_parser as build_prediction_parser,
    run as run_prediction,
)
from scripts.validate_a2_association import validate_output as validate_prediction
from scripts.validate_a2_evaluation import validate_output as validate_evaluation
from tests.test_a2_association import make_observation


class ValidateA2Tests(unittest.TestCase):
    def build_prediction(self, root: Path) -> tuple[Path, list[str]]:
        input_dir = root / "input"
        input_dir.mkdir(parents=True)
        observations = [
            make_observation("a", "f1", [0.0, 0.0, 0.0]),
            make_observation("b", "f2", [0.05, 0.0, 0.0]),
            make_observation("c", "f3", [1.0, 0.0, 0.0]),
        ]
        memory = ObjectMemory(
            metadata={"scene_id": "synthetic-scene", "query": "trash can"}
        )
        memory.stage_many(observations)
        memory_path = input_dir / "object_memory.json"
        memory.save(memory_path)
        output = root / "d12" / "prediction"
        args = argparse.Namespace(
            project_root=str(Path.cwd()),
            memory=str(memory_path),
            output_dir=str(output),
            semantic_threshold=0.70,
            min_observation_quality=0.25,
            center_distance_threshold=0.15,
            min_overlap_iou=0.0,
            min_distinct_frames=2,
        )
        with redirect_stdout(io.StringIO()):
            self.assertEqual(run_prediction(args), 0)
        return output, [item.obs_id for item in observations]

    def write_labels(
        self,
        root: Path,
        *,
        mismatched: bool = False,
    ) -> Path:
        groups = (
            (
                ManualInstanceGroup("instance-a", ("a", "c")),
                ManualInstanceGroup("instance-b", ("b",)),
            )
            if mismatched
            else (
                ManualInstanceGroup("instance-a", ("a", "b")),
                ManualInstanceGroup("instance-b", ("c",)),
            )
        )
        labels = ManualInstanceLabels(
            scene_id="synthetic-scene",
            query="trash can",
            annotation_method="synthetic unit-test labels",
            notes=("evaluation only",),
            instance_groups=groups,
        )
        path = root / ("bad_labels.json" if mismatched else "labels.json")
        path.write_text(
            json.dumps(labels.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def build_evaluation(
        self,
        root: Path,
        *,
        mismatched: bool = False,
    ) -> tuple[Path, Path]:
        prediction, _ = self.build_prediction(root)
        labels = self.write_labels(root, mismatched=mismatched)
        evaluation = root / "d12" / "evaluation"
        args = argparse.Namespace(
            project_root=str(Path.cwd()),
            prediction_dir=str(prediction),
            labels=str(labels),
            output_dir=str(evaluation),
        )
        with redirect_stdout(io.StringIO()):
            self.assertEqual(run_evaluation(args), 0)
        return prediction, evaluation

    def test_prediction_cli_has_no_labels_or_metric_threshold(self) -> None:
        options = {
            option
            for action in build_prediction_parser()._actions
            for option in action.option_strings
        }
        self.assertNotIn("--labels", options)
        self.assertNotIn("--min-pairwise-f1", options)

    def test_prediction_and_evaluation_validators_recompute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prediction, evaluation = self.build_evaluation(Path(directory))
            prediction_report = validate_prediction(prediction)
            evaluation_report = validate_evaluation(evaluation)

        self.assertEqual(prediction_report["status"], "PASS")
        self.assertEqual(prediction_report["pair_count"], 3)
        self.assertEqual(prediction_report["permanent_objects"], 1)
        self.assertTrue(prediction_report["complete_link_pass"])
        self.assertEqual(evaluation_report["status"], "PASS")
        self.assertEqual(evaluation_report["f1"], 1.0)
        self.assertEqual(evaluation_report["failure_case_count"], 0)

    def test_prediction_validator_rejects_tampered_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prediction, _ = self.build_prediction(Path(directory))
            result_path = prediction / "a2_result.json"
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            payload["pairs"][0]["pair_score"] = 0.123456
            result_path.write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
            report = validate_prediction(prediction)

        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any(
            "pairs differ from recompute" in item
            for item in report["failures"]
        ))

    def test_prediction_validator_rejects_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prediction, _ = self.build_prediction(Path(directory))
            result_path = prediction / "a2_result.json"
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            payload["source"]["d8_memory"] = "../../escape.json"
            result_path.write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
            report = validate_prediction(prediction)

        self.assertEqual(report["status"], "FAIL")
        self.assertIn("escapes", report["failures"][0])

    def test_evaluation_records_poor_result_without_tuning_or_failing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prediction, evaluation = self.build_evaluation(
                Path(directory), mismatched=True
            )
            prediction_hash_before = hashlib.sha256(
                (prediction / "a2_result.json").read_bytes()
            ).hexdigest()
            report = validate_evaluation(evaluation)
            prediction_hash_after = hashlib.sha256(
                (prediction / "a2_result.json").read_bytes()
            ).hexdigest()

        self.assertEqual(report["status"], "PASS")
        self.assertLess(report["f1"], 1.0)
        self.assertGreater(report["failure_case_count"], 0)
        self.assertEqual(prediction_hash_before, prediction_hash_after)
        self.assertTrue(report["thresholds_frozen_before_evaluation"])

    def test_evaluation_validator_rejects_tampered_metric(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, evaluation = self.build_evaluation(Path(directory))
            result_path = evaluation / "a2_evaluation.json"
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            payload["metrics"]["f1"] = 0.5
            result_path.write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
            report = validate_evaluation(evaluation)

        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any(
            "metrics differ" in item for item in report["failures"]
        ))


if __name__ == "__main__":
    unittest.main()
