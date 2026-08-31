import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from relground.segmentation_audit import (
    build_segmentation_inventory,
    build_visibility_template,
    evaluate_visibility,
    validate_segmentation_inventory,
)


class SegmentationAuditTests(unittest.TestCase):
    def make_bundle(self, root: Path) -> tuple[Path, Path]:
        run = root / "runs/test-d6"
        (run / "sam_inputs").mkdir(parents=True)
        (run / "previews").mkdir()
        for frame_id in ("f1", "f2"):
            Image.new("RGB", (32, 24), (40, 50, 60)).save(run / f"sam_inputs/{frame_id}.png")
            Image.new("RGB", (32, 24), (60, 50, 40)).save(run / f"previews/{frame_id}.png")
        result = {
            "query": "pillow",
            "sam_threshold": 0.5,
            "processed_frames": [
                {"rank": 1, "frame_id": "f1", "retrieval_score": 0.8, "sam_instances": 1, "lifted_instances": 1, "rejected_instances": 0, "sam_input": "sam_inputs/f1.png", "preview": "previews/f1.png"},
                {"rank": 2, "frame_id": "f2", "retrieval_score": 0.7, "sam_instances": 0, "lifted_instances": 0, "rejected_instances": 0, "sam_input": "sam_inputs/f2.png", "preview": "previews/f2.png"},
            ],
            "sam_instances": 1,
            "lifted_instances": 1,
            "rejected_instances": [],
        }
        observations = {"schema_version": "0.1", "observations": [{"obs_id": "o1", "frame_id": "f1"}]}
        result_path = run / "d6_result.json"
        observations_path = run / "observations.json"
        result_path.write_text(json.dumps(result))
        observations_path.write_text(json.dumps(observations))
        return result_path, observations_path

    def test_inventory_is_label_free_and_replays(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            result, observations = self.make_bundle(root)
            inventory = build_segmentation_inventory(project_root=root, d6_result_path=result, observations_path=observations, scene_id="s", created_at="fixed")
            self.assertEqual(inventory["counts"]["frames_without_masks"], 1)
            self.assertTrue(all("visibility" not in row for row in inventory["frames"]))
            self.assertEqual(validate_segmentation_inventory(inventory, project_root=root)["status"], "PASS")

    def test_pending_labels_do_not_create_recall(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            result, observations = self.make_bundle(root)
            inventory = build_segmentation_inventory(project_root=root, d6_result_path=result, observations_path=observations, scene_id="s", created_at="fixed")
            labels = build_visibility_template(inventory)
            evaluation = evaluate_visibility(inventory, labels)
            self.assertEqual(evaluation["status"], "PENDING_FRAME_VISIBILITY_LABELS")
            self.assertIsNone(evaluation["metrics"])

    def test_complete_visibility_labels_compute_false_negative(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            result, observations = self.make_bundle(root)
            inventory = build_segmentation_inventory(project_root=root, d6_result_path=result, observations_path=observations, scene_id="s", created_at="fixed")
            labels = build_visibility_template(inventory)
            for row in labels["frames"]:
                row["visibility"] = "VISIBLE"
            evaluation = evaluate_visibility(inventory, labels)
            self.assertEqual(evaluation["status"], "PASS")
            self.assertEqual(evaluation["metrics"]["true_positive"], 1)
            self.assertEqual(evaluation["metrics"]["false_negative"], 1)
            self.assertEqual(evaluation["metrics"]["recall"], 0.5)

    def test_source_tampering_fails(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            result, observations = self.make_bundle(root)
            inventory = build_segmentation_inventory(project_root=root, d6_result_path=result, observations_path=observations, scene_id="s", created_at="fixed")
            result.write_text(result.read_text() + "\n")
            self.assertEqual(validate_segmentation_inventory(inventory, project_root=root)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
