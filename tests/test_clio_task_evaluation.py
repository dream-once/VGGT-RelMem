import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import yaml

from relground.clio_task_evaluation import (
    build_clio_task_evaluation,
    point_in_obb,
    validate_clio_task_evaluation,
)


class ClioTaskEvaluationTests(unittest.TestCase):
    def write_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        memory_path = root / "prediction/object_memory.json"
        memory_path.parent.mkdir(parents=True)
        memory_path.write_text(json.dumps({
            "metadata": {"query": "target"},
            "pending_observations": [{"obs_id": "pending"}],
            "objects": [{
                "object_id": "obj_0001",
                "confidence": 0.9,
                "observations": [{"frame_id": "frame_1"}, {"frame_id": "frame_2"}],
                "fused_center": [1.0, 0.0, 0.0],
                "fused_obb": {
                    "center": [1.0, 0.0, 0.0],
                    "extent": [1.0, 1.0, 1.0],
                    "rotation": np.eye(3).tolist(),
                },
            }],
        }))
        alignment_path = root / "runs/alignment.json"
        alignment_path.parent.mkdir(parents=True)
        alignment_path.write_text(json.dumps({
            "status": "PASS",
            "scene_id": "apartment",
            "contract": {"use": "evaluator_only", "main_inference_may_read_alignment": False},
            "sim3": {
                "scale": 2.0,
                "rotation": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                "translation": [10.0, 20.0, 30.0],
            },
            "error_m": {"rmse": 0.1, "median": 0.05, "max": 0.2, "threshold_rmse": 0.15},
        }))
        gt_path = root / "data/tasks.yaml"
        gt_path.parent.mkdir(parents=True)
        gt_path.write_text(yaml.safe_dump({
            "target": [{
                "center": [10.0, 22.0, 30.0],
                "extents": [1.0, 1.0, 1.0],
                "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            }]
        }))
        return memory_path, alignment_path, gt_path

    def test_world_transform_and_top1_metric_replay(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            memory_path, alignment_path, gt_path = self.write_fixture(root)
            payload = build_clio_task_evaluation(
                project_root=root,
                object_memory_path=memory_path,
                world_alignment_path=alignment_path,
                task_yaml_path=gt_path,
                task_query="target",
                created_at="fixed",
            )
            self.assertEqual(payload["metrics"]["center_grounding_acc_at_1"], 1.0)
            self.assertEqual(payload["objects"][0]["center_world_m"], [10.0, 22.0, 30.0])
            self.assertEqual(
                validate_clio_task_evaluation(payload, project_root=root)["status"],
                "PASS",
            )
            payload["metrics"]["center_grounding_acc_at_1"] = 0.0
            self.assertEqual(
                validate_clio_task_evaluation(payload, project_root=root)["status"],
                "FAIL",
            )

    def test_oriented_box_containment_uses_full_extents(self) -> None:
        rotation = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        self.assertTrue(point_in_obb(
            [0.0, 0.9, 0.0], center=[0, 0, 0], extent=[2, 1, 1], rotation=rotation,
        ))
        self.assertFalse(point_in_obb(
            [0.9, 0.0, 0.0], center=[0, 0, 0], extent=[2, 1, 1], rotation=rotation,
        ))

    def test_task_text_may_differ_from_segmentation_query(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            memory_path, alignment_path, gt_path = self.write_fixture(root)
            gt = yaml.safe_load(gt_path.read_text())
            gt["clean target"] = gt.pop("target")
            gt_path.write_text(yaml.safe_dump(gt))
            payload = build_clio_task_evaluation(
                project_root=root,
                object_memory_path=memory_path,
                world_alignment_path=alignment_path,
                task_yaml_path=gt_path,
                task_query="clean target",
                created_at="fixed",
            )
            self.assertEqual(payload["task_query"], "clean target")
            self.assertEqual(payload["segmentation_query"], "target")

    def test_rejects_prediction_visible_alignment(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            memory_path, alignment_path, gt_path = self.write_fixture(root)
            alignment = json.loads(alignment_path.read_text())
            alignment["contract"]["main_inference_may_read_alignment"] = True
            alignment_path.write_text(json.dumps(alignment))
            with self.assertRaisesRegex(ValueError, "not evaluator-only"):
                build_clio_task_evaluation(
                    project_root=root,
                    object_memory_path=memory_path,
                    world_alignment_path=alignment_path,
                    task_yaml_path=gt_path,
                    task_query="target",
                )


if __name__ == "__main__":
    unittest.main()
