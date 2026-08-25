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
from scripts.run_d9_association import run
from scripts.validate_d9_association import validate_output


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
    def build_bundle(self, root: Path) -> Path:
        runs = root / "runs"
        d8_dir = runs / "d8"
        d8_dir.mkdir(parents=True)
        observations = [
            make_observation("a", "frame_0001", [0.0, 0.0, 0.0]),
            make_observation("b", "frame_0002", [0.05, 0.0, 0.0]),
            make_observation("c", "frame_0001", [1.0, 0.0, 0.0]),
            make_observation("d", "frame_0001", [1.04, 0.0, 0.0]),
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
                ManualInstanceGroup("instance_a", ("a", "b")),
                ManualInstanceGroup("instance_b", ("c", "d")),
            ),
        )
        labels_path = root / "labels.json"
        labels_path.write_text(
            json.dumps(labels.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        output_dir = runs / "d9"
        args = argparse.Namespace(
            project_root=str(Path.cwd()),
            memory=str(memory_path),
            labels=str(labels_path),
            output_dir=str(output_dir),
            center_distance_threshold=0.15,
            min_overlap_iou=0.0,
            min_distinct_frames=2,
            min_pairwise_f1=0.95,
        )
        with redirect_stdout(io.StringIO()):
            self.assertEqual(run(args), 0)
        return output_dir

    def test_validator_recomputes_passing_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = self.build_bundle(Path(directory))
            report = validate_output(output_dir)

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["pair_count"], 6)
        self.assertEqual(report["pairwise_f1"], 1.0)
        self.assertEqual(report["permanent_objects"], 1)
        self.assertEqual(report["pending_observations"], 2)
        self.assertTrue(report["round_trip_equal"])

    def test_validator_rejects_tampered_pair_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = self.build_bundle(Path(directory))
            result_path = output_dir / "d9_result.json"
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            payload["pairs"][0]["center_distance"] = 999.0
            result_path.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
            report = validate_output(output_dir)

        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any(
            "pairs differs" in failure
            for failure in report["failures"]
        ))

    def test_validator_rejects_absolute_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = self.build_bundle(Path(directory))
            result_path = output_dir / "d9_result.json"
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            payload["source"]["d8_memory"] = "/tmp/escape.json"
            result_path.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
            report = validate_output(output_dir)

        self.assertEqual(report["status"], "FAIL")
        self.assertIn(
            "D9 source paths must be relative",
            report["failures"][0],
        )


if __name__ == "__main__":
    unittest.main()
