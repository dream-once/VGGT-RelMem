import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from adapters.masks import MaskRecord, save_mask_manifest
from relground.observation_cache import sha256_file
from relground.schemas import ObjectObservation, OrientedBoundingBox
from relground.single_view import (
    BASELINE_SCHEMA_VERSION,
    B0_OFFICIAL,
    B1_ROBUST_SINGLE_VIEW,
    VGGTImageTransform,
)
from scripts.validate_single_view_baselines import validate_output


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class ControlledSingleViewValidationTests(unittest.TestCase):
    def make_output(self, root: Path) -> None:
        (root / "masks").mkdir()
        for baseline_dir in (
            "b0_official",
            "b1_robust_single_view",
        ):
            (root / baseline_dir / "points").mkdir(parents=True)

        mask = np.ones((4, 5), dtype=bool)
        mask_ref = "masks/sam_f0_000.npy"
        np.save(root / mask_ref, mask)
        save_mask_manifest(
            root / "masks.json",
            [
                MaskRecord(
                    "sam_f0_000",
                    "f0",
                    "chair",
                    mask_ref,
                    0.5,
                    0.8,
                )
            ],
        )
        transform = VGGTImageTransform(
            source_size=(5, 4),
            resized_size=(5, 4),
            crop_xyxy=(0, 0, 5, 4),
            padding_ltrb=(0, 0, 0, 0),
            output_size=(5, 4),
            target_size=5,
            patch_size=1,
        )
        Image.new("RGB", transform.output_size, "white").save(
            root / "sam_input.png"
        )
        write_json(
            root / "preprocess.json",
            {
                "schema_version": transform.schema_version,
                "frame_id": "f0",
                "source_image": "source.png",
                "transform": transform.to_dict(),
            },
        )

        points = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            dtype=np.float32,
        )
        baselines = {}
        for baseline_id, baseline_dir, prefix in (
            (B0_OFFICIAL, "b0_official", "b0"),
            (
                B1_ROBUST_SINGLE_VIEW,
                "b1_robust_single_view",
                "b1",
            ),
        ):
            points_ref = (
                f"{baseline_dir}/points/{prefix}_f0_000.npz"
            )
            np.savez_compressed(root / points_ref, points=points)
            observation = ObjectObservation(
                obs_id=f"{prefix}_f0_000",
                class_text="chair",
                frame_id="f0",
                mask_ref=mask_ref,
                retrieval_score=0.5,
                sam_score=0.8,
                valid_point_ratio=1.0,
                points_ref=points_ref,
                center=np.array([0.5, 0.5, 0.0]),
                obb=OrientedBoundingBox(
                    center=np.array([0.5, 0.5, 0.0]),
                    extent=np.array([1.0, 1.0, 0.0]),
                    rotation=np.eye(3),
                ),
                metadata={
                    "baseline_id": baseline_id,
                    "shared_instance_id": "sam_f0_000",
                },
            )
            observations_ref = f"{baseline_dir}/observations.json"
            write_json(
                root / observations_ref,
                {
                    "schema_version": BASELINE_SCHEMA_VERSION,
                    "baseline_id": baseline_id,
                    "observations": [observation.to_dict()],
                },
            )
            baselines[baseline_id] = {
                "method": "test",
                "observations": observations_ref,
                "lifted_instances": 1,
                "rejected_instances": [],
            }

        (root / "preview.png").write_bytes(b"preview")
        write_json(
            root / "run_manifest.json",
            {
                "config": {
                    "pipeline": "controlled single-view B0/B1",
                    "query": "chair",
                }
            },
        )
        write_json(
            root / "single_view_result.json",
            {
                "schema_version": BASELINE_SCHEMA_VERSION,
                "status": "PASS",
                "stage": "D4-controlled-correction",
                "query": "chair",
                "top1": {
                    "frame_id": "f0",
                    "retrieval_score": 0.5,
                },
                "controlled_inputs": {
                    "sam_input": "sam_input.png",
                    "sam_input_sha256": sha256_file(
                        root / "sam_input.png"
                    ),
                    "preprocess": "preprocess.json",
                    "preprocess_sha256": sha256_file(
                        root / "preprocess.json"
                    ),
                    "mask_manifest": "masks.json",
                    "sam_mask_shape": [4, 5],
                    "sam_instances": 1,
                    "mask_resizing_after_sam": False,
                },
                "baselines": baselines,
            },
        )

    def test_complete_controlled_pair_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_output(root)
            result = validate_output(root)
            self.assertEqual(result["status"], "PASS", result["errors"])
            self.assertEqual(result["shared_lifted_instances"], 1)
            self.assertEqual(
                result["lifted_instances"],
                {
                    B0_OFFICIAL: 1,
                    B1_ROBUST_SINGLE_VIEW: 1,
                },
            )

    def test_nonfinite_point_cloud_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_output(root)
            np.savez_compressed(
                root
                / "b0_official"
                / "points"
                / "b0_f0_000.npz",
                points=np.array(
                    [[0, 0, 0], [1, 0, 0], [np.nan, 1, 0]],
                    dtype=np.float32,
                ),
            )
            result = validate_output(root)
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(
                any("non-finite" in error for error in result["errors"])
            )

    def test_query_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_output(root)
            path = root / "b1_robust_single_view" / "observations.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["observations"][0]["class_text"] = "table"
            write_json(path, payload)
            result = validate_output(root)
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(
                any("query mismatch" in error for error in result["errors"])
            )


if __name__ == "__main__":
    unittest.main()
