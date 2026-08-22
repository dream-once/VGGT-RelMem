import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from adapters.masks import MaskRecord, save_mask_manifest
from relground.schemas import ObjectObservation, OrientedBoundingBox
from scripts.validate_open_vocab import validate_output


class OpenVocabValidationTests(unittest.TestCase):
    def test_complete_b0_artifact_set_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "masks").mkdir()
            (root / "points").mkdir()
            mask = np.ones((4, 5), dtype=bool)
            np.save(root / "masks" / "o0.npy", mask)
            np.savez_compressed(
                root / "points" / "o0.npz",
                points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            )
            save_mask_manifest(
                root / "masks.json",
                [MaskRecord("o0", "f0", "printer", "masks/o0.npy", 0.5, 0.8)],
            )
            observation = ObjectObservation(
                obs_id="o0",
                class_text="printer",
                frame_id="f0",
                mask_ref="masks/o0.npy",
                retrieval_score=0.5,
                sam_score=0.8,
                valid_point_ratio=1.0,
                points_ref="points/o0.npz",
                center=np.array([0.5, 0.5, 0.0]),
                obb=OrientedBoundingBox(
                    center=np.array([0.5, 0.5, 0.0]),
                    extent=np.array([1.0, 1.0, 0.0]),
                    rotation=np.eye(3),
                ),
            )
            (root / "observations.json").write_text(
                json.dumps({"schema_version": "0.1", "observations": [observation.to_dict()]})
            )
            (root / "b0_result.json").write_text(
                json.dumps({"status": "PASS", "query": "printer", "top1": {"frame_id": "f0"}})
            )
            (root / "preview.png").write_bytes(b"preview")
            (root / "run_manifest.json").write_text("{}")

            result = validate_output(root)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["mask_instances"], 1)
            self.assertEqual(result["lifted_instances"], 1)


if __name__ == "__main__":
    unittest.main()
