from contextlib import nullcontext
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from adapters.open_vocab import (
    Sam3Backend,
    load_frame_sources,
    prepare_sam_outputs,
    resize_mask_nearest,
    score_frames,
    select_top1,
)


class OpenVocabAdapterTests(unittest.TestCase):
    def test_sam_backend_uses_cuda_autocast(self) -> None:
        fake_torch = SimpleNamespace(
            bfloat16="bf16",
            float16="fp16",
            cuda=SimpleNamespace(is_bf16_supported=lambda: True),
            inference_mode=Mock(return_value=nullcontext()),
            autocast=Mock(return_value=nullcontext()),
        )
        processor = Mock()
        processor.set_image.return_value = {"image": "state"}
        processor.set_text_prompt.return_value = {
            "masks": np.ones((1, 2, 3), dtype=bool),
            "boxes": np.array([[0.0, 0.0, 3.0, 2.0]]),
            "scores": np.array([0.9]),
        }
        backend = object.__new__(Sam3Backend)
        backend.device = "cuda"
        backend.torch = fake_torch
        backend.processor = processor

        batch = backend.segment(SimpleNamespace(height=2, width=3), " printer ")

        self.assertEqual(batch.masks.shape, (1, 2, 3))
        fake_torch.autocast.assert_called_once_with(
            device_type="cuda",
            dtype="bf16",
            enabled=True,
        )
        processor.set_image.assert_called_once()
        processor.set_text_prompt.assert_called_once_with(prompt="printer", state={"image": "state"})

    def test_top1_matches_non_negative_cosine_baseline(self) -> None:
        match = select_top1(
            ["f0", "f1", "f2"],
            np.array([[1.0, 0.0], [0.5, 0.5], [-1.0, 0.0]]),
            np.array([[1.0, 0.0]]),
        )
        self.assertEqual(match.frame_id, "f0")
        self.assertAlmostEqual(match.score, 1.0)
        self.assertAlmostEqual(match.cosine, 1.0)

    def test_all_frame_scores_preserve_geometry_order(self) -> None:
        frame_ids = ["z_first", "a_second", "m_third"]
        images = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
        text = np.array([1.0, 0.0])
        scored = score_frames(frame_ids, images, text)
        self.assertEqual([item.frame_id for item in scored], frame_ids)
        self.assertEqual([item.index for item in scored], [0, 1, 2])
        self.assertEqual([item.score for item in scored], [1.0, 1.0, 0.0])
        self.assertEqual(select_top1(frame_ids, images, text).frame_id, "z_first")

    def test_top1_clips_all_negative_scores_like_upstream(self) -> None:
        match = select_top1(
            ["first", "second"],
            np.array([[-1.0, 0.0], [-0.5, -0.5]]),
            np.array([1.0, 0.0]),
        )
        self.assertEqual(match.frame_id, "first")
        self.assertEqual(match.score, 0.0)
        self.assertLess(match.cosine, 0.0)

    def test_sam_output_contract_and_mask_resize(self) -> None:
        batch = prepare_sam_outputs(
            np.array([[[[True, False], [False, True]]]]),
            np.array([[0.0, 0.0, 2.0, 2.0]]),
            np.array([0.75]),
            image_shape=(2, 2),
        )
        self.assertEqual(batch.masks.shape, (1, 2, 2))
        resized = resize_mask_nearest(batch.masks[0], (4, 4))
        self.assertEqual(resized.shape, (4, 4))
        self.assertEqual(int(resized.sum()), 8)

    def test_manifest_sources_follow_geometry_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "second.jpg").touch()
            (root / "first.jpg").touch()
            manifest = root / "geometry.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "frames": [
                            {
                                "frame_id": "f0",
                                "image_path": "first.jpg",
                                "submap_id": 0,
                                "submap_frame_index": 0,
                            },
                            {
                                "frame_id": "f1",
                                "image_path": "second.jpg",
                                "submap_id": 0,
                                "submap_frame_index": 1,
                            },
                        ]
                    }
                )
            )
            sources = load_frame_sources(
                manifest,
                ["f1", "f0"],
                project_root=root,
            )
            self.assertEqual([item.frame_id for item in sources], ["f1", "f0"])
            self.assertEqual(sources[0].image_path.name, "second.jpg")


if __name__ == "__main__":
    unittest.main()
