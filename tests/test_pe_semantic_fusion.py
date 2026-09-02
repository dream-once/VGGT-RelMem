import copy
import json
from pathlib import Path
import unittest

import numpy as np

from relground.pe_semantic_fusion import (
    aggregate_task_results,
    build_crop_variants,
    expanded_mask_bounds,
    mean_crop_query_score,
    paired_transitions,
    select_medoid,
    select_quality_representative,
    select_semantic_representative,
)
from relground.pe_semantic_fusion_summary import validate_summary


ROOT = Path(__file__).resolve().parents[1]


class PeSemanticFusionTests(unittest.TestCase):
    def test_expanded_bounds_are_clamped(self) -> None:
        mask = np.zeros((8, 10), dtype=bool)
        mask[0:2, 1:4] = True
        self.assertEqual(
            expanded_mask_bounds(
                mask,
                padding_fraction=0.5,
                min_padding_pixels=1,
            ),
            (0, 0, 6, 4),
        )

    def test_crop_variants_keep_foreground_and_neutralize_background(
        self,
    ) -> None:
        image = np.arange(6 * 7 * 3, dtype=np.uint8).reshape(6, 7, 3)
        mask = np.zeros((6, 7), dtype=bool)
        mask[2:4, 3:5] = True
        context, masked, bounds = build_crop_variants(
            image,
            mask,
            padding_fraction=0.0,
            min_padding_pixels=1,
            background_value=127,
        )
        self.assertEqual(bounds, (2, 1, 6, 5))
        self.assertTrue(np.array_equal(context[1:3, 1:3], image[2:4, 3:5]))
        self.assertTrue(np.all(masked[0, 0] == 127))
        self.assertTrue(np.array_equal(masked[1:3, 1:3], image[2:4, 3:5]))

    def test_crop_score_is_mean_of_two_cosines(self) -> None:
        self.assertAlmostEqual(
            mean_crop_query_score(
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ),
            0.5,
        )

    def test_semantic_selection_is_deterministic(self) -> None:
        rows = [
            {"observation_id": "obs_b", "semantic_score": 0.3},
            {"observation_id": "obs_a", "semantic_score": 0.3},
        ]
        self.assertEqual(
            select_semantic_representative(rows)["observation_id"],
            "obs_a",
        )

    def test_quality_selection_uses_frozen_quality(self) -> None:
        rows = [
            {"observation_id": "obs_a", "quality": 0.2},
            {"observation_id": "obs_b", "quality": 0.4},
        ]
        self.assertEqual(
            select_quality_representative(rows)["observation_id"],
            "obs_b",
        )

    def test_medoid_selects_central_observation(self) -> None:
        rows = [
            {"observation_id": "left", "center_vggt": [0.0, 0.0, 0.0]},
            {"observation_id": "middle", "center_vggt": [1.0, 0.0, 0.0]},
            {"observation_id": "right", "center_vggt": [3.0, 0.0, 0.0]},
        ]
        self.assertEqual(select_medoid(rows)["observation_id"], "middle")

    def test_task_metrics_and_transitions_recompute(self) -> None:
        rows = [
            {
                "base": {
                    "answered": True,
                    "correct": False,
                    "correct_with_alignment_rmse_margin": True,
                },
                "variant": {
                    "answered": True,
                    "correct": True,
                    "correct_with_alignment_rmse_margin": True,
                },
            },
            {
                "base": {
                    "answered": False,
                    "correct": False,
                    "correct_with_alignment_rmse_margin": False,
                },
                "variant": {
                    "answered": False,
                    "correct": False,
                    "correct_with_alignment_rmse_margin": False,
                },
            },
        ]
        metric = aggregate_task_results(rows, "variant")
        self.assertEqual(metric["grounding_acc_at_1"], 0.5)
        self.assertEqual(
            paired_transitions(rows, "base", "variant"),
            {
                "both_wrong": 1,
                "regressions": 0,
                "wins": 1,
                "both_correct": 0,
            },
        )

    def test_tracked_summary_passes_and_rejects_heldout_overclaim(
        self,
    ) -> None:
        path = (
            ROOT
            / "evidence/post-d21-pe-fusion/benchmark_summary.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            validate_summary(payload, project_root=ROOT)["status"],
            "PASS",
        )
        changed = copy.deepcopy(payload)
        changed["claim_boundary"]["untouched_held_out_claim"] = True
        self.assertEqual(
            validate_summary(changed, project_root=ROOT)["status"],
            "FAIL",
        )


if __name__ == "__main__":
    unittest.main()
