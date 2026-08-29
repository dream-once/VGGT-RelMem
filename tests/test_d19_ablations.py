import argparse
import copy
import json
import unittest
from pathlib import Path

import numpy as np

from relground.ablation_protocol import (
    FAILURE_CATEGORIES,
    a2_variant_config,
    run_a2_ablation,
    run_q2_ablation,
    validate_ablation_manifest,
)
from relground.experiment_protocol import load_json
from relground.schemas import ObjectObservation, OrientedBoundingBox
from scripts.validate_d19 import run as validate_d19


ROOT = Path(__file__).resolve().parents[1]


def observation(obs_id, frame_id, x):
    center = np.asarray([x, 0.0, 1.0], dtype=np.float64)
    return ObjectObservation(
        obs_id=obs_id,
        class_text="trash can",
        frame_id=frame_id,
        mask_ref=None,
        retrieval_score=0.95,
        sam_score=0.95,
        valid_point_ratio=0.95,
        points_ref=None,
        center=center,
        obb=OrientedBoundingBox(
            center=center,
            extent=np.asarray([0.2, 0.2, 0.2]),
        ),
    )


class D19AblationTests(unittest.TestCase):
    def test_retained_bundle_passes(self):
        report = validate_d19(
            argparse.Namespace(
                project_root=str(ROOT),
                manifest="configs/d19_ablation_manifest.json",
                office_evidence="evidence/week3/d19-ablations/office-loop",
                synthetic_evidence="evidence/week3/d19-ablations/synthetic",
                output=None,
            )
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(
            report["real_ablation_status"], "REAL_ABLATION_PENDING"
        )

    def test_removed_weights_are_zero_and_others_renormalize(self):
        for variant, factor in (
            ("without_semantic", "semantic"),
            ("without_obb_shape", "obb_shape"),
            ("without_quality", "quality"),
        ):
            _, payload = a2_variant_config(variant)
            weights = payload["weights"]
            self.assertEqual(weights[factor], 0.0)
            self.assertAlmostEqual(sum(weights.values()), 1.0)
        semantic, _ = a2_variant_config("without_semantic")
        quality, _ = a2_variant_config("without_quality")
        self.assertEqual(semantic.semantic_threshold, 0.0)
        self.assertEqual(quality.min_observation_quality, 0.0)

    def test_removing_complete_link_exposes_bridge_merge(self):
        observations = [
            observation("a", "frame_0001", 0.0),
            observation("b", "frame_0002", 0.1),
            observation("c", "frame_0003", 0.2),
        ]
        base, _ = run_a2_ablation(observations, "base")
        single, _ = run_a2_ablation(
            observations, "without_complete_link"
        )
        self.assertEqual(base["cluster_count"], 2)
        self.assertEqual(single["cluster_count"], 1)
        self.assertGreater(
            single["predicted_match_pairs"],
            base["predicted_match_pairs"],
        )

    def test_no_gain_patience_is_the_only_stop_change(self):
        cache = load_json(
            ROOT
            / "evidence/week2/d14-fixed-topk/synthetic_cache.json"
        )
        changed = copy.deepcopy(cache)
        for index in (1, 2):
            outcome = changed["candidates"][index]["outcome"]
            outcome["sam_instances"] = 0
            outcome["lifted_instances"] = 0
            outcome["observations"] = []
        changed["counts"]["total_observations"] = 4
        base = run_q2_ablation(changed, "base")
        no_patience = run_q2_ablation(
            changed, "no_gain_patience"
        )
        self.assertEqual(base.stop_reason, "two_consecutive_low_gain")
        self.assertGreater(
            len(no_patience.selected_frames),
            len(base.selected_frames),
        )

    def test_manifest_tampering_is_rejected(self):
        manifest = load_json(
            ROOT / "configs/d19_ablation_manifest.json"
        )
        changed = copy.deepcopy(manifest)
        changed["q2_variants"][1]["changed_factor"] = "outcome"
        with self.assertRaisesRegex(ValueError, "one-factor"):
            validate_ablation_manifest(changed)

    def test_failure_taxonomy_denominator_is_complete(self):
        evaluation = load_json(
            ROOT
            / "evidence/week3/d19-ablations/synthetic/evaluation.json"
        )
        audit = evaluation["failure_audit"]
        self.assertEqual(
            tuple(audit["category_counts"]), FAILURE_CATEGORIES
        )
        self.assertEqual(
            audit["failed_query_count"],
            sum(audit["category_counts"].values())
            + audit["uncategorized_failed_queries"],
        )
        self.assertEqual(
            audit["denominator_query_count"],
            len(audit["frozen_query_ids"]),
        )
        self.assertIsNone(evaluation["performance_claim"])


if __name__ == "__main__":
    unittest.main()
