import argparse
import copy
import json
import unittest
from pathlib import Path

from relground.experiment_protocol import (
    Q0_ID,
    Q2_ID,
    load_json,
    replay_query_policy,
    run_experiment_prediction,
    sha256_file,
    validate_experiment_manifest,
)
from scripts.validate_d18 import run as validate_d18


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "d18_experiment_manifest.json"


def fixture():
    manifest = load_json(MANIFEST_PATH)
    office_source, synthetic_source, clio_source = manifest["sources"]
    office = load_json(ROOT / office_source["cache_ref"])
    synthetic = load_json(ROOT / synthetic_source["cache_ref"])
    clio = load_json(ROOT / clio_source["cache_ref"])
    return manifest, office, synthetic, clio


class D18ExperimentTests(unittest.TestCase):
    def test_retained_bundle_passes(self):
        report = validate_d18(
            argparse.Namespace(
                project_root=str(ROOT),
                manifest="configs/d18_experiment_manifest.json",
                office_evidence="evidence/week3/d18-qxa/office-loop",
                synthetic_evidence="evidence/week3/d18-qxa/synthetic",
                clio_evidence="evidence/week4/clio-apartment-gpu/d18-qxa",
                output=None,
            )
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(
            report["held_out_status"], "CLIO_HELD_OUT_PENDING"
        )

    def test_office_q2_completes_from_public_complete_cache(self):
        _, office, _, _ = fixture()
        replay = replay_query_policy(office, Q2_ID)
        self.assertEqual(office["materialization_status"], "complete")
        self.assertEqual(replay.status, "PASS")
        self.assertEqual(len(replay.selected_frames), 5)
        self.assertEqual(replay.selected_frames[-1], "frame_0031")
        self.assertTrue(
            all(
                status == "available"
                for status in replay.selected_outcome_statuses
            )
        )
        self.assertEqual(replay.stop_reason, "max_budget_reached")

    def test_clio_q2_completes_from_public_complete_cache(self):
        _, _, _, clio = fixture()
        replay = replay_query_policy(clio, Q2_ID)
        self.assertEqual(clio["materialization_status"], "complete")
        self.assertEqual(replay.status, "PASS")
        self.assertGreaterEqual(len(replay.selected_frames), 3)

    def test_missing_outcome_still_blocks_without_skipping(self):
        partial = load_json(
            ROOT
            / "evidence/week2/d11-candidate-cache/candidate_cache.json"
        )
        replay = replay_query_policy(partial, Q2_ID)
        self.assertEqual(replay.status, "BLOCKED_MISSING_OUTCOME")
        self.assertEqual(replay.selected_frames[-1], "frame_0061")
        self.assertEqual(
            replay.selected_outcome_statuses[-1], "unmaterialized"
        )
        self.assertNotIn("frame_0021", replay.selected_frames[3:])

    def test_clio_zero_observation_q0_is_a_valid_replay(self):
        manifest, _, _, clio = fixture()
        prediction = run_experiment_prediction(
            manifest,
            clio,
            manifest_ref="configs/d18_experiment_manifest.json",
            manifest_sha256=sha256_file(MANIFEST_PATH),
            source_id="clio-apartment-development",
            created_at="test",
        )
        q0_rows = [
            row for row in prediction["matrix_rows"]
            if row["q_policy"] == Q0_ID
        ]
        self.assertTrue(q0_rows)
        self.assertTrue(
            all(row["association"]["input_observations"] == 0 for row in q0_rows)
        )

    def test_prediction_is_label_and_metric_free(self):
        manifest, _, synthetic, _ = fixture()
        prediction = run_experiment_prediction(
            manifest,
            synthetic,
            manifest_ref="configs/d18_experiment_manifest.json",
            manifest_sha256=sha256_file(MANIFEST_PATH),
            source_id="synthetic-correctness",
            created_at="test",
        )
        serialized = json.dumps(prediction, sort_keys=True).lower()
        for forbidden in (
            '"labels"',
            '"ground_truth"',
            '"expected_same"',
            '"metrics"',
            '"answer"',
        ):
            self.assertNotIn(forbidden, serialized)

    def test_q0_ignores_future_candidate_outcome(self):
        _, _, synthetic, _ = fixture()
        baseline = replay_query_policy(synthetic, Q0_ID)
        changed = copy.deepcopy(synthetic)
        changed["candidates"][-1]["outcome"]["observations"][0][
            "retrieval_score"
        ] = 0.123
        altered = replay_query_policy(changed, Q0_ID)
        self.assertEqual(
            baseline.selected_frames, altered.selected_frames
        )
        self.assertEqual(
            [item.to_dict() for item in baseline.observations],
            [item.to_dict() for item in altered.observations],
        )

    def test_manifest_path_escape_and_hash_tampering_fail(self):
        manifest, _, _, _ = fixture()
        escaped = copy.deepcopy(manifest)
        escaped["sources"][0]["cache_ref"] = "../cache.json"
        with self.assertRaisesRegex(ValueError, "repository-relative"):
            validate_experiment_manifest(escaped)
        tampered = copy.deepcopy(manifest)
        tampered["sources"][0]["cache_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_experiment_manifest(
                tampered, project_root=ROOT
            )

    def test_synthetic_labels_only_exist_in_evaluation(self):
        prediction = load_json(
            ROOT
            / "evidence/week3/d18-qxa/synthetic/prediction.json"
        )
        evaluation = load_json(
            ROOT
            / "evidence/week3/d18-qxa/synthetic/evaluation.json"
        )
        self.assertNotIn("metrics", json.dumps(prediction))
        self.assertTrue(
            all(row["metrics"] is not None for row in evaluation["rows"])
        )
        self.assertIsNone(evaluation["performance_claim"])


if __name__ == "__main__":
    unittest.main()
