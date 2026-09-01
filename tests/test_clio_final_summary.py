import unittest

from relground.clio_final_summary import _scene, validate_summary


class ClioFinalSummaryTests(unittest.TestCase):
    def test_validator_rejects_hidden_a2_regression(self) -> None:
        payload = {
            "schema_version": "0.2", "status": "PASS",
            "scenes": {}, "headline_result": {},
            "claim_boundary": {"a2_pairwise_improvement_on_cubicle": True},
            "sources": {},
        }
        self.assertEqual(validate_summary(payload)["status"], "FAIL")


    def test_validator_rejects_found_it_as_pending_work(self) -> None:
        payload = {
            "schema_version": "0.2", "status": "PASS",
            "scenes": {}, "headline_result": {},
            "claim_boundary": {
                "a2_pairwise_improvement_on_cubicle": False,
                "found_it_comparison_scope": "NOT_RUN",
            },
            "sources": {},
        }
        report = validate_summary(payload)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("outside project scope", report["failures"][0])

    def test_scene_uses_q1f_as_primary_and_keeps_q1_diagnostic(self) -> None:
        def metric(value: float) -> dict[str, float | int]:
            return {
                "task_count": 1,
                "coverage": value,
                "grounding_acc_at_1": value,
                "grounding_acc_at_1_with_alignment_rmse_margin": value,
            }

        grounding = {
            "metrics": {
                "q0_top1": metric(0.0),
                "q1_top5_a2": metric(0.0),
                "q1f_top5_a2_with_q0_fallback": metric(1.0),
                "delta_q1_minus_q0": metric(0.0),
                "delta_q1f_minus_q0": metric(1.0),
            }
        }
        association_metric = {
            "precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1.0
        }
        association = {
            "metrics": {"A1": association_metric, "A2": association_metric},
            "counts": {
                "frozen_tasks": 1,
                "associable_tasks": 1,
                "unknown_background_pairs_excluded": 0,
            },
        }
        relation_metrics = {
            "query_count": 2,
            "positive_count": 1,
            "negative_count": 1,
            "positive_pair_grounding_acc_at_1_strict": 0.0,
            "positive_pair_grounding_acc_at_1_alignment_rmse_padded": 0.0,
            "negative_rejection_accuracy": 1.0,
            "reason_matched_negative_rejection_accuracy": 1.0,
            "relation_aware_negative_rejection_accuracy": 1.0,
            "end_to_end_task_accuracy_alignment_rmse_padded": 0.5,
            "pair_grounded_task_accuracy_alignment_rmse_padded": 0.5,
            "answer_coverage": 0.0,
            "answer_aurc_discrete": None,
            "answerability_proxy_brier": 0.0,
            "answerability_proxy_ece_10": 0.0,
        }
        scene = _scene(
            role="development",
            geometry_frames=1,
            grounding=grounding,
            association=association,
            relation={
                "metrics": relation_metrics,
                "contract": {"calibration_status": "ENGINEERING_DEFAULT_UNCALIBRATED"},
            },
        )
        self.assertEqual(scene["object_grounding"]["primary_policy"].split()[0], "Q1F")
        self.assertEqual(
            scene["object_grounding"]["q1_top5_a2_diagnostic"]["grounding_acc_at_1"],
            0.0,
        )
        self.assertEqual(
            scene["object_grounding"]["q1f_top5_a2_with_q0_fallback"]["grounding_acc_at_1"],
            1.0,
        )

if __name__ == "__main__":
    unittest.main()
