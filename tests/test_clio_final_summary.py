import unittest

from relground.clio_final_summary import validate_summary


class ClioFinalSummaryTests(unittest.TestCase):
    def test_validator_rejects_hidden_a2_regression(self) -> None:
        payload = {
            "schema_version": "0.1", "status": "PASS",
            "scenes": {}, "headline_result": {},
            "claim_boundary": {"a2_pairwise_improvement_on_cubicle": True},
            "sources": {},
        }
        self.assertEqual(validate_summary(payload)["status"], "FAIL")


    def test_validator_rejects_found_it_as_pending_work(self) -> None:
        payload = {
            "schema_version": "0.1", "status": "PASS",
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

if __name__ == "__main__":
    unittest.main()
