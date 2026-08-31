import unittest

from scripts.compare_segmentation_prompt_sweep import is_valid_diagnostic_d6


class SegmentationSweepComparisonTests(unittest.TestCase):
    def test_passing_d6_is_valid(self):
        self.assertTrue(is_valid_diagnostic_d6(
            {"status": "PASS", "errors": []}, {"status": "PASS"}
        ))

    def test_zero_mask_diagnostic_is_valid_but_extra_error_fails(self):
        result = {
            "status": "INSUFFICIENT_MULTIFRAME_3D_EVIDENCE",
            "sam_instances": 0,
            "lifted_instances": 0,
        }
        report = {
            "status": "FAIL",
            "errors": [
                "D6 result status is 'INSUFFICIENT_MULTIFRAME_3D_EVIDENCE'",
                "fewer than two frames produced valid 3D observations",
            ],
            "mask_instances": 0,
            "lifted_instances": 0,
            "frames_with_lifted_observations": [],
        }
        self.assertTrue(is_valid_diagnostic_d6(report, result))
        report["errors"].append("mask hash mismatch")
        self.assertFalse(is_valid_diagnostic_d6(report, result))


if __name__ == "__main__":
    unittest.main()
