import argparse
from pathlib import Path
import unittest

from scripts.validate_d21 import run as validate_d21


ROOT = Path(__file__).resolve().parents[1]


class D21ReleaseRegressionTests(unittest.TestCase):
    def test_retained_release_bundle_rebuilds_and_passes(self) -> None:
        report = validate_d21(
            argparse.Namespace(
                project_root=str(ROOT),
                manifest="configs/d21_result_card_manifest.json",
                retained_root="evidence/week3/d21-final",
                output=None,
            )
        )
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(all(report["checks"].values()))


if __name__ == "__main__":
    unittest.main()
