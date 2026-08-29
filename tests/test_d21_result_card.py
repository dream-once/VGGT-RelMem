import argparse
import copy
from pathlib import Path
import shutil
import tempfile
import unittest

from relground.result_card import (
    FINAL_CONCLUSION,
    PROJECT_POSITIONING,
    audit_readme_claims,
    build_result_card,
    load_json,
    sha256_file,
    validate_result_card_manifest,
)
from scripts.validate_d21 import run as validate_d21


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/d21_result_card_manifest.json"


class D21ResultCardTests(unittest.TestCase):
    def test_retained_bundle_rebuilds_and_passes(self):
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

    def test_unqualified_sota_claim_fails(self):
        audit = audit_readme_claims("Our method is SOTA.")
        self.assertEqual(audit["status"], "FAIL")
        self.assertEqual(audit["review_required_count"], 1)
        qualified = audit_readme_claims("当前结果不支持 SOTA 结论。")
        self.assertEqual(qualified["status"], "PASS")

    def test_manifest_path_escape_and_hash_tampering_fail(self):
        manifest = load_json(MANIFEST)
        escaped = copy.deepcopy(manifest)
        escaped["inputs"][0]["path"] = "../README.md"
        with self.assertRaisesRegex(ValueError, "repository-relative"):
            validate_result_card_manifest(escaped)
        tampered = copy.deepcopy(manifest)
        tampered["inputs"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_result_card_manifest(
                tampered, project_root=ROOT
            )

    def test_clean_snapshot_builds_after_move(self):
        manifest = load_json(MANIFEST)
        expected = build_result_card(ROOT, manifest)
        with tempfile.TemporaryDirectory() as directory:
            moved = Path(directory) / "clean-tree"
            target_manifest = (
                moved / "configs/d21_result_card_manifest.json"
            )
            target_manifest.parent.mkdir(parents=True)
            shutil.copy2(MANIFEST, target_manifest)
            for item in manifest["inputs"]:
                source = ROOT / item["path"]
                target = moved / item["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            restored = load_json(target_manifest)
            validate_result_card_manifest(restored, project_root=moved)
            self.assertEqual(expected, build_result_card(moved, restored))

    def test_every_result_has_tracked_sources_samples_and_budget(self):
        card = build_result_card(ROOT, load_json(MANIFEST))
        self.assertEqual(card["source_status"], "CPU_COMPLETE")
        self.assertEqual(len(card["results"]), 6)
        for row in card["results"]:
            self.assertTrue(row["sample_size"])
            self.assertTrue(row["budget"])
            self.assertEqual(row["validation_status"], "PASS")
            for source in (row["evidence"], row["config"]):
                path = ROOT / source["path"]
                self.assertTrue(path.is_file())
                self.assertEqual(sha256_file(path), source["sha256"])

    def test_readme_has_final_positioning_and_no_pending_numbers(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(PROJECT_POSITIONING, text)
        self.assertIn(FINAL_CONCLUSION, text)
        audit = audit_readme_claims(text)
        self.assertEqual(audit["status"], "PASS")
        card = build_result_card(ROOT, load_json(MANIFEST))
        self.assertFalse(
            card["claim_boundary"]["held_out_performance"]
        )
        self.assertFalse(
            card["claim_boundary"]["sota_or_superiority_claim"]
        )
        self.assertIsNone(
            card["claim_boundary"]["performance_improvement"]
        )


if __name__ == "__main__":
    unittest.main()
