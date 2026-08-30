import argparse
import copy
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from relground.reproduction_package import (
    build_result_tables,
    load_json,
    validate_reproduction_manifest,
)
from scripts.validate_d20 import run as validate_d20
from scripts.visualize_scene_memory import _require_input_path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/d20_reproduction_manifest.json"


class D20ReproductionTests(unittest.TestCase):
    def test_retained_bundle_rebuilds_and_passes(self):
        report = validate_d20(
            argparse.Namespace(
                project_root=str(ROOT),
                manifest="configs/d20_reproduction_manifest.json",
                retained_root="evidence/week3/d20-reproduction",
                output=None,
            )
        )
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(all(report["checks"].values()))

    def test_minimal_tracked_snapshot_rebuilds_after_move(self):
        manifest = load_json(MANIFEST)
        expected = build_result_tables(ROOT, manifest)
        with tempfile.TemporaryDirectory() as directory:
            moved = Path(directory) / "clean-tree"
            manifest_target = (
                moved / "configs/d20_reproduction_manifest.json"
            )
            manifest_target.parent.mkdir(parents=True)
            shutil.copy2(MANIFEST, manifest_target)
            for item in manifest["inputs"]:
                source = ROOT / item["path"]
                target = moved / item["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            restored_manifest = load_json(manifest_target)
            validate_reproduction_manifest(
                restored_manifest, project_root=moved
            )
            actual = build_result_tables(moved, restored_manifest)
            self.assertEqual(expected, actual)

    def test_path_escape_and_hash_tampering_fail_closed(self):
        manifest = load_json(MANIFEST)
        escaped = copy.deepcopy(manifest)
        escaped["inputs"][0]["path"] = "../artifact.json"
        with self.assertRaisesRegex(ValueError, "repository-relative"):
            validate_reproduction_manifest(escaped)
        tampered = copy.deepcopy(manifest)
        tampered["inputs"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_reproduction_manifest(
                tampered, project_root=ROOT
            )

    def test_viewer_error_explains_arbitrary_working_directory(self):
        original = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                with self.assertRaises(FileNotFoundError) as caught:
                    _require_input_path(
                        Path("runs/missing/geometry.npz"),
                        label="geometry",
                    )
                message = str(caught.exception)
                self.assertIn(
                    "Relative input paths are resolved from the current "
                    "working directory",
                    message,
                )
                self.assertIn(
                    f"cd {ROOT}",
                    message,
                )
                self.assertIn("absolute paths", message)
        finally:
            os.chdir(original)

    def test_office_development_table_uses_complete_cache(self):
        results = load_json(
            ROOT / "evidence/week3/d20-reproduction/result_tables.json"
        )
        rows = results["qxa_development"]["rows"]
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["status"] == "PASS" for row in rows))
        q2 = [row for row in rows if row["combination_id"].startswith("Q2")]
        self.assertEqual([row["selected_count"] for row in q2], [5, 5])
        self.assertTrue(
            all(row["stop_reason"] == "max_budget_reached" for row in q2)
        )

    def test_clio_development_table_is_unlabelled_and_complete(self):
        results = build_result_tables(
            ROOT, load_json(MANIFEST)
        )
        rows = results["qxa_clio_development"]["rows"]
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["status"] == "PASS" for row in rows))
        self.assertIsNone(results["clio_ablations"]["performance_claim"])

    def test_readme_clean_clone_entry_has_test_dependencies(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("git clone https://github.com/dream-once/VGGT-RelMem.git", readme)
        self.assertIn("python -m pip install -e '.[dev]'", readme)
        self.assertIn("python -m scripts.verify_public_clone", readme)
        self.assertIn('"Pillow>=10.0"', pyproject)

    def test_optional_binaries_are_not_claimed_as_tracked(self):
        results = load_json(
            ROOT
            / "evidence/week3/d20-reproduction/result_tables.json"
        )
        self.assertFalse(
            results["d15_5"]["binary_artifacts_retained_in_git"]
        )
        self.assertEqual(
            results["d15_5"]["binary_release_status"],
            "OPTIONAL_BINARY_RELEASE_PENDING",
        )
        self.assertIsNone(
            results["claim_boundary"]["performance_improvement"]
        )


if __name__ == "__main__":
    unittest.main()
