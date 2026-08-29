from pathlib import Path
import shutil
import tempfile
import unittest

from relground.observation_cache import sha256_file
from relground.public_evidence import (
    load_json,
    validate_public_bundle,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "evidence/week3/d15-gpu-public"


class PublicGpuEvidenceTests(unittest.TestCase):
    def test_retained_public_bundle_passes(self):
        report = validate_public_bundle(BUNDLE)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(all(report["checks"].values()))

    def test_gpu_report_contains_no_absolute_paths(self):
        report = load_json(BUNDLE / "gpu_acceptance_report.json")

        def walk(value):
            if isinstance(value, dict):
                for item in value.values():
                    yield from walk(item)
            elif isinstance(value, list):
                for item in value:
                    yield from walk(item)
            elif isinstance(value, str):
                yield value

        absolute = [
            value for value in walk(report)
            if Path(value).is_absolute()
        ]
        self.assertEqual(absolute, [])

    def test_q2_boundary_does_not_claim_coverage(self):
        manifest = load_json(BUNDLE / "artifact_manifest.json")
        semantics = manifest["q2_semantics"]
        self.assertEqual(
            semantics["canonical_metric"], "new_observation_count"
        )
        self.assertFalse(semantics["coverage_aware"])
        self.assertIsNone(semantics["performance_claim"])

    def test_tampered_trace_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            moved_week3 = Path(directory) / "week3"
            shutil.copytree(ROOT / "evidence/week3", moved_week3)
            moved = moved_week3 / "d15-gpu-public"
            trace = load_json(moved / "d15_complete_trace.json")
            trace["steps"][1]["selected"]["policy_score"] += 1e-6
            write_json(moved / "d15_complete_trace.json", trace)
            manifest = load_json(moved / "artifact_manifest.json")
            manifest["artifacts"]["d15_complete_trace"]["sha256"] = (
                sha256_file(moved / "d15_complete_trace.json")
            )
            write_json(moved / "artifact_manifest.json", manifest)
            report = validate_public_bundle(moved)
            self.assertEqual(report["status"], "FAIL")
            self.assertIn("float differs", report["failures"][0])


if __name__ == "__main__":
    unittest.main()
