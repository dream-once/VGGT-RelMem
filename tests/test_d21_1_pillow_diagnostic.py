import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.validate_d21_1_pillow_diagnostic import validate


class D211PillowDiagnosticTests(unittest.TestCase):
    @property
    def root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    @property
    def report(self) -> Path:
        return self.root / "evidence/week4/d21_1-pillow-diagnostic/public_report.json"

    def test_retained_public_bundle_passes_without_local_binaries(self):
        result = validate(self.report, project_root=self.root)
        self.assertEqual(result["status"], "PASS")
        self.assertNotIn("local_artifact_hashes", result["checks"])

    def test_headline_tampering_fails(self):
        with TemporaryDirectory() as directory:
            payload = json.loads(self.report.read_text())
            payload["headline"]["all_prompt_union_frames"] = 24
            report = Path(directory) / "report.json"
            report.write_text(json.dumps(payload))
            self.assertEqual(
                validate(report, project_root=self.root)["status"], "FAIL"
            )

    def test_public_bundle_contains_only_json_and_markdown(self):
        payload = json.loads(self.report.read_text())
        for artifact in payload["artifacts"]["public"]:
            self.assertIn(Path(artifact["path"]).suffix, {".json", ".md"})
        self.assertFalse(
            payload["claim_boundary"]["instance_specific_prompt_is_formal_policy"]
        )
        self.assertIsNone(payload["claim_boundary"]["segmentation_recall"])


if __name__ == "__main__":
    unittest.main()
