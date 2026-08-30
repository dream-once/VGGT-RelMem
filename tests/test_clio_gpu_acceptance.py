import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_clio_gpu_acceptance import validate

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "evidence/week4/clio-apartment-gpu/gpu_acceptance_report.json"

class ClioGpuAcceptanceTests(unittest.TestCase):
    def test_public_report_passes_without_local_binaries(self):
        result = validate(REPORT, project_root=ROOT)
        self.assertEqual(result["status"], "PASS")
        self.assertNotIn("local_artifact_hashes", result["checks"])

    def test_tampered_split_fails(self):
        payload = json.loads(REPORT.read_text(encoding="utf-8"))
        payload["scope"]["held_out_downloaded"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = validate(path, project_root=ROOT)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["checks"]["split_guard"])

if __name__ == "__main__":
    unittest.main()
