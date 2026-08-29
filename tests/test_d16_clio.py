import json
import tempfile
import unittest
from pathlib import Path

from relground.clio_protocol import TEN_GIB, audit_clio_feasibility
from scripts.validate_d16 import validate


ROOT = Path(__file__).resolve().parents[1]


def manifests():
    dataset = json.loads(
        (ROOT / "configs" / "clio_dataset_manifest.json").read_text()
    )
    splits = json.loads((ROOT / "configs" / "clio_splits.json").read_text())
    return dataset, splits


class ClioProtocolTests(unittest.TestCase):
    def test_real_manifest_fails_closed_without_side_effects(self):
        dataset, splits = manifests()
        result = audit_clio_feasibility(
            dataset, splits, available_bytes=24 * 1024**3,
            checked_at="2026-08-29T00:00:00+00:00",
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(
            result["dataset_download_status"],
            "DATA_DOWNLOAD_BLOCKED_SIZE_UNKNOWN",
        )
        self.assertEqual(result["maximum_peak_bytes"], 14 * 1024**3)
        self.assertFalse(any(result["side_effects"].values()))
        self.assertEqual(result["query_status"], "PENDING_DATA_METADATA")

    def test_known_verified_scene_can_be_ready(self):
        dataset, splits = manifests()
        dataset["dataset_license"]["status"] = "VERIFIED_FOR_DATASET"
        dataset["dataset_license"]["identifier"] = "test-only"
        dataset["dataset_license"]["url"] = "https://example.test/license"
        for scene in dataset["scenes"]:
            scene["archive_bytes"] = 1 * 1024**3
            scene["extracted_bytes"] = 2 * 1024**3
            scene["temporary_bytes"] = 1 * 1024**3
            scene["sha256"] = "a" * 64
            scene["declared_download_status"] = "READY_TO_DOWNLOAD"
        result = audit_clio_feasibility(
            dataset, splits, available_bytes=20 * 1024**3, checked_at="test",
        )
        self.assertEqual(result["dataset_download_status"], "READY_TO_DOWNLOAD")
        self.assertTrue(all(row["download_allowed"] for row in result["scenes"]))

    def test_insufficient_space_is_explicit(self):
        dataset, splits = manifests()
        dataset["dataset_license"]["status"] = "VERIFIED_FOR_DATASET"
        dataset["dataset_license"]["identifier"] = "test-only"
        dataset["dataset_license"]["url"] = "https://example.test/license"
        for scene in dataset["scenes"]:
            scene["archive_bytes"] = 2 * 1024**3
            scene["extracted_bytes"] = 4 * 1024**3
            scene["temporary_bytes"] = 1 * 1024**3
            scene["sha256"] = "b" * 64
            scene["declared_download_status"] = (
                "DATA_DOWNLOAD_BLOCKED_INSUFFICIENT_SPACE"
            )
        result = audit_clio_feasibility(
            dataset, splits, available_bytes=16 * 1024**3, checked_at="test",
        )
        self.assertEqual(result["maximum_peak_bytes"], 6 * 1024**3)
        self.assertEqual(
            result["dataset_download_status"],
            "DATA_DOWNLOAD_BLOCKED_INSUFFICIENT_SPACE",
        )

    def test_fabricated_query_list_is_rejected(self):
        dataset, splits = manifests()
        splits["queries"] = [{"query_id": "invented"}]
        with self.assertRaisesRegex(ValueError, "must not fabricate"):
            audit_clio_feasibility(
                dataset, splits, available_bytes=TEN_GIB, checked_at="test",
            )

    def test_stale_declared_status_is_rejected(self):
        dataset, splits = manifests()
        dataset["scenes"][0]["declared_download_status"] = "READY_TO_DOWNLOAD"
        with self.assertRaisesRegex(ValueError, "stale"):
            audit_clio_feasibility(
                dataset, splits, available_bytes=24 * 1024**3,
                checked_at="test",
            )

    def test_bundle_tampering_fails_validator(self):
        source = ROOT / "evidence" / "week3" / "d16-clio-feasibility"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for path in source.iterdir():
                if path.is_file():
                    (target / path.name).write_bytes(path.read_bytes())
            report = json.loads(
                (target / "feasibility_report.json").read_text()
            )
            report["maximum_peak_bytes"] += 1
            (target / "feasibility_report.json").write_text(json.dumps(report))
            self.assertEqual(validate(target)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
