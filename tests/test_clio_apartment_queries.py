import hashlib
import json
from pathlib import Path
import unittest

import yaml


class ClioApartmentQueryManifestTests(unittest.TestCase):
    def test_manifest_covers_official_tasks_and_freezes_hash_split(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "configs/clio_apartment_queries.json").read_text())
        task_path = root / manifest["source"]["task_yaml"]
        self.assertEqual(
            hashlib.sha256(task_path.read_bytes()).hexdigest(),
            manifest["source"]["task_yaml_sha256"],
        )
        official = set(yaml.safe_load(task_path.read_text()))
        records = manifest["queries"]
        self.assertEqual({item["task"] for item in records}, official)
        self.assertEqual(len(records), len(official))
        self.assertTrue(all(item["sam_query"].strip() for item in records))
        expected_calibration = set(sorted(
            official,
            key=lambda query: hashlib.sha256(
                f"clio-apartment-v1|{query}".encode()
            ).hexdigest(),
        )[:8])
        actual_calibration = {
            item["task"] for item in records if item["split"] == "calibration"
        }
        self.assertEqual(actual_calibration, expected_calibration)
        self.assertEqual(sum(item["split"] == "development" for item in records), 18)


if __name__ == "__main__":
    unittest.main()
