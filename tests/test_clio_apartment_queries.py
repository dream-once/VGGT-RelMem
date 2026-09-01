import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from relground.clio_query_manifest import (
    validate_official_task_source,
    validate_tracked_manifest,
)


class ClioApartmentQueryManifestTests(unittest.TestCase):
    def test_tracked_manifest_freezes_hash_split_without_private_data(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "configs/clio_apartment_queries.json").read_text())
        records = manifest["queries"]
        self.assertEqual(validate_tracked_manifest(manifest)["status"], "PASS")
        expected_calibration = set(sorted(
            {item["task"] for item in records},
            key=lambda query: hashlib.sha256(
                f"clio-apartment-v1|{query}".encode()
            ).hexdigest(),
        )[:8])
        actual_calibration = {
            item["task"] for item in records if item["split"] == "calibration"
        }
        self.assertEqual(actual_calibration, expected_calibration)
        self.assertEqual(sum(item["split"] == "development" for item in records), 18)

    def test_tracked_cubicle_manifest_freezes_complete_held_out_split(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "configs/clio_cubicle_queries.json").read_text())
        records = manifest["queries"]
        self.assertEqual(validate_tracked_manifest(manifest)["status"], "PASS")
        self.assertEqual(len(records), 18)
        self.assertTrue(all(item["split"] == "held-out" for item in records))

    def test_official_source_check_is_an_explicit_integration_contract(self) -> None:
        tasks = {"task a": {}, "task b": {}}
        ordered = sorted(
            tasks,
            key=lambda query: hashlib.sha256(
                f"clio-apartment-v1|{query}".encode()
            ).hexdigest(),
        )
        with tempfile.TemporaryDirectory() as directory:
            task_path = Path(directory) / "tasks.yaml"
            task_path.write_text(yaml.safe_dump(tasks), encoding="utf-8")
            manifest = {
                "source": {
                    "task_yaml": "data/tasks.yaml",
                    "task_yaml_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
                },
                "split": {"calibration_count": 1, "development_count": 1},
                "queries": [
                    {"task": ordered[0], "sam_query": ordered[0], "split": "calibration"},
                    {"task": ordered[1], "sam_query": ordered[1], "split": "development"},
                ],
            }
            self.assertEqual(
                validate_official_task_source(manifest, task_path)["status"], "PASS"
            )


if __name__ == "__main__":
    unittest.main()
