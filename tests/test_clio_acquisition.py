import copy
import tempfile
import unittest
from pathlib import Path
import zipfile

from relground.clio_acquisition import build_receipt, validate_receipt


APARTMENT_URL = (
    "https://www.dropbox.com/scl/fo/5bkv8rsa2xvwmvom6bmza/"
    "AF3eng3PcI9H3N-yMhKZIKk/apartment?"
    "rlkey=wx1njghufcxconm1znidc1hgw&dl=0"
)


class ClioAcquisitionTests(unittest.TestCase):
    def _fixture(self, root: Path):
        archive = root / "data/clio/downloads/apartment.zip"
        archive.parent.mkdir(parents=True)
        with zipfile.ZipFile(archive, "w") as handle:
            for index in range(24):
                handle.writestr(
                    f"apartment/images/rgb_{index}.jpg", b"rgb"
                )
            for name in (
                "tasks_apartment.yaml", "rooms_apartment.yaml",
                "region_tasks_apartment.yaml",
            ):
                handle.writestr(f"apartment/metadata/{name}", b"tasks")
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(root / "data/clio")
        return archive, root / "data/clio/apartment"

    def test_local_apartment_receipt_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, extracted = self._fixture(root)
            receipt = build_receipt(
                project_root=root,
                archive_path=archive,
                extraction_path=extracted,
                apartment_folder_url=APARTMENT_URL,
                checked_at="2026-08-30T10:00:00+00:00",
            )
            self.assertEqual(validate_receipt(receipt), receipt)
            self.assertEqual(receipt["archive"]["file_entry_count"], 27)
            self.assertEqual(receipt["archive"]["top_level_entries"], ["apartment"])
            self.assertFalse(
                receipt["materialization_scope"]["full_scene_claimed"]
            )
            self.assertFalse(receipt["split_guard"]["held_out_downloaded"])
            self.assertFalse(receipt["usage_boundary"]["redistribution_allowed"])

    def test_zip_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "data/clio/downloads/apartment.zip"
            archive.parent.mkdir(parents=True)
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../cubicle/secret", b"bad")
            extracted = root / "data/clio/apartment"
            extracted.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "path escape"):
                build_receipt(
                    project_root=root,
                    archive_path=archive,
                    extraction_path=extracted,
                    apartment_folder_url=APARTMENT_URL,
                    checked_at="test",
                )

    def test_redistribution_or_held_out_tampering_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, extracted = self._fixture(root)
            receipt = build_receipt(
                project_root=root,
                archive_path=archive,
                extraction_path=extracted,
                apartment_folder_url=APARTMENT_URL,
                checked_at="test",
            )
            redistributed = copy.deepcopy(receipt)
            redistributed["usage_boundary"]["redistribution_allowed"] = True
            with self.assertRaisesRegex(ValueError, "usage boundary"):
                validate_receipt(redistributed)
            leaked = copy.deepcopy(receipt)
            leaked["split_guard"]["held_out_downloaded"] = True
            with self.assertRaisesRegex(ValueError, "split guard"):
                validate_receipt(leaked)


if __name__ == "__main__":
    unittest.main()
