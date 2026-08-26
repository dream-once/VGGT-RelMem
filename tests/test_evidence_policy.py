import json
import unittest
from pathlib import Path


class EvidencePolicyTests(unittest.TestCase):
    @property
    def week2(self) -> Path:
        return Path(__file__).resolve().parents[1] / "evidence" / "week2"

    def test_week2_is_lightweight_json_or_markdown_only(self) -> None:
        files = sorted(
            path for path in self.week2.rglob("*") if path.is_file()
        )
        self.assertTrue(files)
        unexpected = [
            path.relative_to(self.week2).as_posix()
            for path in files
            if path.suffix.lower() not in {".json", ".md"}
        ]
        self.assertEqual(unexpected, [])
        self.assertLessEqual(
            sum(path.stat().st_size for path in files),
            256 * 1024,
        )

    def test_prediction_evidence_contains_no_ground_truth(self) -> None:
        root = self.week2 / "d9-office-loop-trash-can"
        result = json.loads(
            (root / "prediction" / "d9_result.json").read_text(
                encoding="utf-8"
            )
        )
        serialized = json.dumps(result, sort_keys=True).lower()
        for forbidden in (
            "pair_labels",
            "expected_same",
            "error_type",
            "failure_cases",
            '"metrics"',
            '"f1"',
        ):
            self.assertNotIn(forbidden, serialized)

    def test_evaluation_references_prediction_relatively(self) -> None:
        root = self.week2 / "d9-office-loop-trash-can"
        result = json.loads(
            (root / "evaluation" / "d9_evaluation.json").read_text(
                encoding="utf-8"
            )
        )
        reference = Path(result["source"]["prediction_result"])
        self.assertFalse(reference.is_absolute())
        self.assertEqual(
            reference.as_posix(),
            "../prediction/d9_result.json",
        )


if __name__ == "__main__":
    unittest.main()
