import argparse
from contextlib import redirect_stdout
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from relground.q0_protocol import (
    Q0_PROTOCOL_ID,
    Q0_PROTOCOL_STATUS,
    audit_source_semantics,
    build_q0_protocol,
    validate_q0_payload,
)
from scripts.freeze_q0_protocol import run as run_freeze
from scripts.validate_q0_protocol import validate_output


class Q0ProtocolTests(unittest.TestCase):
    @property
    def root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    def protocol(self) -> dict:
        return build_q0_protocol(
            self.root,
            created_at="2026-08-27T00:00:00+00:00",
        )

    def build_bundle(self, root: Path) -> Path:
        output = root / "d13"
        args = argparse.Namespace(
            project_root=str(self.root),
            output_dir=str(output),
        )
        with redirect_stdout(io.StringIO()):
            self.assertEqual(run_freeze(args), 0)
        return output

    def test_protocol_name_status_and_claim_boundaries_are_frozen(self) -> None:
        protocol = self.protocol()

        validate_q0_payload(protocol)
        self.assertEqual(protocol["protocol_id"], Q0_PROTOCOL_ID)
        self.assertEqual(protocol["status"], Q0_PROTOCOL_STATUS)
        self.assertFalse(protocol["claims"]["found_it_official"])
        self.assertFalse(
            protocol["claims"]["vggt_slam_official_reproduction"]
        )
        self.assertEqual(
            protocol["claims"]["local_b0_label_scope"],
            "audited single-view lifting baseline only",
        )

    def test_all_pinned_source_semantics_are_detected(self) -> None:
        checks = audit_source_semantics(self.root)

        self.assertEqual(len(checks), 10)
        self.assertTrue(all(checks.values()))

    def test_q0_top1_matches_d5_raw_rank_one(self) -> None:
        selection = self.protocol()["development_selection"]

        self.assertTrue(selection["top1_matches_d5_first"])
        self.assertEqual(selection["raw_rank_1_frame"], "frame_0001")
        self.assertEqual(
            selection["raw_rank_1_frame"],
            selection["upstream_top1_frame"],
        )

    def test_retained_d4_gap_is_explicit_not_silently_passed(self) -> None:
        retained = self.protocol()["retained_d4"]

        self.assertEqual(retained["saved_validator_status"], "PASS")
        self.assertEqual(
            retained["strict_validator_rerun"],
            "FAIL_MISSING_MASKS_AND_PREVIEW",
        )
        self.assertEqual(retained["sam_threshold"], 0.5)
        self.assertFalse(retained["mask_resizing_after_sam"])
        self.assertEqual(retained["preprocess_target_size"], 518)

    def test_protocol_rejects_preprocess_or_sam_tampering(self) -> None:
        protocol = self.protocol()
        preprocess = copy.deepcopy(protocol)
        preprocess["preprocess"]["target_size"] = 512
        threshold = copy.deepcopy(protocol)
        threshold["segmentation"]["confidence_threshold"] = 0.25

        with self.assertRaisesRegex(ValueError, "preprocess"):
            validate_q0_payload(preprocess)
        with self.assertRaisesRegex(ValueError, "segmentation"):
            validate_q0_payload(threshold)

    def test_protocol_rejects_robust_lifting_or_obb_tampering(self) -> None:
        protocol = self.protocol()
        lifting = copy.deepcopy(protocol)
        lifting["lifting"]["mad_filter"] = True
        obb = copy.deepcopy(protocol)
        obb["obb"]["method"] = "robust PCA"

        with self.assertRaisesRegex(ValueError, "lifting"):
            validate_q0_payload(lifting)
        with self.assertRaisesRegex(ValueError, "OBB"):
            validate_q0_payload(obb)

    def test_static_validator_replays_passing_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = self.build_bundle(Path(directory))
            report = validate_output(output, project_root=self.root)

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["protocol_status"], "upstream-aligned")
        self.assertEqual(report["source_checks_passed"], 10)
        self.assertTrue(report["top1_matches_d5_first"])
        self.assertTrue(report["d4_gap_accounted"])

    def test_static_validator_rejects_source_hash_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = self.build_bundle(Path(directory))
            path = output / "q0_protocol.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["source_files"][0]["sha256"] = "0" * 64
            path.write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
            report = validate_output(output, project_root=self.root)

        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any(
            "differs from static" in item for item in report["failures"]
        ))


if __name__ == "__main__":
    unittest.main()
