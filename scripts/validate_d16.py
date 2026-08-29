"""Validate a self-contained D16 Clio feasibility bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import json

from relground.clio_protocol import audit_clio_feasibility


REQUIRED_FILES = (
    "dataset_manifest.json", "split_manifest.json",
    "feasibility_report.json", "README.md",
)


def validate(bundle: str | Path) -> dict[str, Any]:
    root = Path(bundle)
    errors = [
        f"missing {name}" for name in REQUIRED_FILES
        if not (root / name).is_file()
    ]
    if errors:
        return {"status": "FAIL", "stage": "D16", "errors": errors}
    try:
        dataset = json.loads(
            (root / "dataset_manifest.json").read_text(encoding="utf-8")
        )
        splits = json.loads(
            (root / "split_manifest.json").read_text(encoding="utf-8")
        )
        report = json.loads(
            (root / "feasibility_report.json").read_text(encoding="utf-8")
        )
        expected = audit_clio_feasibility(
            dataset, splits, available_bytes=int(report["available_bytes"]),
            checked_at=str(report["checked_at"]),
        )
        if report != expected:
            errors.append("feasibility report does not recompute exactly")
        if report.get("completion") != "CPU_COMPLETE":
            errors.append("D16 completion must be CPU_COMPLETE")
        if report.get("dataset_download_status") != (
            "DATA_DOWNLOAD_BLOCKED_SIZE_UNKNOWN"
        ):
            errors.append("unknown official sizes must block download")
        if report.get("dataset_license_status") != "DATA_LICENSE_UNVERIFIED":
            errors.append("dataset licence status is not explicit")
        if report.get("query_status") != "PENDING_DATA_METADATA":
            errors.append("query list must remain pending")
        if any(report.get("side_effects", {}).values()):
            errors.append("D16 metadata audit must have no side effects")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(str(exc))
    return {
        "status": "PASS" if not errors else "FAIL", "stage": "D16",
        "checks": {
            "manifest_round_trip": not errors,
            "disk_formula_recomputed": not errors,
            "fail_closed_download_gate": not errors,
            "no_download_or_gpu_side_effects": not errors,
        },
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle")
    parser.add_argument("--write-report")
    args = parser.parse_args()
    result = validate(args.bundle)
    if args.write_report:
        Path(args.write_report).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
