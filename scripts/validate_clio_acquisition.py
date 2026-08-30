"""Validate a Clio apartment acquisition receipt, optionally against local data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relground.clio_acquisition import build_receipt, validate_receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt")
    parser.add_argument("--project-root")
    parser.add_argument("--verify-local", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    receipt = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
    validate_receipt(receipt)
    checks = {"contract": True, "local_files": None}
    if args.verify_local:
        if not args.project_root:
            raise ValueError("--project-root is required with --verify-local")
        root = Path(args.project_root).resolve()
        rebuilt = build_receipt(
            project_root=root,
            archive_path=root / receipt["archive"]["path"],
            extraction_path=root / receipt["extraction"]["path"],
            apartment_folder_url=receipt["source"]["apartment_folder_url"],
            checked_at=receipt["checked_at"],
        )
        checks["local_files"] = rebuilt == receipt
    report = {
        "schema_version": "0.1",
        "status": "PASS" if all(v is not False for v in checks.values()) else "FAIL",
        "stage": "D16.1-clio-apartment-validation",
        "checks": checks,
        "held_out_status": "CLIO_CUBICLE_UNTOUCHED",
    }
    if args.output:
        Path(args.output).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
