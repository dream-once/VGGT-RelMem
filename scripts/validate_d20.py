"""Validate D20 by rebuilding into a moved temporary directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from scripts.reproduce_d20 import execute


def run(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.project_root).resolve()
    with tempfile.TemporaryDirectory(prefix="vggt-relmem-d20-") as directory:
        rebuilt = Path(directory) / "moved" / "results"
        report = execute(
            project_root=root,
            manifest_path=(root / args.manifest).resolve(),
            output_dir=rebuilt,
            retained_root=(root / args.retained_root).resolve(),
        )
        checks = {
            "rebuild_pass": report["status"] == "PASS",
            "all_stage_validators_pass": (
                set(report["validators"].values()) == {"PASS"}
            ),
            "readme_numbers_match": all(
                report["readme_checks"].values()
            ),
            "moved_output_matches": all(
                report["retained_output_checks"].values()
            ),
            "optional_binaries_explicit": (
                report["optional_binary_release_status"]
                == "OPTIONAL_BINARY_RELEASE_PENDING"
            ),
        }
    status = "PASS" if all(checks.values()) else "FAIL"
    validation = {
        "schema_version": "0.1",
        "status": status,
        "stage": "D20-validation",
        "checks": checks,
        "source_status": "CPU_COMPLETE",
        "optional_binary_release_status": (
            "OPTIONAL_BINARY_RELEASE_PENDING"
        ),
    }
    if args.output:
        Path(args.output).write_text(
            json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return validation


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument(
        "--manifest", default="configs/d20_reproduction_manifest.json"
    )
    parser.add_argument(
        "--retained-root",
        default="evidence/week3/d20-reproduction",
    )
    parser.add_argument("--output")
    return parser


if __name__ == "__main__":
    raise SystemExit(
        0 if run(build_parser().parse_args())["status"] == "PASS" else 2
    )
