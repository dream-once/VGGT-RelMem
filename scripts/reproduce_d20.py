"""Verify D16-D19 evidence and rebuild every D20 result table."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile

from relground.reproduction_package import (
    OUTPUT_FILES,
    build_result_tables,
    load_json,
    readme_numeric_checks,
    sha256_file,
    validate_reproduction_manifest,
    write_derived_outputs,
)
from scripts.validate_d16 import validate as validate_d16
from scripts.validate_d17 import validate as validate_d17
from scripts.validate_d18 import run as validate_d18
from scripts.validate_d19 import run as validate_d19


def validation_suite(root: Path) -> dict[str, str]:
    with redirect_stdout(StringIO()):
        reports = {
            "D16": validate_d16(
                root / "evidence/week3/d16-clio-feasibility"
            ),
            "D17": validate_d17(
                root / "evidence/week3/d17-relations"
            ),
            "D18": validate_d18(
                argparse.Namespace(
                    project_root=str(root),
                    manifest="configs/d18_experiment_manifest.json",
                    office_evidence=(
                        "evidence/week3/d18-qxa/office-loop"
                    ),
                    synthetic_evidence=(
                        "evidence/week3/d18-qxa/synthetic"
                    ),
                    clio_evidence=(
                        "evidence/week4/clio-apartment-gpu/d18-qxa"
                    ),
                    output=None,
                )
            ),
            "D19": validate_d19(
                argparse.Namespace(
                    project_root=str(root),
                    manifest="configs/d19_ablation_manifest.json",
                    office_evidence=(
                        "evidence/week3/d19-ablations/office-loop"
                    ),
                    synthetic_evidence=(
                        "evidence/week3/d19-ablations/synthetic"
                    ),
                    clio_evidence=(
                        "evidence/week4/clio-apartment-gpu/d19-ablations"
                    ),
                    output=None,
                )
            ),
        }
    return {key: value["status"] for key, value in reports.items()}


def compare_retained(
    generated: Path, retained: Path
) -> dict[str, bool]:
    return {
        name: (
            (retained / name).is_file()
            and (generated / name).read_bytes()
            == (retained / name).read_bytes()
        )
        for name in OUTPUT_FILES
    }


def execute(
    *,
    project_root: Path,
    manifest_path: Path,
    output_dir: Path,
    retained_root: Path | None,
) -> dict[str, object]:
    manifest = load_json(manifest_path)
    validate_reproduction_manifest(manifest, project_root=project_root)
    results = build_result_tables(project_root, manifest)
    output_hashes = write_derived_outputs(output_dir, results)
    validators = validation_suite(project_root)
    readme_checks = readme_numeric_checks(
        (project_root / "README.md").read_text(encoding="utf-8"),
        results,
    )
    retained_checks = (
        None
        if retained_root is None
        else compare_retained(output_dir, retained_root)
    )
    status = "PASS"
    if (
        set(validators.values()) != {"PASS"}
        or not all(readme_checks.values())
        or (
            retained_checks is not None
            and not all(retained_checks.values())
        )
    ):
        status = "FAIL"
    report = {
        "schema_version": "0.1",
        "status": status,
        "stage": "D20-reproduction",
        "validators": validators,
        "readme_checks": readme_checks,
        "retained_output_checks": retained_checks,
        "output_sha256": output_hashes,
        "source_status": "CPU_COMPLETE",
        "optional_binary_release_status": (
            "OPTIONAL_BINARY_RELEASE_PENDING"
        ),
    }
    (output_dir / "reproduction_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def run(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.project_root).resolve()
    report = execute(
        project_root=root,
        manifest_path=(root / args.manifest).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        retained_root=(
            None
            if args.verify_retained is None
            else (root / args.verify_retained).resolve()
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument(
        "--manifest", default="configs/d20_reproduction_manifest.json"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--verify-retained")
    return parser


if __name__ == "__main__":
    raise SystemExit(
        0 if run(build_parser().parse_args())["status"] == "PASS" else 2
    )
