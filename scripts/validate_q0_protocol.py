"""Validate D13 Q0 against pinned source and retained D4/D5 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from relground.q0_protocol import (
    Q0_PROTOCOL_ID,
    Q0_PROTOCOL_STATUS,
    audit_source_semantics,
    build_q0_protocol,
    validate_q0_payload,
)
from scripts.validate_single_view_baselines import (
    validate_output as validate_d4_output,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} root must be an object")
    return payload


def _git_commit(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def validate_output(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(path).resolve()
    root = Path(project_root or Path.cwd()).resolve()
    failures: list[str] = []
    for name in ("q0_protocol.json", "run_manifest.json"):
        artifact = output_dir / name
        if not artifact.is_file() or artifact.stat().st_size == 0:
            failures.append(f"missing or empty artifact: {name}")
    if failures:
        return {"status": "FAIL", "failures": failures}

    try:
        protocol = _read_json(output_dir / "q0_protocol.json")
        manifest = _read_json(output_dir / "run_manifest.json")
        validate_q0_payload(protocol)
        expected = build_q0_protocol(
            root,
            created_at=str(protocol["created_at"]),
        )
        source_checks = audit_source_semantics(root)
        d4_report = validate_d4_output(
            root / "evidence/week1/d4-single-view"
        )
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        return {
            "status": "FAIL",
            "failures": [f"invalid Q0 protocol artifact: {error}"],
        }

    if protocol != expected:
        failures.append("Q0 protocol differs from static source/evidence replay")
    if not all(source_checks.values()):
        failures.append(
            "Q0 source semantics failed: "
            + ", ".join(
                key for key, value in source_checks.items() if not value
            )
        )
    pins = protocol["source_pins"]
    actual_commits = {
        "vggt_slam": _git_commit(root / "third_party/VGGT-SLAM"),
        "vggt": _git_commit(
            root / "third_party/VGGT-SLAM/third_party/vggt"
        ),
        "perception_models": _git_commit(
            root
            / "third_party/VGGT-SLAM/third_party/perception_models"
        ),
        "sam3": _git_commit(
            root / "third_party/VGGT-SLAM/third_party/sam3"
        ),
    }
    if pins != actual_commits:
        failures.append("Q0 pinned upstream commits changed")

    expected_d4_errors = {
        "missing or empty artifact: masks.json",
        "missing or empty artifact: preview.png",
    }
    d4_gap_accounted = (
        d4_report.get("status") == "FAIL"
        and set(d4_report.get("errors", [])) == expected_d4_errors
        and protocol["retained_d4"]["strict_validator_rerun"]
        == "FAIL_MISSING_MASKS_AND_PREVIEW"
        and protocol["retained_d4"]["saved_validator_status"] == "PASS"
    )
    if not d4_gap_accounted:
        failures.append("retained D4 strict-validator limitation changed")

    manifest_config = manifest.get("config", {})
    if not isinstance(manifest_config, Mapping):
        failures.append("D13 run manifest config is not an object")
    else:
        expected_manifest = {
            "stage": "D13",
            "status": "CPU_COMPLETE",
            "gpu_acceptance": "PENDING",
            "protocol_id": Q0_PROTOCOL_ID,
            "protocol_status": Q0_PROTOCOL_STATUS,
            "source_file_count": len(protocol["source_files"]),
            "source_check_count": len(protocol["source_checks"]),
            "limitation_count": len(protocol["limitations"]),
        }
        for key, value in expected_manifest.items():
            if manifest_config.get(key) != value:
                failures.append(f"D13 run manifest {key} is inconsistent")
    if manifest.get("peak_vram_mb") is not None:
        failures.append("CPU-only D13 records GPU memory")

    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "stage": "D13",
        "cpu_completion": "COMPLETE",
        "gpu_acceptance": "PENDING",
        "protocol_id": protocol["protocol_id"],
        "protocol_status": protocol["status"],
        "found_it_official": protocol["claims"]["found_it_official"],
        "top1_matches_d5_first": protocol["development_selection"][
            "top1_matches_d5_first"
        ],
        "source_check_count": len(source_checks),
        "source_checks_passed": sum(source_checks.values()),
        "retained_d4_saved_validator_status": protocol["retained_d4"][
            "saved_validator_status"
        ],
        "retained_d4_strict_rerun": protocol["retained_d4"][
            "strict_validator_rerun"
        ],
        "d4_gap_accounted": d4_gap_accounted,
        "limitations": protocol["limitations"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--report")
    args = parser.parse_args()
    report = validate_output(
        args.output_dir,
        project_root=args.project_root,
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
