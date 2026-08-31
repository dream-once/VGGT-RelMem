"""Validate and visualize a completed D21.1 SAM diagnostic sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw

from relground.segmentation_sweep import (
    read_json,
    resolve_reference,
    sha256_file,
    validate_sweep_plan,
)
from scripts.validate_d6 import validate_output as validate_d6


EXPECTED_EMPTY_DIAGNOSTIC_ERRORS = {
    "D6 result status is 'INSUFFICIENT_MULTIFRAME_3D_EVIDENCE'",
    "fewer than two frames produced valid 3D observations",
}


def is_valid_diagnostic_d6(report: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
    if result.get("status") == "PASS":
        return report.get("status") == "PASS" and not report.get("errors")
    if result.get("status") != "INSUFFICIENT_MULTIFRAME_3D_EVIDENCE":
        return False
    return (
        report.get("status") == "FAIL"
        and set(report.get("errors", [])) == EXPECTED_EMPTY_DIAGNOSTIC_ERRORS
        and int(report.get("mask_instances", -1)) == int(result.get("sam_instances", -2))
        and int(report.get("lifted_instances", -1)) == int(result.get("lifted_instances", -2))
        and len(report.get("frames_with_lifted_observations", [])) < 2
    )


def normalized_baseline_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(payload))
    result.pop("created_at", None)
    result.pop("selection_source", None)
    return result


def mask_hashes(root: Path) -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted((root / "masks").glob("*.npy"))
    }


def validate_and_compare(
    *,
    project_root: Path,
    sweep_root: Path,
    baseline_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    project_root = project_root.resolve()
    sweep_root = sweep_root.resolve()
    plan = read_json(sweep_root / "sweep_plan.json")
    execution = read_json(sweep_root / "sweep_execution.json")
    plan_report = validate_sweep_plan(plan, project_root=project_root)
    failures: list[str] = []
    if plan_report["status"] != "PASS":
        failures.append("sweep plan failed independent validation")
    if execution.get("status") != "GPU_DIAGNOSTIC_COMPLETE" or execution.get("mode") != "execute":
        failures.append("sweep execution is not a completed GPU diagnostic")
    plan_rows = plan.get("experiments", [])
    execution_rows = execution.get("experiments", [])
    if len(plan_rows) != len(execution_rows):
        failures.append("plan/execution experiment counts differ")
    candidate_ids = list(plan.get("candidate_frame_ids", []))
    comparison_rows: list[dict[str, Any]] = []
    result_by_id: dict[str, dict[str, Any]] = {}
    validation_by_id: dict[str, dict[str, Any]] = {}
    for plan_row, summary in zip(plan_rows, execution_rows):
        experiment_id = str(plan_row.get("experiment_id", ""))
        if experiment_id != summary.get("experiment_id"):
            failures.append("plan/execution experiment order differs")
            continue
        output = resolve_reference(project_root, str(plan_row["output_ref"]))
        expected_output = sweep_root / "experiments" / experiment_id
        if output != expected_output:
            failures.append(f"unexpected output path for {experiment_id}")
            continue
        result = read_json(output / "d6_result.json")
        d6_report = validate_d6(output)
        if not is_valid_diagnostic_d6(d6_report, result):
            failures.append(f"D6 artifact validation failed for {experiment_id}")
        result_ids = [str(row.get("frame_id", "")) for row in result.get("selected_frames", [])]
        if result_ids != candidate_ids:
            failures.append(f"candidate universe changed for {experiment_id}")
        expected_summary = {
            "experiment_id": experiment_id,
            "query": plan_row["query"],
            "sam_threshold": plan_row["sam_threshold"],
            "role": plan_row["role"],
            "status": result["status"],
            "sam_instances": result["sam_instances"],
            "lifted_instances": result["lifted_instances"],
            "rejected_instances": len(result["rejected_instances"]),
            "frames_with_masks": result["frames_with_masks"],
            "frames_with_lifted_observations": result["frames_with_lifted_observations"],
            "inference_executed": True,
        }
        if summary != expected_summary:
            failures.append(f"execution summary differs from D6 result for {experiment_id}")
        result_by_id[experiment_id] = result
        validation_by_id[experiment_id] = d6_report
        comparison_rows.append({
            **expected_summary,
            "d6_result_ref": (output / "d6_result.json").relative_to(project_root).as_posix(),
            "d6_result_sha256": sha256_file(output / "d6_result.json"),
            "artifact_contract": (
                "PASS" if d6_report["status"] == "PASS"
                else "PASS_VALID_ZERO_OR_SINGLE_VIEW_DIAGNOSTIC"
            ),
        })

    baseline_id = "baseline-pillow-050"
    baseline_replay = False
    if baseline_id not in result_by_id:
        failures.append("frozen pillow@0.5 baseline is missing")
    else:
        new_root = sweep_root / "experiments" / baseline_id
        old_result = read_json(baseline_root / "d6_result.json")
        new_result = result_by_id[baseline_id]
        baseline_replay = (
            normalized_baseline_result(old_result) == normalized_baseline_result(new_result)
            and read_json(baseline_root / "masks.json") == read_json(new_root / "masks.json")
            and read_json(baseline_root / "observations.json") == read_json(new_root / "observations.json")
            and read_json(baseline_root / "selection.json") == read_json(new_root / "selection.json")
            and mask_hashes(baseline_root) == mask_hashes(new_root)
        )
        if not baseline_replay:
            failures.append("pillow@0.5 baseline differs from retained GPU run")

    frame_rows: list[dict[str, Any]] = []
    for rank, frame_id in enumerate(candidate_ids, start=1):
        detections: dict[str, Any] = {}
        for plan_row in plan_rows:
            experiment_id = plan_row["experiment_id"]
            if experiment_id not in result_by_id:
                continue
            processed = {
                row["frame_id"]: row
                for row in result_by_id[experiment_id]["processed_frames"]
            }[frame_id]
            detections[experiment_id] = {
                "sam_instances": processed["sam_instances"],
                "lifted_instances": processed["lifted_instances"],
                "rejected_instances": processed["rejected_instances"],
                "preview_ref": (
                    sweep_root / "experiments" / experiment_id / processed["preview"]
                ).relative_to(project_root).as_posix(),
            }
        frame_rows.append({"rank": rank, "frame_id": frame_id, "detections": detections})

    detection_sets = {
        row["experiment_id"]: set(row["frames_with_masks"])
        for row in comparison_rows
    }
    all_detected = set().union(*detection_sets.values()) if detection_sets else set()
    baseline_detected = detection_sets.get(baseline_id, set())
    dinosaur_detected = detection_sets.get("dinosaur-pillow-050", set())
    low_threshold_detected = (
        detection_sets.get("baseline-pillow-040", set())
        | detection_sets.get("baseline-pillow-030", set())
    )
    comparison = {
        "schema_version": "0.1",
        "status": "PASS" if not failures else "FAIL",
        "stage": "D21.1-segmentation-prompt-sweep-comparison",
        "scene_id": plan.get("scene_id"),
        "split_role": plan.get("split_role"),
        "source": {
            "sweep_plan": (sweep_root / "sweep_plan.json").relative_to(project_root).as_posix(),
            "sweep_plan_sha256": sha256_file(sweep_root / "sweep_plan.json"),
            "sweep_execution": (sweep_root / "sweep_execution.json").relative_to(project_root).as_posix(),
            "sweep_execution_sha256": sha256_file(sweep_root / "sweep_execution.json"),
        },
        "candidate_frame_count": len(candidate_ids),
        "experiments": comparison_rows,
        "frame_results": frame_rows,
        "set_comparison": {
            "baseline_detected_frames": sorted(baseline_detected),
            "dinosaur_prompt_detected_frames": sorted(dinosaur_detected),
            "low_threshold_detected_frames": sorted(low_threshold_detected),
            "all_experiments_union_frames": sorted(all_detected),
            "all_experiments_union_count": len(all_detected),
            "dinosaur_complementary_to_baseline": sorted(dinosaur_detected - baseline_detected),
            "low_threshold_complementary_to_baseline": sorted(low_threshold_detected - baseline_detected),
        },
        "acceptance": {
            "plan_valid": plan_report["status"] == "PASS",
            "all_d6_artifacts_internally_consistent": not any(
                item.startswith("D6 artifact validation failed") for item in failures
            ),
            "candidate_universe_preserved": not any(
                "candidate universe changed" in item for item in failures
            ),
            "baseline_bitwise_replayed": baseline_replay,
            "label_free_comparison": True,
        },
        "claim_boundary": {
            "frame_visibility_labels": "PENDING_MANUAL_ANNOTATION",
            "segmentation_recall": None,
            "dinosaur_prompt_is_formal_policy": False,
            "cubicle_accessed": False,
            "performance_claim": None,
        },
        "failures": failures,
    }
    validation = {
        "schema_version": "0.1",
        "status": comparison["status"],
        "stage": "D21.1-segmentation-prompt-sweep-validation",
        "checks": comparison["acceptance"],
        "failures": failures,
        "d6_reports": validation_by_id,
    }
    return comparison, validation


def save_contact_sheet(
    comparison: Mapping[str, Any], *, project_root: Path, output: Path
) -> None:
    experiments = comparison["experiments"]
    frames = comparison["frame_results"]
    thumb_width, thumb_height = 192, 144
    cell_width, cell_height = 202, 174
    label_width = 138
    header_height = 52
    sheet = Image.new(
        "RGB",
        (label_width + len(experiments) * cell_width, header_height + len(frames) * cell_height),
        (20, 20, 20),
    )
    draw = ImageDraw.Draw(sheet)
    for column, experiment in enumerate(experiments):
        x = label_width + column * cell_width + 4
        draw.text((x, 8), experiment["experiment_id"], fill=(245, 245, 245))
        draw.text(
            (x, 27),
            f"masks={experiment['sam_instances']} lifted={experiment['lifted_instances']}",
            fill=(180, 180, 180),
        )
    for row_index, frame in enumerate(frames):
        y = header_height + row_index * cell_height
        draw.text((6, y + 12), f"#{frame['rank']:02d}", fill=(180, 180, 180))
        draw.text((6, y + 34), frame["frame_id"], fill=(245, 245, 245))
        for column, experiment in enumerate(experiments):
            experiment_id = experiment["experiment_id"]
            detection = frame["detections"][experiment_id]
            preview = project_root / detection["preview_ref"]
            with Image.open(preview) as source:
                image = source.convert("RGB").resize((thumb_width, thumb_height))
            x = label_width + column * cell_width + 4
            sheet.paste(image, (x, y + 4))
            color = (80, 220, 120) if detection["sam_instances"] else (150, 150, 150)
            draw.text(
                (x, y + 151),
                f"SAM {detection['sam_instances']} / 3D {detection['lifted_instances']}",
                fill=color,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=88)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument(
        "--sweep-root",
        default=str(root / "runs/clio-apartment-gpu/d21_1-pillow-prompt-sweep-gpu"),
    )
    parser.add_argument(
        "--baseline-root", default=str(root / "runs/clio-apartment-gpu/d6-pillow-all")
    )
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    sweep_root = Path(args.sweep_root).resolve()
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else sweep_root / "comparison"
    )
    comparison, validation = validate_and_compare(
        project_root=project_root,
        sweep_root=sweep_root,
        baseline_root=Path(args.baseline_root).resolve(),
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_contact_sheet(
        comparison,
        project_root=project_root,
        output=output / "comparison_contact_sheet.jpg",
    )
    detected_ids = set(
        comparison["set_comparison"]["all_experiments_union_frames"]
    )
    compact = dict(comparison)
    compact["frame_results"] = [
        row for row in comparison["frame_results"]
        if row["frame_id"] in detected_ids
    ]
    save_contact_sheet(
        compact,
        project_root=project_root,
        output=output / "detected_union_contact_sheet.jpg",
    )
    print(json.dumps({
        "status": comparison["status"],
        "candidate_frame_count": comparison["candidate_frame_count"],
        "union_count": comparison["set_comparison"]["all_experiments_union_count"],
        "acceptance": comparison["acceptance"],
        "contact_sheet": str(output / "comparison_contact_sheet.jpg"),
        "detected_union_contact_sheet": str(
            output / "detected_union_contact_sheet.jpg"
        ),
        "failures": comparison["failures"],
    }, ensure_ascii=False, indent=2))
    return 0 if comparison["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
