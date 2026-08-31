"""Prepare or execute the frozen Clio apartment SAM diagnostic sweep."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from relground.segmentation_sweep import (
    build_sweep_plan,
    derive_selection,
    read_json,
    validate_prompt_config,
    validate_source_selection,
    validate_sweep_plan,
)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    project_root = Path(args.project_root).resolve()
    config_path = Path(args.config).resolve()
    source_selection_path = Path(args.selection).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"prompt-sweep output is not empty: {output_root}; use a new directory"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    config = read_json(config_path)
    experiments = validate_prompt_config(config)
    source_selection = read_json(source_selection_path)
    validate_source_selection(source_selection)
    for experiment in experiments:
        write_json(
            output_root / "selections" / f"{experiment['experiment_id']}.json",
            derive_selection(source_selection, experiment["query"]),
        )
    plan = build_sweep_plan(
        project_root=project_root,
        config_path=config_path,
        source_selection_path=source_selection_path,
        output_root=output_root,
    )
    report = validate_sweep_plan(plan, project_root=project_root)
    if report["status"] != "PASS":
        raise ValueError("prepared prompt sweep failed validation")
    write_json(output_root / "sweep_plan.json", plan)
    write_json(output_root / "preparation_validation.json", report)
    return output_root, plan


def d6_command(
    args: argparse.Namespace,
    *,
    selection: Path,
    output_dir: Path,
    threshold: float,
    check_only: bool,
) -> list[str]:
    if not args.sam3_checkpoint:
        raise ValueError("--sam3-checkpoint is required for check/execute mode")
    command = [
        sys.executable, "-m", "scripts.run_sam_topk_lifting",
        "--selection", str(selection),
        "--geometry", str(Path(args.geometry).resolve()),
        "--geometry-manifest", str(Path(args.geometry_manifest).resolve()),
        "--output-dir", str(output_dir),
        "--project-root", str(Path(args.project_root).resolve()),
        "--sam3-checkpoint", str(Path(args.sam3_checkpoint).resolve()),
        "--sam-threshold", str(threshold),
        "--device", args.device,
    ]
    if check_only:
        command.append("--check-only")
    return command


def run_prepared(
    args: argparse.Namespace,
    *,
    output_root: Path,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    check_only = args.mode == "check"
    rows: list[dict[str, Any]] = []
    for experiment in plan["experiments"]:
        selection = project_root / experiment["derived_selection_ref"]
        output_dir = project_root / experiment["output_ref"]
        command = d6_command(
            args,
            selection=selection,
            output_dir=output_dir,
            threshold=float(experiment["sam_threshold"]),
            check_only=check_only,
        )
        completed = subprocess.run(
            command,
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if check_only:
            if completed.returncode != 0:
                raise RuntimeError(
                    f"source check failed for {experiment['experiment_id']}: "
                    f"{completed.stderr.strip()}"
                )
            result = json.loads(completed.stdout)
            if result.get("status") != "SOURCE_READY":
                raise RuntimeError("D6 source check did not report SOURCE_READY")
            rows.append({
                "experiment_id": experiment["experiment_id"],
                "query": experiment["query"],
                "sam_threshold": experiment["sam_threshold"],
                "status": "SOURCE_READY",
                "selected_frames": result["selected_frames"],
                "inference_executed": False,
            })
            continue
        result_path = output_dir / "d6_result.json"
        if completed.returncode not in (0, 2) or not result_path.is_file():
            raise RuntimeError(
                f"D6 execution failed for {experiment['experiment_id']}: "
                f"{completed.stderr.strip()}"
            )
        result = read_json(result_path)
        if result.get("query") != experiment["query"]:
            raise ValueError("D6 result query differs from sweep experiment")
        result_frame_ids = [
            row["frame_id"] for row in result.get("selected_frames", [])
        ]
        source_frame_ids = [
            row["frame_id"] for row in read_json(selection)["frames"]
        ]
        if result_frame_ids != source_frame_ids:
            raise ValueError("D6 result changed the frozen candidate universe")
        rows.append({
            "experiment_id": experiment["experiment_id"],
            "query": experiment["query"],
            "sam_threshold": experiment["sam_threshold"],
            "role": experiment["role"],
            "status": result["status"],
            "sam_instances": result["sam_instances"],
            "lifted_instances": result["lifted_instances"],
            "rejected_instances": len(result["rejected_instances"]),
            "frames_with_masks": result["frames_with_masks"],
            "frames_with_lifted_observations": result[
                "frames_with_lifted_observations"
            ],
            "inference_executed": True,
        })
    status = "SOURCE_READY_GPU_INFERENCE_PENDING" if check_only else "GPU_DIAGNOSTIC_COMPLETE"
    report = {
        "schema_version": "0.1",
        "status": status,
        "stage": "D21.1-segmentation-prompt-sweep-execution",
        "scene_id": plan["scene_id"],
        "split_role": plan["split_role"],
        "mode": args.mode,
        "experiments": rows,
        "claim_boundary": plan["claim_boundary"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_root / "sweep_execution.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "check", "execute"), default="prepare")
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument(
        "--config", default=str(root / "configs/clio_apartment_pillow_prompt_audit.json")
    )
    parser.add_argument(
        "--selection", default=str(root / "runs/clio-apartment-gpu/d5-pillow-all/topk_24.json")
    )
    parser.add_argument(
        "--geometry", default=str(root / "runs/clio-apartment-gpu/geometry.npz")
    )
    parser.add_argument(
        "--geometry-manifest", default=str(root / "runs/clio-apartment-gpu/geometry.manifest.json")
    )
    parser.add_argument(
        "--output-root", default=str(root / "runs/clio-apartment-gpu/d21_1-pillow-prompt-sweep")
    )
    parser.add_argument("--sam3-checkpoint")
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root, plan = prepare(args)
    if args.mode == "prepare":
        payload = {
            "status": "SOURCE_PREPARED_GPU_INFERENCE_PENDING",
            "plan": str(output_root / "sweep_plan.json"),
            "experiment_count": len(plan["experiments"]),
        }
    else:
        payload = run_prepared(args, output_root=output_root, plan=plan)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
