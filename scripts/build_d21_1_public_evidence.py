"""Build lightweight public evidence for the D21.1 pillow diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from relground.clio_alignment_audit import validate_alignment_readiness
from relground.segmentation_sweep import read_json, sha256_file
from scripts.validate_a2_association import validate_output as validate_a2
from scripts.validate_a2_evaluation import validate_output as validate_a2_evaluation
from scripts.validate_a21_scale_association import validate_output as validate_a21


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def receipt(project_root: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(project_root.resolve()).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def execute(*, project_root: Path, output_dir: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"public evidence output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_root = project_root / "runs/clio-apartment-gpu/d21_1-pillow-prompt-sweep-gpu"
    comparison = read_json(sweep_root / "comparison/comparison.json")
    comparison_validation = read_json(sweep_root / "comparison/validation.json")
    if comparison.get("status") != "PASS" or comparison_validation.get("status") != "PASS":
        raise ValueError("D21.1 sweep comparison is not independently validated")
    a2_root = project_root / "runs/clio-apartment-gpu/d21_1-dinosaur-pillow-association/a2"
    a2_eval_root = project_root / "runs/clio-apartment-gpu/d21_1-dinosaur-pillow-association/a2-evaluation"
    a21_root = project_root / "runs/clio-apartment-gpu/d21_1-dinosaur-pillow-association/a21"
    a2_report = validate_a2(a2_root)
    a2_eval_report = validate_a2_evaluation(a2_eval_root)
    a21_report = validate_a21(a21_root)
    if any(report["status"] != "PASS" for report in (a2_report, a2_eval_report, a21_report)):
        raise ValueError("D21.1 association bundle failed independent validation")
    a21_eval = read_json(a21_root / "evaluation.json")
    if a21_eval.get("status") != "PASS":
        raise ValueError("A2.1 evaluation is not PASS")
    alignment_path = project_root / "runs/clio-apartment-gpu/d21_1-pillow-audit/alignment_readiness.json"
    alignment = read_json(alignment_path)
    if validate_alignment_readiness(alignment, project_root=project_root)["status"] != "PASS":
        raise ValueError("alignment readiness report failed deterministic replay")

    sweep_summary = {
        "schema_version": "0.1",
        "status": "PASS",
        "stage": "D21.1-pillow-segmentation-diagnostic",
        "scene_id": "clio-apartment",
        "split_role": "development_diagnostic_only",
        "hardware": "NVIDIA GeForce RTX 4090",
        "candidate_frame_count": comparison["candidate_frame_count"],
        "experiments": [
            {
                key: row[key]
                for key in (
                    "experiment_id", "query", "sam_threshold", "role",
                    "status", "sam_instances", "lifted_instances",
                    "rejected_instances", "frames_with_masks",
                    "frames_with_lifted_observations", "artifact_contract",
                )
            }
            for row in comparison["experiments"]
        ],
        "set_comparison": comparison["set_comparison"],
        "acceptance": comparison["acceptance"],
        "visual_audit": {
            "dinosaur_prompt_masks_reviewed": 5,
            "all_reviewed_masks_same_physical_object": True,
            "physical_object_description": "green dinosaur pillow",
            "reviewed_frames": ["rgb_70", "rgb_80", "rgb_114", "rgb_50", "rgb_40"],
        },
        "claim_boundary": {
            "frame_visibility_labels": "PENDING_MANUAL_ANNOTATION",
            "segmentation_recall": None,
            "instance_specific_prompt_is_formal_policy": False,
            "low_threshold_extra_masks_accepted_as_ground_truth": False,
            "cubicle_accessed": False,
            "performance_claim": None,
        },
    }
    association_summary = {
        "schema_version": "0.1",
        "status": "PASS",
        "stage": "D21.1-pillow-association-diagnostic",
        "scene_id": "clio-apartment-dinosaur-pillow-050",
        "query": "dinosaur pillow",
        "input_observations": 5,
        "manual_label_scope": {
            "positive_pairs": 10,
            "negative_pairs": 0,
            "association_only": True,
        },
        "frozen_a2": {
            "validator_status": a2_report["status"],
            "gate_pass_pairs": a2_report["gate_pass_pairs"],
            "predicted_match_pairs": a2_report["predicted_match_pairs"],
            "permanent_objects": a2_report["permanent_objects"],
            "pending_observations": a2_report["pending_observations"],
            "true_positive": a2_eval_report["true_positive"],
            "false_positive": a2_eval_report["false_positive"],
            "false_negative": a2_eval_report["false_negative"],
        },
        "a21_development_candidate": {
            "validator_status": a21_report["status"],
            "gate_pass_pairs": a21_report["counts"]["gate_pass_pairs"],
            "predicted_match_pairs": a21_report["counts"]["predicted_match_pairs"],
            "permanent_objects": a21_report["counts"]["permanent_objects"],
            "pending_observations": a21_report["counts"]["pending_observations"],
            "true_positive": a21_eval["metrics"]["true_positive"],
            "false_positive": a21_eval["metrics"]["false_positive"],
            "false_negative": a21_eval["metrics"]["false_negative"],
        },
        "decision": {
            "formal_association_remains": "A2-evidence-aware-complete-link",
            "a21_upgraded_to_formal": False,
            "reason": "A2 and A2.1 produce the same correct one-object result on the five clean masks; no held-out or negative-pair evidence supports an upgrade.",
        },
        "claim_boundary": {
            "development_only": True,
            "no_negative_pairs": True,
            "general_association_performance": None,
            "cubicle_held_out_untouched": True,
        },
    }
    alignment_summary = {
        "schema_version": "0.1",
        "status": "PASS_WITH_ALIGNMENT_INPUTS_PENDING",
        "stage": "D16.2-clio-alignment-readiness-summary",
        "scene_id": "apartment",
        "database": {
            "sha256": alignment["source"]["database_sha256"],
            "bytes": alignment["source"]["database_bytes"],
            "integrity": alignment["database"]["integrity"],
            "image_count": alignment["database"]["image_count"],
            "images_with_prior_pose": alignment["database"]["images_with_prior_pose"],
        },
        "local_rgb": {
            "available_count": alignment["local_rgb"]["available_count"],
            "missing_count": alignment["local_rgb"]["missing_count"],
            "extra_count": alignment["local_rgb"]["extra_count"],
        },
        "pose_sources": alignment["pose_sources"],
        "alignment": alignment["alignment"],
        "claim_boundary": {
            "full_trajectory_reconstruction_completed": False,
            "metric_alignment_completed": False,
            "main_inference_reads_ground_truth": False,
        },
    }
    write_json(output_dir / "sweep_summary.json", sweep_summary)
    write_json(output_dir / "association_summary.json", association_summary)
    write_json(output_dir / "alignment_readiness_summary.json", alignment_summary)
    readme = """# D21.1 Clio pillow diagnostic\n\nThis lightweight bundle records a real RTX 4090 development diagnostic on the\n24-frame Clio `apartment` subset. It isolates SAM prompt/threshold sensitivity\nfrom 3D association. It is not a formal prompt-policy change, held-out evidence,\nsegmentation recall, Grounding Acc@1, or a performance-improvement claim.\n\nThe frozen `pillow@0.5` run bitwise replays the retained three masks in two\nframes. `dinosaur pillow@0.5` yields five visually reviewed masks in five other\nframes; all cover the same physical green dinosaur pillow. The task phrase\n`bring me a pillow` yields zero masks. Across all six experiments, only 8/24\nframes contain a mask; frame visibility is still awaiting manual annotation.\n\nWith the five clean instance-specific masks, frozen A2 and development A2.1 both\nform one permanent object and classify all ten positive pairs correctly. There\nare no negative pairs, so A2 remains the formal method and A2.1 is not promoted.\n\nThe recovered COLMAP database lists 1,845 RGB images, but only 786 RGB files are\nlocal and no sparse poses or rosbag are available. Full-trajectory evaluator-only\nSim(3) alignment therefore remains blocked. Raw data, masks, points, images and\nvideo are not redistributed.\n\nValidate with:\n\n```bash\npython -m scripts.validate_d21_1_pillow_diagnostic \\\n  evidence/week4/d21_1-pillow-diagnostic/public_report.json\n```\n"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    public_paths = [
        output_dir / "README.md",
        output_dir / "sweep_summary.json",
        output_dir / "association_summary.json",
        output_dir / "alignment_readiness_summary.json",
        project_root / "configs/clio_apartment_pillow_prompt_audit.json",
        project_root / "configs/clio_apartment_dinosaur_pillow_prompt_labels.json",
    ]
    local_paths = [
        sweep_root / "comparison/detected_union_contact_sheet.jpg",
        sweep_root / "sweep_execution.json",
        sweep_root / "experiments/dinosaur-pillow-050/d6_result.json",
        project_root / "runs/clio-apartment-gpu/d21_1-dinosaur-pillow-d7/stage_video.mp4",
        a2_root / "a2_result.json",
        a21_root / "a21_result.json",
        alignment_path,
    ]
    report = {
        "schema_version": "d21.1-pillow-diagnostic/0.1",
        "status": "PASS",
        "stage": "D21.1-clio-apartment-pillow-diagnostic",
        "scope": {
            "scene_id": "apartment",
            "role": "development_diagnostic_only",
            "base_query": "pillow",
            "held_out_scene": "cubicle",
            "held_out_accessed": False,
            "result_scope": "real_gpu_failure_diagnosis_not_performance",
        },
        "headline": {
            "candidate_frames": 24,
            "baseline_masks": 3,
            "baseline_evidence_frames": 2,
            "all_prompt_union_frames": 8,
            "clean_dinosaur_prompt_masks": 5,
            "clean_dinosaur_prompt_frames": 5,
            "frozen_a2_permanent_objects": 1,
            "a21_permanent_objects": 1,
            "formal_method_changed": False,
        },
        "findings": [
            "SAM_PROMPT_SENSITIVITY_IS_THE_PRIMARY_OBSERVED_FAILURE",
            "TASK_PHRASE_RETURNED_ZERO_MASKS",
            "INSTANCE_SPECIFIC_PROMPT_RECOVERED_FIVE_CLEAN_COMPLEMENTARY_FRAMES",
            "LOWER_THRESHOLDS_ADDED_LOCAL_OR_QUESTIONABLE_MASKS",
            "FROZEN_A2_SUCCEEDS_WHEN_MASKS_ARE_CROSS_VIEW_CONSISTENT",
            "A21_NOT_PROMOTED_BECAUSE_IT_ADDS_NO_RESULT_ON_CLEAN_MASKS",
            "FRAME_LEVEL_SEGMENTATION_RECALL_PENDING_MANUAL_VISIBILITY_LABELS",
            "FULL_TRAJECTORY_SIM3_BLOCKED_MISSING_POSE_INPUTS",
            "CLIO_CUBICLE_UNTOUCHED",
        ],
        "artifacts": {
            "public": [receipt(project_root, path) for path in public_paths],
            "local_not_in_git": [receipt(project_root, path) for path in local_paths],
        },
        "claim_boundary": {
            "instance_specific_prompt_is_formal_policy": False,
            "segmentation_recall": None,
            "grounding_acc_at_1": None,
            "held_out_performance": None,
            "performance_improvement": None,
            "dataset_redistribution": False,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "public_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument(
        "--output-dir",
        default=str(root / "evidence/week4/d21_1-pillow-diagnostic"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = execute(
        project_root=Path(args.project_root), output_dir=Path(args.output_dir)
    )
    print(json.dumps({
        "status": report["status"],
        "stage": report["stage"],
        "public_artifacts": len(report["artifacts"]["public"]),
        "local_receipts": len(report["artifacts"]["local_not_in_git"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
