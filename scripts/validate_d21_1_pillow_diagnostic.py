"""Validate lightweight public evidence for the D21.1 pillow diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from relground.segmentation_sweep import sha256_file


def contained(project_root: Path, reference: str) -> Path:
    relative = Path(reference)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("D21.1 evidence paths must be repository-relative")
    root = project_root.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError("D21.1 evidence path escapes repository")
    return path


def has_absolute_path(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(has_absolute_path(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(has_absolute_path(item) for item in value)
    return isinstance(value, str) and Path(value).is_absolute()


def artifact_by_name(project_root: Path, artifacts: list[Mapping[str, Any]], name: str) -> Path:
    matches = [item for item in artifacts if Path(str(item["path"])).name == name]
    if len(matches) != 1:
        raise ValueError(f"expected one public artifact named {name}")
    return contained(project_root, str(matches[0]["path"]))


def validate(
    report_path: str | Path,
    *,
    project_root: str | Path,
    verify_local: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    checks["contract"] = (
        report.get("schema_version") == "d21.1-pillow-diagnostic/0.1"
        and report.get("status") == "PASS"
        and report.get("stage") == "D21.1-clio-apartment-pillow-diagnostic"
    )
    scope = report.get("scope", {})
    checks["split_guard"] = (
        scope.get("scene_id") == "apartment"
        and scope.get("role") == "development_diagnostic_only"
        and scope.get("held_out_scene") == "cubicle"
        and scope.get("held_out_accessed") is False
        and scope.get("result_scope") == "real_gpu_failure_diagnosis_not_performance"
    )
    checks["portable_paths"] = not has_absolute_path(report)
    public = report.get("artifacts", {}).get("public", [])
    public_ok = isinstance(public, list) and len(public) == 6
    if public_ok:
        for artifact in public:
            try:
                path = contained(root, str(artifact["path"]))
                public_ok = (
                    public_ok
                    and path.suffix in {".json", ".md"}
                    and path.is_file()
                    and path.stat().st_size == int(artifact["bytes"])
                    and sha256_file(path) == artifact["sha256"]
                )
            except (KeyError, OSError, TypeError, ValueError):
                public_ok = False
    checks["public_artifact_hashes"] = public_ok

    try:
        sweep = json.loads(artifact_by_name(root, public, "sweep_summary.json").read_text())
        association = json.loads(artifact_by_name(root, public, "association_summary.json").read_text())
        alignment = json.loads(artifact_by_name(root, public, "alignment_readiness_summary.json").read_text())
        experiments = {row["experiment_id"]: row for row in sweep["experiments"]}
        baseline = experiments["baseline-pillow-050"]
        task = experiments["task-phrase-050"]
        article = experiments["article-pillow-050"]
        dinosaur = experiments["dinosaur-pillow-050"]
        threshold_04 = experiments["baseline-pillow-040"]
        threshold_03 = experiments["baseline-pillow-030"]
        checks["gpu_sweep"] = (
            sweep["status"] == "PASS"
            and sweep["candidate_frame_count"] == 24
            and len(experiments) == 6
            and baseline["sam_instances"] == 3
            and baseline["frames_with_masks"] == ["rgb_128", "rgb_90"]
            and task["sam_instances"] == 0
            and article["sam_instances"] == 3
            and dinosaur["sam_instances"] == 5
            and len(dinosaur["frames_with_masks"]) == 5
            and threshold_04["sam_instances"] == 6
            and threshold_03["sam_instances"] == 8
            and sweep["set_comparison"]["all_experiments_union_count"] == 8
            and sweep["acceptance"]["baseline_bitwise_replayed"] is True
        )
        visual = sweep["visual_audit"]
        checks["visual_audit_boundary"] = (
            visual["dinosaur_prompt_masks_reviewed"] == 5
            and visual["all_reviewed_masks_same_physical_object"] is True
            and sweep["claim_boundary"]["instance_specific_prompt_is_formal_policy"] is False
            and sweep["claim_boundary"]["segmentation_recall"] is None
        )
        frozen = association["frozen_a2"]
        candidate = association["a21_development_candidate"]
        checks["association_diagnostic"] = (
            association["status"] == "PASS"
            and association["input_observations"] == 5
            and association["manual_label_scope"] == {
                "positive_pairs": 10,
                "negative_pairs": 0,
                "association_only": True,
            }
            and frozen["validator_status"] == "PASS"
            and frozen["predicted_match_pairs"] == 10
            and frozen["permanent_objects"] == 1
            and frozen["pending_observations"] == 0
            and candidate["validator_status"] == "PASS"
            and candidate["predicted_match_pairs"] == 10
            and candidate["permanent_objects"] == 1
            and association["decision"]["a21_upgraded_to_formal"] is False
            and association["decision"]["formal_association_remains"] == "A2-evidence-aware-complete-link"
        )
        checks["alignment_boundary"] = (
            alignment["status"] == "PASS_WITH_ALIGNMENT_INPUTS_PENDING"
            and alignment["database"]["integrity"] == "PASS"
            and alignment["database"]["image_count"] == 1845
            and alignment["database"]["images_with_prior_pose"] == 0
            and alignment["local_rgb"]["available_count"] == 786
            and alignment["local_rgb"]["missing_count"] == 1059
            and alignment["alignment"]["readiness"] == "BLOCKED_MISSING_SPARSE_OR_ROSBAG_POSES"
            and alignment["alignment"]["main_inference_may_read_gt"] is False
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        checks["gpu_sweep"] = False
        checks["visual_audit_boundary"] = False
        checks["association_diagnostic"] = False
        checks["alignment_boundary"] = False

    headline = report.get("headline", {})
    checks["headline_consistent"] = (
        headline.get("candidate_frames") == 24
        and headline.get("baseline_masks") == 3
        and headline.get("all_prompt_union_frames") == 8
        and headline.get("clean_dinosaur_prompt_masks") == 5
        and headline.get("frozen_a2_permanent_objects") == 1
        and headline.get("a21_permanent_objects") == 1
        and headline.get("formal_method_changed") is False
    )
    boundary = report.get("claim_boundary", {})
    checks["claim_boundary"] = (
        boundary.get("instance_specific_prompt_is_formal_policy") is False
        and boundary.get("segmentation_recall") is None
        and boundary.get("grounding_acc_at_1") is None
        and boundary.get("held_out_performance") is None
        and boundary.get("performance_improvement") is None
        and boundary.get("dataset_redistribution") is False
    )
    findings = set(report.get("findings", []))
    checks["failure_findings"] = {
        "SAM_PROMPT_SENSITIVITY_IS_THE_PRIMARY_OBSERVED_FAILURE",
        "TASK_PHRASE_RETURNED_ZERO_MASKS",
        "FROZEN_A2_SUCCEEDS_WHEN_MASKS_ARE_CROSS_VIEW_CONSISTENT",
        "A21_NOT_PROMOTED_BECAUSE_IT_ADDS_NO_RESULT_ON_CLEAN_MASKS",
        "FRAME_LEVEL_SEGMENTATION_RECALL_PENDING_MANUAL_VISIBILITY_LABELS",
        "FULL_TRAJECTORY_SIM3_BLOCKED_MISSING_POSE_INPUTS",
        "CLIO_CUBICLE_UNTOUCHED",
    }.issubset(findings)
    if verify_local:
        local = report.get("artifacts", {}).get("local_not_in_git", [])
        local_ok = isinstance(local, list) and len(local) == 7
        if local_ok:
            for artifact in local:
                try:
                    path = contained(root, str(artifact["path"]))
                    local_ok = (
                        local_ok
                        and path.is_file()
                        and path.stat().st_size == int(artifact["bytes"])
                        and sha256_file(path) == artifact["sha256"]
                    )
                except (KeyError, OSError, TypeError, ValueError):
                    local_ok = False
        checks["local_artifact_hashes"] = local_ok
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "schema_version": "0.1",
        "status": status,
        "stage": "D21.1-pillow-diagnostic-validation",
        "checks": checks,
        "scope": "development_failure_diagnosis_not_performance",
        "held_out_status": "CLIO_CUBICLE_UNTOUCHED",
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--verify-local", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = validate(
        args.report,
        project_root=args.project_root,
        verify_local=args.verify_local,
    )
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
