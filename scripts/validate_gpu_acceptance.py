"""Validate the local GPU follow-up bundle without rewriting day evidence.

This validator is intentionally additive.  It reuses the frozen D3--D15
validators/contracts, proves that the formerly partial D11 cache was completed
without changing its retained outcomes, and emits one compact acceptance report.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from adapters.open_vocab import PE_SOURCE_COMMIT, SAM3_SOURCE_COMMIT
from relground.candidate_cache import CandidateOutcomeCache
from relground.observation_cache import sha256_file
from relground.q1_fixed_topk import validate_prediction_payload
from relground.q2_sequential import validate_trace_payload
from scripts.validate_a2_association import validate_output as validate_a2
from scripts.validate_candidate_cache import validate_output as validate_d11
from scripts.validate_d6 import validate_output as validate_d6
from scripts.validate_d7_cache import validate_output as validate_d7
from scripts.validate_d8_memory import validate_output as validate_d8
from scripts.validate_multiview_geometry import validate_multiview_geometry
from scripts.validate_single_view_baselines import (
    validate_output as validate_single_view,
)
from scripts.validate_topk_retrieval import validate_output as validate_d5


GPU_ACCEPTANCE = "COMPLETE"
EVALUATION_SCOPE = "ENGINEERING_REPLAY_NO_NEW_MANUAL_LABELS"
EXPECTED_TOP1 = "frame_0001"
EXPECTED_CANDIDATES = 8
EXPECTED_PROJECT_SUBDIRECTORIES = {
    "d5": "d5-all-candidates",
    "d6": "d6-all-candidates",
    "d7": "d7-all-candidates",
    "d8": "d8-all-candidates",
    "d11": "d11-complete-cache",
    "d12": "d12-a2-prediction",
    "d13": "d13-q0-single-view",
    "d14": "d14-complete-replay",
    "d15": "d15-complete-replay",
}
FORBIDDEN_EVALUATION_KEYS = {
    "ground_truth",
    "labels",
    "pair_labels",
    "expected_same",
    "answer",
    "metrics",
    "precision",
    "recall",
    "f1",
    "instance_id",
    "error_type",
}
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} root must be an object")
    return payload


def _inside(root: Path, path: Path, name: str) -> Path:
    boundary = root.resolve()
    resolved = path.resolve()
    if resolved != boundary and boundary not in resolved.parents:
        raise ValueError(f"{name} escapes the GPU bundle")
    return resolved


def _bundle_dir(root: Path, name: str) -> Path:
    path = _inside(root, root / EXPECTED_PROJECT_SUBDIRECTORIES[name], name)
    if not path.is_dir():
        raise ValueError(f"missing GPU bundle directory: {path.name}")
    return path


def _resolve_reference(bundle_root: Path, owner: Path, reference: Any) -> Path:
    relative = Path(str(reference))
    if relative.is_absolute():
        raise ValueError("artifact reference must be relative")
    path = _inside(bundle_root, owner / relative, "artifact reference")
    if not path.is_file():
        raise ValueError(f"missing referenced artifact: {reference}")
    return path


def _manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise ValueError(f"{path} config must be an object")
    return manifest


def _positive_vram(manifest: Mapping[str, Any], stage: str) -> float:
    try:
        value = float(manifest["peak_vram_mb"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{stage} does not record peak_vram_mb") from error
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{stage} peak_vram_mb must be finite and positive")
    return value


def _commit(value: Any, name: str) -> str:
    text = str(value).lower()
    if COMMIT_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{name} is not a full Git commit")
    return text


def _forbidden_paths(payload: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(payload, Mapping):
        for raw_key, value in payload.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            if key.lower() in FORBIDDEN_EVALUATION_KEYS:
                found.append(path)
            found.extend(_forbidden_paths(value, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.extend(_forbidden_paths(value, f"{prefix}[{index}]"))
    return found


def _require_no_evaluation_files(path: Path, stage: str) -> None:
    forbidden = [
        item.name
        for item in path.rglob("*")
        if item.is_file()
        and any(token in item.name.lower() for token in ("label", "evaluation"))
    ]
    if forbidden:
        raise ValueError(
            f"{stage} unexpectedly contains labels/evaluation files: "
            + ", ".join(sorted(forbidden))
        )


def _normalize_rank_provenance(outcome: Mapping[str, Any]) -> dict[str, Any]:
    """Ignore only selection-rank provenance when comparing two outcomes.

    The old cache came from hybrid Top-K (frame_0021 selection rank 4), while
    the complete materialization uses raw Top-8 (source rank 8).  The rank is
    checked against each containing candidate before being normalized; all SAM,
    lifting, rejection, and 3D observation content remains byte-for-byte equal.
    """

    normalized = copy.deepcopy(dict(outcome))
    for row in normalized.get("observations", []):
        if isinstance(row, dict):
            metadata = row.get("metadata")
            if isinstance(metadata, dict) and "selected_rank" in metadata:
                metadata["selected_rank"] = "<candidate-rank>"
    for row in normalized.get("rejections", []):
        if isinstance(row, dict) and "selected_rank" in row:
            row["selected_rank"] = "<candidate-rank>"
    return normalized


def _assert_outcome_rank(candidate: Mapping[str, Any], expected_rank: int) -> None:
    outcome = candidate.get("outcome")
    if not isinstance(outcome, Mapping):
        return
    for collection in ("observations", "rejections"):
        for row in outcome.get(collection, []):
            if not isinstance(row, Mapping):
                continue
            values: list[Any] = []
            if "selected_rank" in row:
                values.append(row["selected_rank"])
            metadata = row.get("metadata")
            if isinstance(metadata, Mapping) and "selected_rank" in metadata:
                values.append(metadata["selected_rank"])
            if any(int(value) != expected_rank for value in values):
                raise ValueError(
                    f"{candidate['frame_id']} outcome selected_rank is inconsistent"
                )


def _outcome_selected_ranks(candidate: Mapping[str, Any]) -> list[int]:
    outcome = candidate.get("outcome")
    values: set[int] = set()
    if not isinstance(outcome, Mapping):
        return []
    for collection in ("observations", "rejections"):
        for row in outcome.get(collection, []):
            if not isinstance(row, Mapping):
                continue
            if "selected_rank" in row:
                values.add(int(row["selected_rank"]))
            metadata = row.get("metadata")
            if isinstance(metadata, Mapping) and "selected_rank" in metadata:
                values.add(int(metadata["selected_rank"]))
    return sorted(values)


def compare_retained_partial(
    retained_cache_path: Path,
    complete_cache_path: Path,
) -> dict[str, Any]:
    """Prove completion of the old cache without changing retained outcomes."""

    old = CandidateOutcomeCache.load(retained_cache_path).to_dict()
    new = CandidateOutcomeCache.load(complete_cache_path).to_dict()
    if old["materialization_status"] != "partial":
        raise ValueError("retained D11 cache is no longer partial")
    if new["materialization_status"] != "complete":
        raise ValueError("GPU D11 cache is not complete")
    if old["candidate_universe"] != new["candidate_universe"]:
        raise ValueError("D11 candidate universe/order changed")

    old_by_frame = {row["frame_id"]: row for row in old["candidates"]}
    new_by_frame = {row["frame_id"]: row for row in new["candidates"]}
    old_d6 = _read_json(retained_cache_path.parent / "source_d6_result.json")
    new_d6 = _read_json(complete_cache_path.parent / "source_d6_result.json")
    old_selection_ranks = {
        str(row["frame_id"]): int(row["rank"])
        for row in old_d6["selected_frames"]
    }
    new_selection_ranks = {
        str(row["frame_id"]): int(row["rank"])
        for row in new_d6["selected_frames"]
    }
    retained_available: list[str] = []
    newly_materialized: list[str] = []
    rank_provenance_changes: list[dict[str, Any]] = []
    for frame_id in old["candidate_universe"]:
        before = old_by_frame[frame_id]
        after = new_by_frame[frame_id]
        for key in (
            "frame_id",
            "geometry_index",
            "image_ref",
            "image_sha256",
            "camera_center",
            "view_direction",
            "retrieval_score",
            "retrieval_cosine",
        ):
            if before[key] != after[key]:
                raise ValueError(f"retained candidate metadata changed: {frame_id}.{key}")
        if before["outcome_status"] == "available":
            retained_available.append(frame_id)
            if after["outcome_status"] != "available":
                raise ValueError(f"retained available outcome disappeared: {frame_id}")
            if frame_id not in old_selection_ranks or frame_id not in new_selection_ranks:
                raise ValueError(f"missing D6 selection provenance: {frame_id}")
            _assert_outcome_rank(before, old_selection_ranks[frame_id])
            _assert_outcome_rank(after, new_selection_ranks[frame_id])
            if _outcome_selected_ranks(before) != _outcome_selected_ranks(after):
                rank_provenance_changes.append({
                    "frame_id": frame_id,
                    "retained_selection_rank": old_selection_ranks[frame_id],
                    "complete_source_rank": new_selection_ranks[frame_id],
                })
            if _normalize_rank_provenance(before["outcome"]) != (
                _normalize_rank_provenance(after["outcome"])
            ):
                raise ValueError(f"retained available outcome drifted: {frame_id}")
        elif before["outcome_status"] == "unmaterialized":
            newly_materialized.append(frame_id)
            if after["outcome_status"] != "available":
                raise ValueError(f"candidate was not materialized: {frame_id}")
            if frame_id not in new_selection_ranks:
                raise ValueError(f"missing new D6 selection provenance: {frame_id}")
            _assert_outcome_rank(after, new_selection_ranks[frame_id])
        else:
            raise ValueError(
                f"unexpected retained candidate status: {frame_id}="
                f"{before['outcome_status']}"
            )
    if len(retained_available) != 4 or len(newly_materialized) != 4:
        raise ValueError("retained D11 cache must contain the frozen 4+4 split")
    expected_rank_change = [{
        "frame_id": "frame_0021",
        "retained_selection_rank": 4,
        "complete_source_rank": 8,
    }]
    if rank_provenance_changes != expected_rank_change:
        raise ValueError(
            "selection-rank provenance change is not the documented "
            "frame_0021 4-to-8 transition"
        )

    old_retrieval = _read_json(retained_cache_path.parent / "source_d5_retrieval.json")
    new_retrieval = _read_json(complete_cache_path.parent / "source_d5_retrieval.json")
    old_ranking = [
        (row["frame_id"], row["retrieval_score"], row["retrieval_cosine"])
        for row in old_retrieval["raw_ranking"]
    ]
    new_ranking = [
        (row["frame_id"], row["retrieval_score"], row["retrieval_cosine"])
        for row in new_retrieval["raw_ranking"]
    ]
    if old_ranking != new_ranking:
        raise ValueError("D5 raw ranking frame order/scores changed")
    return {
        "retained_available_unchanged": retained_available,
        "newly_materialized": newly_materialized,
        "raw_ranking_unchanged": True,
        "normalized_fields": [
            "observations[].metadata.selected_rank",
            "rejections[].selected_rank",
        ],
        "rank_provenance_changes": rank_provenance_changes,
    }


def _require_report_pass(report: Mapping[str, Any], stage: str) -> None:
    if report.get("status") != "PASS":
        details = report.get("failures", report.get("errors", []))
        raise ValueError(f"{stage} validator failed: {details}")


def validate_output(
    bundle: str | Path,
    *,
    project_root: str | Path | None = None,
    retained_partial_cache: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(bundle).resolve()
    repository = Path(project_root or Path.cwd()).resolve()
    retained = Path(
        retained_partial_cache
        or repository / "evidence/week2/d11-candidate-cache/candidate_cache.json"
    ).resolve()
    failures: list[str] = []
    checks: dict[str, Any] = {}

    try:
        if not root.is_dir():
            raise ValueError(f"GPU bundle is missing: {root}")
        paths = {name: _bundle_dir(root, name) for name in EXPECTED_PROJECT_SUBDIRECTORIES}

        reports = {
            "D3": validate_multiview_geometry(root / "geometry.npz"),
            "D5": validate_d5(paths["d5"]),
            "D6": validate_d6(paths["d6"]),
            "D7": validate_d7(paths["d7"]),
            "D8": validate_d8(paths["d8"]),
            "D11": validate_d11(paths["d11"], project_root=repository),
            "D12": validate_a2(paths["d12"]),
            "D13": validate_single_view(paths["d13"]),
        }
        for stage, report in reports.items():
            _require_report_pass(report, stage)
        checks["validators"] = {
            stage: report["status"] for stage, report in reports.items()
        }
        checks["D6_counts"] = {
            "selected_frames": len(reports["D6"]["selected_frames"]),
            "sam_instances": reports["D6"]["mask_instances"],
            "lifted_instances": reports["D6"]["lifted_instances"],
            "rejected_instances": reports["D6"]["rejected_instances"],
        }
        checks["D7_counts"] = {
            "frames": reports["D7"]["frames"],
            "observations": reports["D7"]["observations"],
            "points": reports["D7"]["points"],
        }

        geometry_manifest = _manifest(root / "run_manifest.json")
        d5_manifest = _manifest(paths["d5"] / "run_manifest.json")
        d6_manifest = _manifest(paths["d6"] / "run_manifest.json")
        d13_manifest = _manifest(paths["d13"] / "run_manifest.json")
        checks["peak_vram_mb"] = {
            "D3": _positive_vram(geometry_manifest, "D3"),
            "D5": _positive_vram(d5_manifest, "D5"),
            "D6": _positive_vram(d6_manifest, "D6"),
            "D13": _positive_vram(d13_manifest, "D13"),
        }

        geometry_config = geometry_manifest["config"]
        d5_config = d5_manifest["config"]
        d6_config = d6_manifest["config"]
        d13_config = d13_manifest["config"]
        geometry_commits = {
            "vggt_slam": _commit(geometry_config["upstream_commit"], "VGGT-SLAM commit"),
            "vggt": _commit(geometry_config["vggt_commit"], "VGGT commit"),
            "salad": _commit(geometry_config["salad_commit"], "SALAD commit"),
        }
        d5_commits = dict(d5_config["source_commits"])
        d6_commits = dict(d6_config["source_commits"])
        d13_commits = dict(d13_config["source_commits"])
        if d5_commits != {"perception_models": PE_SOURCE_COMMIT}:
            raise ValueError("D5 does not use the pinned PE source commit")
        if d6_commits != {"sam3": SAM3_SOURCE_COMMIT}:
            raise ValueError("D6 does not use the pinned SAM3 source commit")
        if d13_commits != {
            "perception_models": PE_SOURCE_COMMIT,
            "sam3": SAM3_SOURCE_COMMIT,
        }:
            raise ValueError("D13 source commits differ from D5/D6 pins")
        project_sha = _commit(d5_manifest["git_sha"], "D5 project commit")
        if _commit(d6_manifest["git_sha"], "D6 project commit") != project_sha:
            raise ValueError("D5/D6 project commits differ")
        if _commit(d13_manifest["git_sha"], "D13 project commit") != project_sha:
            raise ValueError("D5/D13 project commits differ")
        if _commit(geometry_config["project_git_sha"], "D3 project commit") != project_sha:
            raise ValueError("D3 project commit differs from D5/D6/D13")
        checks["source_commits"] = {
            **geometry_commits,
            "perception_models": PE_SOURCE_COMMIT,
            "sam3": SAM3_SOURCE_COMMIT,
            "project": project_sha,
        }

        d6_result = _read_json(paths["d6"] / "d6_result.json")
        frozen_lifter = {
            "confidence_threshold": 0.5,
            "min_points": 30,
            "outlier_mad_scale": 3.5,
        }
        if d6_result.get("sam_threshold") != 0.5:
            raise ValueError("D6 SAM threshold is not frozen at 0.5")
        if d6_result.get("lifter_config") != frozen_lifter:
            raise ValueError("D6 lifter configuration changed")
        if d6_result.get("mask_resizing_after_sam") is not False:
            raise ValueError("D6 resized masks after SAM")
        if d6_config.get("sam_threshold") != 0.5:
            raise ValueError("D6 manifest SAM threshold changed")
        if d6_config.get("lifter_config") != frozen_lifter:
            raise ValueError("D6 manifest lifter configuration changed")
        if any(
            row.get("mask_resizing_after_sam") is not False
            for row in d6_result.get("processed_frames", [])
        ):
            raise ValueError("a D6 processed frame resized its SAM mask")
        checks["frozen_d6"] = {
            "sam_threshold": 0.5,
            "confidence_threshold": 0.5,
            "min_points": 30,
            "outlier_mad_scale": 3.5,
            "mask_resizing_after_sam": False,
        }

        d5_result = _read_json(paths["d5"] / "retrieval.json")
        raw_top1 = d5_result["raw_ranking"][0]["frame_id"]
        upstream_top1 = d5_result["upstream_top1"]["frame_id"]
        if raw_top1 != EXPECTED_TOP1 or upstream_top1 != EXPECTED_TOP1:
            raise ValueError("D5 raw/upstream Top-1 is not frame_0001")
        checks["D5_top1"] = {
            "raw_ranking_first": raw_top1,
            "upstream_top1": upstream_top1,
        }

        complete_cache_path = paths["d11"] / "candidate_cache.json"
        cache = CandidateOutcomeCache.load(complete_cache_path).to_dict()
        if cache["materialization_status"] != "complete":
            raise ValueError("D11 GPU cache is not complete")
        if cache["counts"]["candidate_count"] != EXPECTED_CANDIDATES:
            raise ValueError("D11 candidate count is not 8")
        if cache["counts"]["available_candidates"] != EXPECTED_CANDIDATES:
            raise ValueError("D11 available count is not 8")
        if cache["counts"]["unmaterialized_candidates"] != 0:
            raise ValueError("D11 still contains unmaterialized candidates")
        if cache["inference_config"] != {
            "retrieval_policy": cache["inference_config"]["retrieval_policy"],
            "sam_threshold": 0.5,
            "lifter_config": frozen_lifter,
            "mask_resizing_after_sam": False,
        }:
            raise ValueError("D11 cache inference configuration is not frozen")
        checks["D11_completion"] = {
            "candidate_count": EXPECTED_CANDIDATES,
            "available_candidates": EXPECTED_CANDIDATES,
            "materialization_status": "complete",
            "total_observations": cache["counts"]["total_observations"],
            "total_rejections": cache["counts"]["total_rejections"],
        }
        checks["retained_partial_crosscheck"] = compare_retained_partial(
            retained,
            complete_cache_path,
        )

        _require_no_evaluation_files(paths["d12"], "D12")
        d12_result = _read_json(paths["d12"] / "a2_result.json")
        leaked = _forbidden_paths(d12_result)
        if leaked:
            raise ValueError("D12 prediction leaks evaluation fields: " + ", ".join(leaked))
        checks["D12_prediction"] = {
            "status": "PASS",
            "manual_labels_used": False,
            "evaluation_run": False,
        }

        single_view = _read_json(paths["d13"] / "single_view_result.json")
        if single_view.get("top1", {}).get("frame_id") != EXPECTED_TOP1:
            raise ValueError("D13 Q0 Top-1 is not frame_0001")
        baseline_counts = reports["D13"].get("lifted_instances", {})
        if set(baseline_counts) != {"B0-official", "B1-robust-single-view"}:
            raise ValueError("D13 does not contain the frozen B0/B1 pair")
        if any(int(value) < 1 for value in baseline_counts.values()):
            raise ValueError("D13 B0/B1 did not both produce a binary PASS outcome")
        checks["D13_Q0"] = {
            "top1_frame": EXPECTED_TOP1,
            "B0-official": "PASS",
            "B1-robust-single-view": "PASS",
        }

        _require_no_evaluation_files(paths["d14"], "D14")
        q1_path = paths["d14"] / "real_prediction.json"
        q1 = _read_json(q1_path)
        q1_cache_path = _resolve_reference(
            root, paths["d14"], q1["source"]["candidate_cache"]
        )
        if q1_cache_path != complete_cache_path.resolve():
            raise ValueError("D14 does not reference the GPU complete cache")
        if sha256_file(q1_cache_path) != q1["source"]["candidate_cache_sha256"]:
            raise ValueError("D14 candidate-cache hash changed")
        validate_prediction_payload(q1, cache)
        if q1.get("status") != "PASS":
            raise ValueError("D14 prediction is not PASS")
        if q1["source"].get("cache_materialization_status") != "complete":
            raise ValueError("D14 did not replay a complete cache")
        if not q1["acceptance"].get("k1_matches_q0"):
            raise ValueError("D14 K=1 does not match Q0")
        if q1["curves"][0]["selected_frames"][0]["frame_id"] != EXPECTED_TOP1:
            raise ValueError("D14 K=1 frame is not Q0 frame_0001")
        leaked = _forbidden_paths(q1)
        if leaked:
            raise ValueError("D14 prediction leaks evaluation fields: " + ", ".join(leaked))
        checks["D14_prediction"] = {
            "status": "PASS",
            "complete_cache": True,
            "k1_matches_q0": True,
            "manual_labels_used": False,
            "evaluation_run": False,
        }

        q2_path = paths["d15"] / "real_trace.json"
        q2 = _read_json(q2_path)
        q2_cache_path = _resolve_reference(
            root, paths["d15"], q2["source"]["candidate_cache"]
        )
        if q2_cache_path != complete_cache_path.resolve():
            raise ValueError("D15 does not reference the GPU complete cache")
        if sha256_file(q2_cache_path) != q2["source"]["candidate_cache_sha256"]:
            raise ValueError("D15 candidate-cache hash changed")
        validate_trace_payload(q2, cache)
        summary = q2["summary"]
        if q2.get("status") != "PASS":
            raise ValueError("D15 trace is not PASS/complete")
        if q2["source"].get("cache_materialization_status") != "complete":
            raise ValueError("D15 did not replay a complete cache")
        if not summary.get("budget1_matches_q0"):
            raise ValueError("D15 budget=1 does not match Q0")
        if summary.get("q0_top1_frame") != EXPECTED_TOP1:
            raise ValueError("D15 Q0 Top-1 is not frame_0001")
        if summary.get("performance_claim") is not None:
            raise ValueError("D15 publishes an unsupported performance claim")
        if summary.get("selected_count") != 5:
            raise ValueError("D15 did not consume the frozen maximum budget of 5")
        if summary.get("stop_reason") != "max_budget_reached":
            raise ValueError("D15 did not stop at max_budget_reached")
        leaked = _forbidden_paths(q2)
        if leaked:
            raise ValueError("D15 trace leaks evaluation fields: " + ", ".join(leaked))
        checks["D15_trace"] = {
            "status": "PASS",
            "complete_cache": True,
            "budget1_matches_q0": True,
            "performance_claim": None,
            "selected_frames": summary["selected_frames"],
            "selected_count": summary["selected_count"],
            "stop_reason": summary["stop_reason"],
            "observed_gain": summary["cumulative_cost"]["observed_gain"],
        }
    except (
        IndexError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        failures.append(str(error))

    passed = not failures
    return {
        "status": "PASS" if passed else "FAIL",
        "gpu_acceptance": GPU_ACCEPTANCE if passed else "INCOMPLETE",
        "evaluation_scope": EVALUATION_SCOPE,
        "bundle": str(root),
        "retained_partial_cache": str(retained),
        "checks": checks,
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--retained-partial-cache")
    parser.add_argument("--report")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = validate_output(
        args.bundle,
        project_root=args.project_root,
        retained_partial_cache=args.retained_partial_cache,
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
