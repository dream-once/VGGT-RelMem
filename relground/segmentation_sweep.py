"""Contracts for a label-free SAM prompt/threshold development sweep."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROMPT_SWEEP_SCHEMA_VERSION = "0.1"
PROMPT_SWEEP_STAGE = "D21.1-segmentation-prompt-sweep"
CONFIG_FIELDS = (
    "schema_version", "scene_id", "split_role", "source_task",
    "candidate_universe", "experiments", "guards",
)
EXPERIMENT_FIELDS = (
    "experiment_id", "query", "sam_threshold", "role",
)
PLAN_EXPERIMENT_FIELDS = (
    *EXPERIMENT_FIELDS, "derived_selection_ref", "derived_selection_sha256",
    "output_ref",
)
GUARD_FIELDS = (
    "q0_threshold_remains_0_5", "no_cubicle_access_or_tuning",
    "same_geometry_and_candidate_universe",
    "formal_prompt_policy_must_not_use_image_specific_description",
)
PLAN_FIELDS = (
    "schema_version", "status", "stage", "scene_id", "split_role",
    "source", "guards", "candidate_frame_ids", "experiments",
    "claim_boundary", "created_at",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def project_reference(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"prompt-sweep artifact is outside project root: {path}") from error


def resolve_reference(project_root: Path, reference: str) -> Path:
    relative = Path(reference)
    if relative.is_absolute():
        raise ValueError("prompt-sweep references must be relative")
    root = project_root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("prompt-sweep reference escapes project root") from error
    return candidate


def validate_prompt_config(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if tuple(payload) != CONFIG_FIELDS:
        raise ValueError("prompt-sweep config fields are not frozen")
    if payload["schema_version"] != PROMPT_SWEEP_SCHEMA_VERSION:
        raise ValueError("unsupported prompt-sweep config schema")
    if payload["scene_id"] != "clio-apartment":
        raise ValueError("prompt sweep is frozen to apartment development")
    if payload["split_role"] != "development_calibration_only":
        raise ValueError("prompt sweep must be development-only")
    guards = payload["guards"]
    if not isinstance(guards, Mapping) or tuple(guards) != GUARD_FIELDS:
        raise ValueError("prompt-sweep guards are not frozen")
    if not all(value is True for value in guards.values()):
        raise ValueError("all prompt-sweep leakage guards must be true")
    rows = payload["experiments"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("prompt sweep requires experiments")
    normalized: list[dict[str, Any]] = []
    ids: set[str] = set()
    for raw in rows:
        if not isinstance(raw, Mapping) or tuple(raw) != EXPERIMENT_FIELDS:
            raise ValueError("prompt-sweep experiment fields are not frozen")
        experiment_id = str(raw["experiment_id"]).strip()
        query = str(raw["query"]).strip()
        threshold = float(raw["sam_threshold"])
        role = str(raw["role"]).strip()
        if not experiment_id or experiment_id in ids or not query or not role:
            raise ValueError("experiment ids must be unique and text non-empty")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("SAM threshold must be in [0, 1]")
        ids.add(experiment_id)
        normalized.append({
            "experiment_id": experiment_id,
            "query": query,
            "sam_threshold": threshold,
            "role": role,
        })
    baselines = [
        row for row in normalized
        if row["role"] == "frozen_upstream_threshold_baseline"
    ]
    if len(baselines) != 1 or baselines[0]["query"] != "pillow" or baselines[0]["sam_threshold"] != 0.5:
        raise ValueError("the frozen baseline must be exactly pillow@0.5")
    for row in normalized:
        if row["query"] == "dinosaur pillow" and row["role"] != "instance_description_diagnostic_not_formal_policy":
            raise ValueError("instance-specific prompt cannot enter the formal policy")
    return normalized


def validate_source_selection(payload: Mapping[str, Any]) -> list[str]:
    if payload.get("stage") != "D5" or payload.get("query") != "pillow":
        raise ValueError("source selection must be the frozen D5 pillow ranking")
    rows = payload.get("frames")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source selection frames are required")
    frame_ids: list[str] = []
    for rank, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError("selection frames must be objects")
        frame_id = str(row.get("frame_id", ""))
        if int(row.get("rank", -1)) != rank or not frame_id or frame_id in frame_ids:
            raise ValueError("selection rank/frame ids must be contiguous and unique")
        frame_ids.append(frame_id)
    if int(payload.get("selected_count", -1)) != len(frame_ids):
        raise ValueError("selection count differs from frames")
    return frame_ids


def derive_selection(source: Mapping[str, Any], query: str) -> dict[str, Any]:
    derived = json.loads(json.dumps(source))
    derived["query"] = query
    for key in source:
        if key != "query" and derived[key] != source[key]:
            raise AssertionError("derived selection changed non-query content")
    return derived


def build_sweep_plan(
    *,
    project_root: Path,
    config_path: Path,
    source_selection_path: Path,
    output_root: Path,
    created_at: str | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    source_selection_path = source_selection_path.resolve()
    output_root = output_root.resolve()
    config = read_json(config_path)
    experiments = validate_prompt_config(config)
    source_selection = read_json(source_selection_path)
    frame_ids = validate_source_selection(source_selection)
    plan_rows: list[dict[str, Any]] = []
    for experiment in experiments:
        selection_path = output_root / "selections" / f"{experiment['experiment_id']}.json"
        output_path = output_root / "experiments" / experiment["experiment_id"]
        plan_rows.append({
            **experiment,
            "derived_selection_ref": project_reference(project_root, selection_path),
            "derived_selection_sha256": sha256_file(selection_path),
            "output_ref": project_reference(project_root, output_path),
        })
    return {
        "schema_version": PROMPT_SWEEP_SCHEMA_VERSION,
        "status": "SOURCE_PREPARED_GPU_INFERENCE_PENDING",
        "stage": PROMPT_SWEEP_STAGE,
        "scene_id": config["scene_id"],
        "split_role": config["split_role"],
        "source": {
            "config_ref": project_reference(project_root, config_path),
            "config_sha256": sha256_file(config_path),
            "selection_ref": project_reference(project_root, source_selection_path),
            "selection_sha256": sha256_file(source_selection_path),
        },
        "guards": dict(config["guards"]),
        "candidate_frame_ids": frame_ids,
        "experiments": plan_rows,
        "claim_boundary": {
            "labels_available_to_runner": False,
            "formal_policy_changed": False,
            "cubicle_accessed": False,
            "performance_claim": None,
        },
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }


def validate_sweep_plan(payload: Mapping[str, Any], *, project_root: Path) -> dict[str, Any]:
    failures: list[str] = []
    try:
        if tuple(payload) != PLAN_FIELDS:
            raise ValueError("prompt-sweep plan fields are not frozen")
        if payload["schema_version"] != PROMPT_SWEEP_SCHEMA_VERSION or payload["stage"] != PROMPT_SWEEP_STAGE:
            raise ValueError("prompt-sweep plan schema/stage changed")
        source = payload["source"]
        if not isinstance(source, Mapping) or tuple(source) != (
            "config_ref", "config_sha256", "selection_ref", "selection_sha256",
        ):
            raise ValueError("prompt-sweep source fields are not frozen")
        config_path = resolve_reference(project_root, str(source["config_ref"]))
        selection_path = resolve_reference(project_root, str(source["selection_ref"]))
        if sha256_file(config_path) != source["config_sha256"] or sha256_file(selection_path) != source["selection_sha256"]:
            raise ValueError("prompt-sweep source hash mismatch")
        config = read_json(config_path)
        experiments = validate_prompt_config(config)
        source_selection = read_json(selection_path)
        frame_ids = validate_source_selection(source_selection)
        if payload["candidate_frame_ids"] != frame_ids:
            raise ValueError("prompt-sweep candidate universe changed")
        if payload["guards"] != config["guards"]:
            raise ValueError("prompt-sweep guards differ from config")
        plan_rows = payload["experiments"]
        if not isinstance(plan_rows, list) or len(plan_rows) != len(experiments):
            raise ValueError("prompt-sweep experiment count changed")
        for expected, row in zip(experiments, plan_rows):
            if not isinstance(row, Mapping) or tuple(row) != PLAN_EXPERIMENT_FIELDS:
                raise ValueError("prompt-sweep plan experiment fields changed")
            if any(row[key] != expected[key] for key in EXPERIMENT_FIELDS):
                raise ValueError("prompt-sweep experiment differs from config")
            derived_path = resolve_reference(project_root, str(row["derived_selection_ref"]))
            resolve_reference(project_root, str(row["output_ref"]))
            if sha256_file(derived_path) != row["derived_selection_sha256"]:
                raise ValueError("derived selection hash mismatch")
            derived = read_json(derived_path)
            if derived != derive_selection(source_selection, expected["query"]):
                raise ValueError("derived selection changed rank/candidate metadata")
        boundary = payload["claim_boundary"]
        expected_boundary = {
            "labels_available_to_runner": False,
            "formal_policy_changed": False,
            "cubicle_accessed": False,
            "performance_claim": None,
        }
        if boundary != expected_boundary:
            raise ValueError("prompt-sweep claim boundary changed")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        failures.append(str(error))
    return {
        "schema_version": "0.1",
        "status": "PASS" if not failures else "FAIL",
        "stage": "D21.1-segmentation-prompt-sweep-validation",
        "checks": {
            "sources_hash_pinned": not failures,
            "same_candidate_universe": not failures,
            "formal_baseline_frozen": not failures,
            "label_and_held_out_guard": not failures,
        },
        "failures": failures,
    }
