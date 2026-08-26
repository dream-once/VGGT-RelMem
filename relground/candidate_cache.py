"""D11 policy-independent visual-memory and candidate-outcome contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import copy
import json
import re

import numpy as np

from .schemas import ObjectObservation


VISUAL_MEMORY_SCHEMA_VERSION = "0.1"
CANDIDATE_CACHE_SCHEMA_VERSION = "0.1"
OUTCOME_STATUSES = {"available", "rejected", "unmaterialized", "error"}
FORBIDDEN_CACHE_KEYS = {
    "ground_truth",
    "labels",
    "pair_labels",
    "expected_same",
    "answer",
    "metrics",
    "policy_trace",
}
VISUAL_MEMORY_FIELDS = (
    "schema_version",
    "scene_id",
    "frame_count",
    "embedding_status",
    "embedding_artifact",
    "encoder",
    "frames",
    "source_commits",
    "created_at",
)
VISUAL_FRAME_FIELDS = (
    "frame_id",
    "geometry_index",
    "image_ref",
    "image_sha256",
    "camera_center",
    "view_direction",
    "embedding_row",
)
EMBEDDING_ARTIFACT_FIELDS = ("path", "sha256", "shape", "dtype")
CACHE_FIELDS = (
    "schema_version",
    "scene_id",
    "query_id",
    "query_text",
    "materialization_status",
    "candidate_universe",
    "sources",
    "inference_config",
    "candidates",
    "counts",
    "costs",
    "artifacts",
    "source_commits",
    "created_at",
)
CACHE_SOURCE_FIELDS = (
    "d5_retrieval_sha256",
    "d6_result_sha256",
    "d7_observations_sha256",
)
CACHE_ARTIFACT_FIELDS = (
    "visual_memory_manifest",
    "d5_retrieval",
    "d6_result",
    "d7_observations",
)
INFERENCE_CONFIG_FIELDS = (
    "retrieval_policy",
    "sam_threshold",
    "lifter_config",
    "mask_resizing_after_sam",
)
CANDIDATE_FIELDS = (
    "rank",
    "frame_id",
    "geometry_index",
    "image_ref",
    "image_sha256",
    "camera_center",
    "view_direction",
    "retrieval_score",
    "retrieval_cosine",
    "outcome_status",
    "failure_reason",
    "outcome",
    "cost",
)
OUTCOME_FIELDS = (
    "sam_instances",
    "lifted_instances",
    "rejected_instances",
    "observations",
    "rejections",
)
CANDIDATE_COST_FIELDS = (
    "sam_calls",
    "runtime_seconds",
    "peak_vram_mb",
)
COUNT_FIELDS = (
    "candidate_count",
    "available_candidates",
    "rejected_candidates",
    "unmaterialized_candidates",
    "error_candidates",
    "total_observations",
    "total_rejections",
)
COST_FIELDS = ("sam_calls", "runtime_seconds", "peak_vram_mb")


def _strict_keys(
    payload: Mapping[str, Any],
    fields: tuple[str, ...],
    name: str,
) -> None:
    actual, expected = set(payload), set(fields)
    if actual != expected:
        raise ValueError(
            f"{name} fields changed: missing={sorted(expected - actual)} "
            f"unexpected={sorted(actual - expected)}"
        )


def _relative_reference(value: Any, name: str) -> str:
    text = str(value)
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a contained relative path")
    return path.as_posix()


def _sha256(value: Any, name: str) -> str:
    text = str(value).lower()
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    return text


def _vector3(value: Any, name: str, *, unit: bool = False) -> list[float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite length-3 vector")
    if unit and not np.isclose(np.linalg.norm(array), 1.0, atol=1e-5):
        raise ValueError(f"{name} must be unit length")
    return array.tolist()


def _finite_score(value: Any, name: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _walk_forbidden(payload: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key)
            current = f"{path}.{key_text}" if path else key_text
            if key_text in FORBIDDEN_CACHE_KEYS:
                found.append(current)
            found.extend(_walk_forbidden(value, current))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.extend(_walk_forbidden(value, f"{path}[{index}]"))
    return found


def validate_visual_memory_payload(payload: Mapping[str, Any]) -> None:
    _strict_keys(payload, VISUAL_MEMORY_FIELDS, "VisualMemoryManifest")
    if payload["schema_version"] != VISUAL_MEMORY_SCHEMA_VERSION:
        raise ValueError("unsupported VisualMemoryManifest schema")
    if not str(payload["scene_id"]).strip():
        raise ValueError("visual-memory scene_id is required")
    status = str(payload["embedding_status"])
    if status not in {"available", "not_retained"}:
        raise ValueError("unsupported embedding_status")
    artifact = payload["embedding_artifact"]
    if status == "not_retained":
        if artifact is not None:
            raise ValueError("not-retained embeddings cannot have an artifact")
    else:
        if not isinstance(artifact, Mapping):
            raise ValueError("available embeddings require an artifact")
        _strict_keys(
            artifact,
            EMBEDDING_ARTIFACT_FIELDS,
            "embedding_artifact",
        )
        _relative_reference(artifact["path"], "embedding_artifact.path")
        _sha256(artifact["sha256"], "embedding_artifact.sha256")
        shape = artifact["shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(not isinstance(item, int) or item < 1 for item in shape)
        ):
            raise ValueError("embedding shape must contain two positive integers")
        if not str(artifact["dtype"]).strip():
            raise ValueError("embedding dtype is required")
    encoder = payload["encoder"]
    if (
        not isinstance(encoder, Mapping)
        or set(encoder) != {"model", "revision"}
        or not str(encoder["model"]).strip()
        or not str(encoder["revision"]).strip()
    ):
        raise ValueError("encoder model and revision are required")
    frames = payload["frames"]
    if not isinstance(frames, list) or not frames:
        raise ValueError("visual memory requires frames")
    if int(payload["frame_count"]) != len(frames):
        raise ValueError("visual-memory frame_count is inconsistent")
    frame_ids: list[str] = []
    rows: list[int] = []
    for raw in frames:
        if not isinstance(raw, Mapping):
            raise ValueError("visual-memory frames must be objects")
        _strict_keys(raw, VISUAL_FRAME_FIELDS, "visual-memory frame")
        frame_id = str(raw["frame_id"])
        if not frame_id:
            raise ValueError("visual-memory frame_id is required")
        frame_ids.append(frame_id)
        index = int(raw["geometry_index"])
        row = int(raw["embedding_row"])
        if index < 0 or row < 0:
            raise ValueError("geometry_index and embedding_row must be non-negative")
        rows.append(row)
        _relative_reference(raw["image_ref"], "frame.image_ref")
        _sha256(raw["image_sha256"], "frame.image_sha256")
        _vector3(raw["camera_center"], "frame.camera_center")
        _vector3(raw["view_direction"], "frame.view_direction", unit=True)
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("visual-memory frame ids must be unique")
    if len(rows) != len(set(rows)):
        raise ValueError("visual-memory embedding rows must be unique")
    if frames != sorted(frames, key=lambda item: int(item["geometry_index"])):
        raise ValueError("visual-memory frames must follow geometry order")
    commits = payload["source_commits"]
    if not isinstance(commits, Mapping) or not commits:
        raise ValueError("visual-memory source commits are required")
    if not str(payload["created_at"]).strip():
        raise ValueError("visual-memory created_at is required")


def validate_candidate_cache_payload(payload: Mapping[str, Any]) -> None:
    _strict_keys(payload, CACHE_FIELDS, "CandidateOutcomeCache")
    if payload["schema_version"] != CANDIDATE_CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported CandidateOutcomeCache schema")
    if not all(
        str(payload[key]).strip()
        for key in ("scene_id", "query_id", "query_text", "created_at")
    ):
        raise ValueError("cache scene/query/time fields are required")
    forbidden = _walk_forbidden(payload)
    if forbidden:
        raise ValueError(
            "candidate cache contains forbidden evaluation keys: "
            + ", ".join(forbidden)
        )
    if payload["materialization_status"] not in {"complete", "partial"}:
        raise ValueError("unsupported materialization_status")
    sources = payload["sources"]
    artifacts = payload["artifacts"]
    inference = payload["inference_config"]
    counts = payload["counts"]
    costs = payload["costs"]
    for item, fields, name in (
        (sources, CACHE_SOURCE_FIELDS, "cache sources"),
        (artifacts, CACHE_ARTIFACT_FIELDS, "cache artifacts"),
        (inference, INFERENCE_CONFIG_FIELDS, "inference config"),
        (counts, COUNT_FIELDS, "cache counts"),
        (costs, COST_FIELDS, "cache costs"),
    ):
        if not isinstance(item, Mapping):
            raise ValueError(f"{name} must be an object")
        _strict_keys(item, fields, name)
    for key, value in sources.items():
        _sha256(value, f"sources.{key}")
    for key, value in artifacts.items():
        _relative_reference(value, f"artifacts.{key}")
    if not isinstance(inference["retrieval_policy"], Mapping):
        raise ValueError("retrieval_policy must be an object")
    _finite_score(inference["sam_threshold"], "sam_threshold")
    if not isinstance(inference["lifter_config"], Mapping):
        raise ValueError("lifter_config must be an object")
    if not isinstance(inference["mask_resizing_after_sam"], bool):
        raise ValueError("mask_resizing_after_sam must be boolean")

    candidates = payload["candidates"]
    universe = payload["candidate_universe"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate cache requires candidates")
    if not isinstance(universe, list):
        raise ValueError("candidate_universe must be a list")
    frame_ids: list[str] = []
    statuses: list[str] = []
    total_observations = total_rejections = sam_calls = 0
    for expected_rank, raw in enumerate(candidates, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError("candidates must be objects")
        _strict_keys(raw, CANDIDATE_FIELDS, "candidate")
        if int(raw["rank"]) != expected_rank:
            raise ValueError("candidate ranks must be contiguous")
        frame_id = str(raw["frame_id"])
        frame_ids.append(frame_id)
        if int(raw["geometry_index"]) < 0:
            raise ValueError("candidate geometry_index must be non-negative")
        _relative_reference(raw["image_ref"], "candidate.image_ref")
        _sha256(raw["image_sha256"], "candidate.image_sha256")
        _vector3(raw["camera_center"], "candidate.camera_center")
        _vector3(raw["view_direction"], "candidate.view_direction", unit=True)
        _finite_score(raw["retrieval_score"], "candidate.retrieval_score")
        _finite_score(raw["retrieval_cosine"], "candidate.retrieval_cosine")
        status = str(raw["outcome_status"])
        if status not in OUTCOME_STATUSES:
            raise ValueError("unsupported candidate outcome_status")
        statuses.append(status)
        if status == "available":
            if raw["failure_reason"] is not None:
                raise ValueError("available candidate cannot have failure_reason")
            outcome = raw["outcome"]
            cost = raw["cost"]
            if not isinstance(outcome, Mapping) or not isinstance(cost, Mapping):
                raise ValueError("available candidate requires outcome and cost")
            _strict_keys(outcome, OUTCOME_FIELDS, "candidate outcome")
            _strict_keys(cost, CANDIDATE_COST_FIELDS, "candidate cost")
            observations = outcome["observations"]
            rejections = outcome["rejections"]
            if not isinstance(observations, list) or not isinstance(rejections, list):
                raise ValueError("outcome observations/rejections must be lists")
            for observation in observations:
                parsed = ObjectObservation.from_dict(observation)
                if parsed.frame_id != frame_id:
                    raise ValueError("candidate observation frame mismatch")
            sam_instances = int(outcome["sam_instances"])
            lifted_instances = int(outcome["lifted_instances"])
            rejected_instances = int(outcome["rejected_instances"])
            if min(sam_instances, lifted_instances, rejected_instances) < 0:
                raise ValueError("candidate counts must be non-negative")
            if lifted_instances != len(observations):
                raise ValueError("candidate lifted count is inconsistent")
            if rejected_instances != len(rejections):
                raise ValueError("candidate rejection count is inconsistent")
            if sam_instances != lifted_instances + rejected_instances:
                raise ValueError("candidate SAM count is inconsistent")
            if int(cost["sam_calls"]) != 1:
                raise ValueError("available candidate must cost one SAM call")
            for key in ("runtime_seconds", "peak_vram_mb"):
                value = cost[key]
                if value is not None and _finite_score(value, f"cost.{key}") < 0:
                    raise ValueError("candidate cost cannot be negative")
            total_observations += lifted_instances
            total_rejections += rejected_instances
            sam_calls += 1
        else:
            if raw["outcome"] is not None or raw["cost"] is not None:
                raise ValueError("non-available candidate cannot expose outcome/cost")
            if not str(raw["failure_reason"] or "").strip():
                raise ValueError("non-available candidate requires a reason")
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("candidate frame ids must be unique")
    if universe != frame_ids:
        raise ValueError("candidate_universe must equal ranked frame ids")
    expected_counts = {
        "candidate_count": len(candidates),
        "available_candidates": statuses.count("available"),
        "rejected_candidates": statuses.count("rejected"),
        "unmaterialized_candidates": statuses.count("unmaterialized"),
        "error_candidates": statuses.count("error"),
        "total_observations": total_observations,
        "total_rejections": total_rejections,
    }
    if dict(counts) != expected_counts:
        raise ValueError("candidate cache counts are inconsistent")
    expected_materialization = (
        "complete"
        if not {"unmaterialized", "error"} & set(statuses)
        else "partial"
    )
    if payload["materialization_status"] != expected_materialization:
        raise ValueError("materialization_status is inconsistent")
    if int(costs["sam_calls"]) != sam_calls:
        raise ValueError("cache SAM-call cost is inconsistent")
    for key in ("runtime_seconds", "peak_vram_mb"):
        value = costs[key]
        if value is not None and _finite_score(value, f"costs.{key}") < 0:
            raise ValueError("cache costs cannot be negative")
    commits = payload["source_commits"]
    if not isinstance(commits, Mapping) or not commits:
        raise ValueError("candidate cache source commits are required")


@dataclass(frozen=True)
class VisualMemoryManifest:
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        validate_visual_memory_payload(self.payload)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VisualMemoryManifest":
        return cls(copy.deepcopy(dict(payload)))

    @classmethod
    def load(cls, path: str | Path) -> "VisualMemoryManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("VisualMemoryManifest root must be an object")
        return cls.from_dict(payload)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True)
class CandidateOutcomeCache:
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        validate_candidate_cache_payload(self.payload)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateOutcomeCache":
        return cls(copy.deepcopy(dict(payload)))

    @classmethod
    def load(cls, path: str | Path) -> "CandidateOutcomeCache":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("CandidateOutcomeCache root must be an object")
        return cls.from_dict(payload)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def build_d11_payloads(
    *,
    retrieval: Mapping[str, Any],
    d6_result: Mapping[str, Any],
    observations_payload: Mapping[str, Any],
    scene_id: str,
    query_id: str,
    image_refs: Mapping[str, str],
    image_hashes: Mapping[str, str],
    source_hashes: Mapping[str, str],
    artifact_refs: Mapping[str, str],
    created_at: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build deterministic D11 payloads from retained D5/D6/D7 JSON."""

    query = str(retrieval.get("query", "")).strip()
    if (
        not query
        or str(d6_result.get("query", "")).strip() != query
        or str(observations_payload.get("query", "")).strip() != query
    ):
        raise ValueError("D5/D6/D7 query mismatch")
    ranking = retrieval.get("raw_ranking")
    processed = d6_result.get("processed_frames")
    observations = observations_payload.get("observations")
    rejections = d6_result.get("rejected_instances")
    if not all(isinstance(item, list) for item in (
        ranking, processed, observations, rejections
    )):
        raise ValueError("D5/D6/D7 candidate collections are required")
    processed_by_frame = {
        str(item["frame_id"]): item for item in processed
    }
    observations_by_frame: dict[str, list[dict[str, Any]]] = {}
    for raw in observations:
        parsed = ObjectObservation.from_dict(raw)
        observations_by_frame.setdefault(parsed.frame_id, []).append(
            parsed.to_dict()
        )
    for rows in observations_by_frame.values():
        rows.sort(key=lambda item: item["obs_id"])
    rejections_by_frame: dict[str, list[dict[str, Any]]] = {}
    for rejection in rejections:
        frame_id = str(rejection["frame_id"])
        rejections_by_frame.setdefault(frame_id, []).append(
            copy.deepcopy(dict(rejection))
        )
    for rows in rejections_by_frame.values():
        rows.sort(key=lambda item: item["obs_id"])

    visual_frames: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for expected_rank, row in enumerate(ranking, start=1):
        frame_id = str(row["frame_id"])
        if int(row["rank"]) != expected_rank:
            raise ValueError("D5 raw ranking must be contiguous")
        common = {
            "frame_id": frame_id,
            "geometry_index": int(row["geometry_index"]),
            "image_ref": _relative_reference(
                image_refs[frame_id], f"image_refs.{frame_id}"
            ),
            "image_sha256": _sha256(
                image_hashes[frame_id], f"image_hashes.{frame_id}"
            ),
            "camera_center": _vector3(
                row["camera_center"], f"{frame_id}.camera_center"
            ),
            "view_direction": _vector3(
                row["view_direction"],
                f"{frame_id}.view_direction",
                unit=True,
            ),
        }
        visual_frames.append({
            **common,
            "embedding_row": int(row["geometry_index"]),
        })
        processed_row = processed_by_frame.get(frame_id)
        if processed_row is None:
            status = "unmaterialized"
            outcome = None
            cost = None
            failure_reason = "gpu_outcome_not_materialized"
        else:
            frame_observations = observations_by_frame.get(frame_id, [])
            frame_rejections = rejections_by_frame.get(frame_id, [])
            outcome = {
                "sam_instances": int(processed_row["sam_instances"]),
                "lifted_instances": int(processed_row["lifted_instances"]),
                "rejected_instances": int(processed_row["rejected_instances"]),
                "observations": frame_observations,
                "rejections": frame_rejections,
            }
            cost = {
                "sam_calls": 1,
                "runtime_seconds": None,
                "peak_vram_mb": None,
            }
            status = "available"
            failure_reason = None
        candidates.append({
            "rank": expected_rank,
            **common,
            "retrieval_score": float(row["retrieval_score"]),
            "retrieval_cosine": float(row["retrieval_cosine"]),
            "outcome_status": status,
            "failure_reason": failure_reason,
            "outcome": outcome,
            "cost": cost,
        })

    visual_frames.sort(key=lambda item: item["geometry_index"])
    source_commits = {
        **dict(retrieval.get("source_commits", {})),
        **dict(d6_result.get("source_commits", {})),
    }
    visual_payload = {
        "schema_version": VISUAL_MEMORY_SCHEMA_VERSION,
        "scene_id": scene_id,
        "frame_count": len(visual_frames),
        "embedding_status": "not_retained",
        "embedding_artifact": None,
        "encoder": {
            "model": str(retrieval.get("backend", "unknown")),
            "revision": str(source_commits.get("perception_models", "unknown")),
        },
        "frames": visual_frames,
        "source_commits": source_commits,
        "created_at": created_at,
    }
    statuses = [item["outcome_status"] for item in candidates]
    counts = {
        "candidate_count": len(candidates),
        "available_candidates": statuses.count("available"),
        "rejected_candidates": statuses.count("rejected"),
        "unmaterialized_candidates": statuses.count("unmaterialized"),
        "error_candidates": statuses.count("error"),
        "total_observations": sum(
            item["outcome"]["lifted_instances"]
            for item in candidates
            if item["outcome"] is not None
        ),
        "total_rejections": sum(
            item["outcome"]["rejected_instances"]
            for item in candidates
            if item["outcome"] is not None
        ),
    }
    cache_payload = {
        "schema_version": CANDIDATE_CACHE_SCHEMA_VERSION,
        "scene_id": scene_id,
        "query_id": query_id,
        "query_text": query,
        "materialization_status": (
            "complete" if counts["unmaterialized_candidates"] == 0 else "partial"
        ),
        "candidate_universe": [item["frame_id"] for item in candidates],
        "sources": {
            key: _sha256(source_hashes[key], f"source_hashes.{key}")
            for key in CACHE_SOURCE_FIELDS
        },
        "inference_config": {
            "retrieval_policy": copy.deepcopy(
                retrieval.get("retrieval_config", {})
            ),
            "sam_threshold": float(d6_result["sam_threshold"]),
            "lifter_config": copy.deepcopy(d6_result["lifter_config"]),
            "mask_resizing_after_sam": bool(
                d6_result["mask_resizing_after_sam"]
            ),
        },
        "candidates": candidates,
        "counts": counts,
        "costs": {
            "sam_calls": counts["available_candidates"],
            "runtime_seconds": None,
            "peak_vram_mb": None,
        },
        "artifacts": {
            key: _relative_reference(artifact_refs[key], f"artifacts.{key}")
            for key in CACHE_ARTIFACT_FIELDS
        },
        "source_commits": source_commits,
        "created_at": created_at,
    }
    VisualMemoryManifest.from_dict(visual_payload)
    CandidateOutcomeCache.from_dict(cache_payload)
    return visual_payload, cache_payload
