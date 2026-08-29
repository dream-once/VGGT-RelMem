"""D18 frozen Q x A experiment contracts and label-separated replay."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import re

from .a2_association import (
    EvidenceAssociationConfig,
    associate_pending_a2,
    evaluate_a2_predictions,
)
from .association import ObjectMemory
from .candidate_cache import CandidateOutcomeCache
from .d9_association import (
    ManualInstanceGroup,
    ManualInstanceLabels,
    SpatialGateConfig,
    associate_pending,
    evaluate_predictions,
)
from .q1_fixed_topk import FixedTopKConfig, FixedTopKPolicy, candidate_metadata
from .q2_sequential import SequentialSearchConfig, run_sequential_search
from .schemas import ObjectObservation


D18_SCHEMA_VERSION = "0.1"
D18_EXPERIMENT_ID = "D18-frozen-QxA-v1"
D18_STATUS = "CPU_COMPLETE"
D18_HELD_OUT_STATUS = "CLIO_HELD_OUT_PENDING"
Q0_ID = "Q0-vggt-slam-upstream-top1"
Q1_ID = "Q1-fixed-topk-hybrid"
Q2_ID = "Q2-gain-based-sequential-search"
A1_ID = "A1-exact-class-spatial-gate"
A2_ID = "A2-evidence-aware-complete-link"
REQUIRED_MATRIX = (
    (Q0_ID, A1_ID),
    (Q0_ID, A2_ID),
    (Q1_ID, A1_ID),
    (Q1_ID, A2_ID),
    (Q2_ID, A2_ID),
)
DIAGNOSTIC_MATRIX = ((Q2_ID, A1_ID),)
FORBIDDEN_PREDICTION_KEYS = {
    "answer",
    "expected_same",
    "ground_truth",
    "labels",
    "metrics",
    "task_labels",
}
MANIFEST_FIELDS = (
    "schema_version",
    "experiment_id",
    "status",
    "held_out_status",
    "repository_base_commit",
    "split_manifest",
    "split_manifest_sha256",
    "budgets",
    "query_policies",
    "association_policies",
    "required_matrix",
    "diagnostic_matrix",
    "sources",
    "label_policy",
    "claims",
    "frozen_at",
)
SOURCE_FIELDS = (
    "source_id",
    "scene_id",
    "split_role",
    "cache_ref",
    "cache_sha256",
    "labels_ref",
    "labels_sha256",
    "result_scope",
)


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _strict(payload: Mapping[str, Any], fields: Sequence[str], name: str) -> None:
    if set(payload) != set(fields):
        raise ValueError(f"{name} fields are not frozen")


def _relative_ref(value: Any, name: str) -> str:
    text = str(value)
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a contained repository-relative path")
    return path.as_posix()


def _digest(value: Any, name: str) -> str:
    text = str(value).lower()
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError(f"{name} must be a SHA-256 digest")
    return text


def _walk_forbidden(payload: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            current = f"{path}.{key}" if path else str(key)
            if str(key) in FORBIDDEN_PREDICTION_KEYS:
                found.append(current)
            found.extend(_walk_forbidden(value, current))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.extend(_walk_forbidden(value, f"{path}[{index}]"))
    return found


def _matrix(payload: Any, name: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(payload, list):
        raise ValueError(f"{name} must be a list")
    rows: list[tuple[str, str]] = []
    for row in payload:
        if not isinstance(row, Mapping) or set(row) != {"q_policy", "a_policy"}:
            raise ValueError(f"{name} row fields are not frozen")
        rows.append((str(row["q_policy"]), str(row["a_policy"])))
    return tuple(rows)


def validate_experiment_manifest(
    payload: Mapping[str, Any],
    *,
    project_root: str | Path | None = None,
) -> None:
    _strict(payload, MANIFEST_FIELDS, "D18 experiment manifest")
    if payload["schema_version"] != D18_SCHEMA_VERSION:
        raise ValueError("unsupported D18 experiment schema")
    if payload["experiment_id"] != D18_EXPERIMENT_ID:
        raise ValueError("D18 experiment id changed")
    if payload["status"] != D18_STATUS:
        raise ValueError("D18 status must remain CPU_COMPLETE")
    if payload["held_out_status"] != D18_HELD_OUT_STATUS:
        raise ValueError("D18 held-out status must remain pending")
    if re.fullmatch(
        r"[0-9a-f]{40}", str(payload["repository_base_commit"])
    ) is None:
        raise ValueError("repository_base_commit must be a Git commit hash")
    split_ref = _relative_ref(payload["split_manifest"], "split_manifest")
    split_hash = _digest(
        payload["split_manifest_sha256"], "split_manifest_sha256"
    )
    if payload["budgets"] != {"Q0": 1, "Q1": [1, 3, 5], "Q2": 5}:
        raise ValueError("D18 budgets changed")
    if payload["query_policies"] != [Q0_ID, Q1_ID, Q2_ID]:
        raise ValueError("D18 query policies changed")
    if payload["association_policies"] != [A1_ID, A2_ID]:
        raise ValueError("D18 association policies changed")
    if _matrix(payload["required_matrix"], "required_matrix") != REQUIRED_MATRIX:
        raise ValueError("D18 required matrix changed")
    if (
        _matrix(payload["diagnostic_matrix"], "diagnostic_matrix")
        != DIAGNOSTIC_MATRIX
    ):
        raise ValueError("D18 diagnostic matrix changed")
    if payload["label_policy"] != {
        "prediction": "FORBIDDEN",
        "evaluation": "SEPARATE_LABEL_FILE_ONLY",
    }:
        raise ValueError("D18 label policy changed")
    if payload["claims"] != {
        "office_loop": "development_engineering_replay_not_performance",
        "synthetic": "correctness_fixture_not_performance",
        "clio": "readiness_only_no_held_out_numbers",
    }:
        raise ValueError("D18 claim boundary changed")
    sources = payload["sources"]
    if not isinstance(sources, list) or len(sources) != 2:
        raise ValueError("D18 requires office-loop and synthetic sources")
    source_ids: list[str] = []
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("D18 source must be an object")
        _strict(source, SOURCE_FIELDS, "D18 source")
        source_ids.append(str(source["source_id"]))
        cache_ref = _relative_ref(source["cache_ref"], "source.cache_ref")
        cache_hash = _digest(source["cache_sha256"], "source.cache_sha256")
        labels_ref = source["labels_ref"]
        labels_hash = source["labels_sha256"]
        if labels_ref is None:
            if labels_hash is not None:
                raise ValueError("unlabelled source cannot have a labels hash")
        else:
            labels_ref = _relative_ref(labels_ref, "source.labels_ref")
            labels_hash = _digest(labels_hash, "source.labels_sha256")
        if source["split_role"] not in {"development", "synthetic"}:
            raise ValueError("unsupported D18 split role")
        if project_root is not None:
            root = Path(project_root).resolve()
            if sha256_file(root / cache_ref) != cache_hash:
                raise ValueError(f"cache hash mismatch: {cache_ref}")
            if (
                labels_ref is not None
                and sha256_file(root / labels_ref) != labels_hash
            ):
                raise ValueError(f"labels hash mismatch: {labels_ref}")
    if source_ids != ["office-loop-development", "synthetic-correctness"]:
        raise ValueError("D18 source order or identity changed")
    if project_root is not None:
        root = Path(project_root).resolve()
        if sha256_file(root / split_ref) != split_hash:
            raise ValueError("split manifest hash mismatch")


def source_by_id(
    manifest: Mapping[str, Any], source_id: str
) -> Mapping[str, Any]:
    matches = [
        item for item in manifest["sources"] if item["source_id"] == source_id
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown D18 source: {source_id}")
    return matches[0]


@dataclass(frozen=True)
class QueryReplay:
    status: str
    selected_frames: tuple[str, ...]
    selected_outcome_statuses: tuple[str, ...]
    stop_reason: str | None
    observations: tuple[ObjectObservation, ...]
    sam_calls: int
    sam_instances: int
    lifted_instances: int
    rejected_instances: int


def _reveal(
    cache_payload: Mapping[str, Any],
    selected_frames: Sequence[str],
    *,
    stop_reason: str | None = None,
) -> QueryReplay:
    candidates = {
        str(item["frame_id"]): item for item in cache_payload["candidates"]
    }
    statuses: list[str] = []
    observations: list[ObjectObservation] = []
    counts = {
        "sam_calls": 0,
        "sam_instances": 0,
        "lifted_instances": 0,
        "rejected_instances": 0,
    }
    status = "PASS"
    for frame_id in selected_frames:
        candidate = candidates[str(frame_id)]
        outcome_status = str(candidate["outcome_status"])
        statuses.append(outcome_status)
        if outcome_status != "available":
            status = "BLOCKED_MISSING_OUTCOME"
            stop_reason = "BLOCKED_MISSING_OUTCOME"
            break
        outcome = candidate["outcome"]
        counts["sam_calls"] += int(candidate["cost"]["sam_calls"])
        counts["sam_instances"] += int(outcome["sam_instances"])
        counts["lifted_instances"] += int(outcome["lifted_instances"])
        counts["rejected_instances"] += int(outcome["rejected_instances"])
        observations.extend(
            ObjectObservation.from_dict(item)
            for item in outcome["observations"]
        )
    return QueryReplay(
        status=status,
        selected_frames=tuple(str(item) for item in selected_frames),
        selected_outcome_statuses=tuple(statuses),
        stop_reason=stop_reason,
        observations=tuple(observations),
        sam_calls=counts["sam_calls"],
        sam_instances=counts["sam_instances"],
        lifted_instances=counts["lifted_instances"],
        rejected_instances=counts["rejected_instances"],
    )


def replay_query_policy(
    cache_payload: Mapping[str, Any], q_policy: str
) -> QueryReplay:
    CandidateOutcomeCache.from_dict(cache_payload)
    if q_policy == Q0_ID:
        return _reveal(
            cache_payload, [cache_payload["candidates"][0]["frame_id"]]
        )
    if q_policy == Q1_ID:
        policy = FixedTopKPolicy(FixedTopKConfig())
        selected = policy.select(candidate_metadata(cache_payload), 5)
        reason = (
            "nonredundant_candidates_exhausted"
            if len(selected) < 5
            else "max_budget_reached"
        )
        return _reveal(
            cache_payload,
            [item["frame_id"] for item in selected],
            stop_reason=reason,
        )
    if q_policy == Q2_ID:
        trace = run_sequential_search(
            cache_payload,
            cache_ref="candidate_cache.json",
            cache_sha256="0" * 64,
            created_at="D18-deterministic-replay",
            config=SequentialSearchConfig(),
        )
        return _reveal(
            cache_payload,
            [item["selected"]["frame_id"] for item in trace["steps"]],
            stop_reason=trace["summary"]["stop_reason"],
        )
    raise ValueError(f"unsupported D18 query policy: {q_policy}")


def _associate(
    observations: Sequence[ObjectObservation],
    a_policy: str,
    *,
    scene_id: str,
    query_text: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    memory = ObjectMemory(
        metadata={"scene_id": scene_id, "query": query_text}
    )
    memory.stage_many(observations)
    if a_policy == A1_ID:
        outcome = associate_pending(memory, SpatialGateConfig())
        components = outcome["components"]
        details = outcome["pairs"]
        cluster_count = len(components)
        promoted = sum(bool(item["promoted"]) for item in components)
    elif a_policy == A2_ID:
        outcome = associate_pending_a2(memory, EvidenceAssociationConfig())
        clusters = outcome["clusters"]
        details = outcome["pairs"]
        cluster_count = len(clusters)
        promoted = sum(bool(item["promoted"]) for item in clusters)
    else:
        raise ValueError(f"unsupported D18 association policy: {a_policy}")
    summary = {
        "input_observations": len(observations),
        "pair_count": len(details),
        "cluster_count": cluster_count,
        "promoted_clusters": promoted,
        "permanent_objects": len(memory.objects),
        "pending_observations": len(memory.pending_observations),
        "association_decisions": len(memory.decisions),
        "pair_prediction_sha256": canonical_sha256(details),
    }
    return summary, details


def _row(
    cache_payload: Mapping[str, Any],
    q_policy: str,
    a_policy: str,
    role: str,
) -> dict[str, Any]:
    query = replay_query_policy(cache_payload, q_policy)
    association = None
    if query.status == "PASS":
        association, _ = _associate(
            query.observations,
            a_policy,
            scene_id=str(cache_payload["scene_id"]),
            query_text=str(cache_payload["query_text"]),
        )
    return {
        "combination_id": f"{q_policy}__{a_policy}",
        "matrix_role": role,
        "q_policy": q_policy,
        "a_policy": a_policy,
        "status": query.status,
        "requested_budget": {Q0_ID: 1, Q1_ID: 5, Q2_ID: 5}[q_policy],
        "selected_count": len(query.selected_frames),
        "selected_frames": list(query.selected_frames),
        "selected_outcome_statuses": list(
            query.selected_outcome_statuses
        ),
        "stop_reason": query.stop_reason,
        "candidate_count": len(cache_payload["candidates"]),
        "sam_calls": query.sam_calls,
        "sam_instances": query.sam_instances,
        "lifted_instances": query.lifted_instances,
        "rejected_instances": query.rejected_instances,
        "observation_count": len(query.observations),
        "association": association,
        "performance_claim": None,
    }


def run_experiment_prediction(
    manifest: Mapping[str, Any],
    cache_payload: Mapping[str, Any],
    *,
    manifest_ref: str,
    manifest_sha256: str,
    source_id: str,
    created_at: str,
) -> dict[str, Any]:
    validate_experiment_manifest(manifest)
    CandidateOutcomeCache.from_dict(cache_payload)
    source = source_by_id(manifest, source_id)
    if cache_payload["scene_id"] != source["scene_id"]:
        raise ValueError("D18 cache scene differs from frozen source")
    rows = [
        *[
            _row(cache_payload, q_policy, a_policy, "required")
            for q_policy, a_policy in REQUIRED_MATRIX
        ],
        *[
            _row(cache_payload, q_policy, a_policy, "diagnostic")
            for q_policy, a_policy in DIAGNOSTIC_MATRIX
        ],
    ]
    overall = (
        "PASS"
        if all(row["status"] == "PASS" for row in rows)
        else "PASS_WITH_BLOCKED_ROWS"
    )
    payload = {
        "schema_version": D18_SCHEMA_VERSION,
        "status": overall,
        "stage": "D18-QxA-prediction",
        "experiment_id": D18_EXPERIMENT_ID,
        "source_id": source_id,
        "scene_id": cache_payload["scene_id"],
        "query_id": cache_payload["query_id"],
        "query_text": cache_payload["query_text"],
        "split_role": source["split_role"],
        "result_scope": source["result_scope"],
        "source": {
            "experiment_manifest": _relative_ref(
                manifest_ref, "manifest_ref"
            ),
            "experiment_manifest_sha256": _digest(
                manifest_sha256, "manifest_sha256"
            ),
            "candidate_cache": source["cache_ref"],
            "candidate_cache_sha256": source["cache_sha256"],
        },
        "matrix_rows": rows,
        "held_out": {
            "scene": "cubicle",
            "status": D18_HELD_OUT_STATUS,
            "metric_values": None,
        },
        "created_at": created_at,
    }
    validate_prediction_payload(payload, manifest, cache_payload)
    return payload


def validate_prediction_payload(
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    cache_payload: Mapping[str, Any],
) -> None:
    forbidden = _walk_forbidden(payload)
    if forbidden:
        raise ValueError(
            f"D18 prediction contains evaluator-only keys: {forbidden}"
        )
    rows = payload.get("matrix_rows")
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("D18 prediction requires six frozen matrix rows")
    required = tuple(
        (row["q_policy"], row["a_policy"]) for row in rows[:5]
    )
    diagnostic = tuple(
        (row["q_policy"], row["a_policy"]) for row in rows[5:]
    )
    if required != REQUIRED_MATRIX or diagnostic != DIAGNOSTIC_MATRIX:
        raise ValueError("D18 prediction matrix differs from manifest")
    if payload["source"]["candidate_cache_sha256"] != source_by_id(
        manifest, str(payload["source_id"])
    )["cache_sha256"]:
        raise ValueError("D18 prediction cache hash differs from manifest")
    for row in rows:
        if row["candidate_count"] != len(cache_payload["candidates"]):
            raise ValueError("D18 rows do not share one candidate universe")
        if row["performance_claim"] is not None:
            raise ValueError("D18 replay cannot claim performance")
        if row["status"] == "PASS" and row["association"] is None:
            raise ValueError("passing D18 row requires association output")
        if (
            row["status"] == "BLOCKED_MISSING_OUTCOME"
            and set(row["selected_outcome_statuses"]) <= {"available"}
        ):
            raise ValueError("blocked D18 row lacks a missing outcome")
        if (
            row["status"] == "BLOCKED_MISSING_OUTCOME"
            and row["association"] is not None
        ):
            raise ValueError("blocked D18 row cannot run association")
    expected_heldout = {
        "scene": "cubicle",
        "status": D18_HELD_OUT_STATUS,
        "metric_values": None,
    }
    if payload["held_out"] != expected_heldout:
        raise ValueError("D18 held-out readiness boundary changed")


def _subset_labels(
    labels: ManualInstanceLabels,
    observations: Sequence[ObjectObservation],
) -> ManualInstanceLabels:
    selected = {item.obs_id for item in observations}
    groups = tuple(
        ManualInstanceGroup(
            instance_id=group.instance_id,
            observation_ids=tuple(
                item for item in group.observation_ids if item in selected
            ),
        )
        for group in labels.instance_groups
        if any(item in selected for item in group.observation_ids)
    )
    if not groups:
        raise ValueError("D18 selected observations have no labels")
    return ManualInstanceLabels(
        scene_id=labels.scene_id,
        query=labels.query,
        annotation_method=labels.annotation_method,
        notes=labels.notes,
        instance_groups=groups,
    )


def evaluate_synthetic_prediction(
    prediction: Mapping[str, Any],
    cache_payload: Mapping[str, Any],
    labels: ManualInstanceLabels,
    *,
    prediction_ref: str,
    prediction_sha256: str,
    labels_ref: str,
    labels_sha256: str,
    created_at: str,
) -> dict[str, Any]:
    if prediction["source_id"] != "synthetic-correctness":
        raise ValueError("D18 labelled evaluation is synthetic-only")
    rows: list[dict[str, Any]] = []
    for prediction_row in prediction["matrix_rows"]:
        query = replay_query_policy(
            cache_payload, str(prediction_row["q_policy"])
        )
        if query.status != "PASS":
            rows.append(
                {
                    "combination_id": prediction_row["combination_id"],
                    "status": query.status,
                    "sample_observations": len(query.observations),
                    "metrics": None,
                }
            )
            continue
        _, pairs = _associate(
            query.observations,
            str(prediction_row["a_policy"]),
            scene_id=str(cache_payload["scene_id"]),
            query_text=str(cache_payload["query_text"]),
        )
        subset = _subset_labels(labels, query.observations)
        if prediction_row["a_policy"] == A1_ID:
            evaluation = evaluate_predictions(
                query.observations, pairs, subset
            )
        else:
            evaluation = evaluate_a2_predictions(
                query.observations, pairs, subset
            )
        rows.append(
            {
                "combination_id": prediction_row["combination_id"],
                "status": "PASS",
                "sample_observations": len(query.observations),
                "metrics": evaluation["metrics"],
            }
        )
    return {
        "schema_version": D18_SCHEMA_VERSION,
        "status": "PASS",
        "stage": "D18-QxA-synthetic-evaluation",
        "experiment_id": D18_EXPERIMENT_ID,
        "scope": "synthetic_correctness_fixture_not_performance",
        "source": {
            "prediction": _relative_ref(
                prediction_ref, "prediction_ref"
            ),
            "prediction_sha256": _digest(
                prediction_sha256, "prediction_sha256"
            ),
            "labels": _relative_ref(labels_ref, "labels_ref"),
            "labels_sha256": _digest(labels_sha256, "labels_sha256"),
        },
        "rows": rows,
        "performance_claim": None,
        "created_at": created_at,
    }


def deterministic_prediction_replay(
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    cache_payload: Mapping[str, Any],
) -> dict[str, Any]:
    return run_experiment_prediction(
        manifest,
        cache_payload,
        manifest_ref=str(payload["source"]["experiment_manifest"]),
        manifest_sha256=str(
            payload["source"]["experiment_manifest_sha256"]
        ),
        source_id=str(payload["source_id"]),
        created_at=str(payload["created_at"]),
    )
