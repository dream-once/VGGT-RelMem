"""D16 Clio dataset contracts and fail-closed download feasibility audit."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
import json
import shutil


CLIO_SCHEMA_VERSION = "0.1"
CLIO_SPLIT_SCHEMA_VERSION = "0.1"
TEN_GIB = 10 * 1024**3
DATASET_FIELDS = (
    "schema_version", "dataset_id", "official_readme_url",
    "official_download_url", "official_source_checked_at", "code_license",
    "dataset_license", "download_policy", "scenes",
)
LICENSE_FIELDS = ("status", "identifier", "url", "notes")
POLICY_FIELDS = (
    "reserve_bytes", "formula", "unknown_size_action",
    "unknown_license_action", "unknown_checksum_action",
    "delete_existing_assets",
)
SCENE_FIELDS = (
    "scene_id", "role", "modalities", "archive_bytes", "extracted_bytes",
    "temporary_bytes", "sha256", "coordinate_status", "camera_status",
    "gt_policy", "declared_download_status",
)
SPLIT_FIELDS = (
    "schema_version", "dataset_id", "development_scenes", "held_out_scenes",
    "query_status", "query_manifest_sha256", "queries", "frozen_at",
)
ALLOWED_ROLES = {"development", "held-out"}
ALLOWED_DOWNLOAD_STATUSES = {
    "READY_TO_DOWNLOAD",
    "DATA_DOWNLOAD_BLOCKED_SIZE_UNKNOWN",
    "DATA_DOWNLOAD_BLOCKED_LICENSE_UNVERIFIED",
    "DATA_DOWNLOAD_BLOCKED_CHECKSUM_UNVERIFIED",
    "DATA_DOWNLOAD_BLOCKED_INSUFFICIENT_SPACE",
}


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    rendered = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(rendered).hexdigest()


def _strict_fields(
    payload: Mapping[str, Any], fields: tuple[str, ...], name: str
) -> None:
    if set(payload) != set(fields):
        raise ValueError(f"{name} fields are not frozen")


def _optional_nonnegative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or null")
    return value


def _validate_url(value: Any, name: str) -> str:
    text = str(value)
    if not text.startswith("https://"):
        raise ValueError(f"{name} must use https")
    return text


def validate_dataset_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    _strict_fields(payload, DATASET_FIELDS, "Clio dataset manifest")
    if payload["schema_version"] != CLIO_SCHEMA_VERSION:
        raise ValueError("unsupported Clio dataset manifest schema")
    if str(payload["dataset_id"]) != "mit-spark-clio-custom-scenes":
        raise ValueError("unexpected Clio dataset id")
    _validate_url(payload["official_readme_url"], "official_readme_url")
    _validate_url(payload["official_download_url"], "official_download_url")
    code_license = payload["code_license"]
    dataset_license = payload["dataset_license"]
    if not isinstance(code_license, Mapping) or not isinstance(
        dataset_license, Mapping
    ):
        raise ValueError("license records must be objects")
    _strict_fields(code_license, LICENSE_FIELDS, "code license")
    _strict_fields(dataset_license, LICENSE_FIELDS, "dataset license")
    if code_license["status"] != "VERIFIED_FOR_CODE_ONLY":
        raise ValueError("code license scope must remain explicit")
    if dataset_license["status"] not in {
        "DATA_LICENSE_UNVERIFIED", "VERIFIED_FOR_DATASET"
    }:
        raise ValueError("invalid dataset license status")
    _validate_url(code_license["url"], "code license URL")
    if dataset_license["url"] is not None:
        _validate_url(dataset_license["url"], "dataset license URL")
    policy = payload["download_policy"]
    if not isinstance(policy, Mapping):
        raise ValueError("download_policy must be an object")
    _strict_fields(policy, POLICY_FIELDS, "download policy")
    if policy["reserve_bytes"] != TEN_GIB:
        raise ValueError("Clio safety reserve must be exactly 10 GiB")
    if policy["formula"] != (
        "available_bytes-(archive_bytes+extracted_bytes+temporary_bytes)"
        ">=reserve_bytes"
    ):
        raise ValueError("download space formula changed")
    if any(
        policy[key] != "block"
        for key in (
            "unknown_size_action", "unknown_license_action",
            "unknown_checksum_action",
        )
    ):
        raise ValueError("unknown dataset metadata must fail closed")
    if policy["delete_existing_assets"] is not False:
        raise ValueError("download audit must never delete existing assets")
    scenes = payload["scenes"]
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("scenes must be a non-empty list")
    seen: set[str] = set()
    normalized_scenes: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes):
        if not isinstance(scene, Mapping):
            raise ValueError(f"scene {index} must be an object")
        _strict_fields(scene, SCENE_FIELDS, f"scene {index}")
        scene_id = str(scene["scene_id"])
        if not scene_id or scene_id in seen:
            raise ValueError("scene ids must be non-empty and unique")
        seen.add(scene_id)
        if scene["role"] not in ALLOWED_ROLES:
            raise ValueError("invalid scene role")
        modalities = scene["modalities"]
        if (
            not isinstance(modalities, list) or not modalities
            or any(not isinstance(item, str) or not item for item in modalities)
            or len(set(modalities)) != len(modalities)
        ):
            raise ValueError("scene modalities must be unique strings")
        for name in ("archive_bytes", "extracted_bytes", "temporary_bytes"):
            _optional_nonnegative_int(scene[name], name)
        checksum = scene["sha256"]
        if checksum is not None and (
            not isinstance(checksum, str) or len(checksum) != 64
            or any(char not in "0123456789abcdef" for char in checksum)
        ):
            raise ValueError("scene checksum must be lowercase SHA-256 or null")
        if scene["gt_policy"] != "evaluator_only":
            raise ValueError("Clio GT must remain evaluator-only")
        if scene["declared_download_status"] not in ALLOWED_DOWNLOAD_STATUSES:
            raise ValueError("invalid declared download status")
        normalized_scenes.append(dict(scene))
    result = deepcopy(dict(payload))
    result["scenes"] = normalized_scenes
    return result


def validate_split_manifest(
    payload: Mapping[str, Any], dataset: Mapping[str, Any]
) -> dict[str, Any]:
    _strict_fields(payload, SPLIT_FIELDS, "Clio split manifest")
    if payload["schema_version"] != CLIO_SPLIT_SCHEMA_VERSION:
        raise ValueError("unsupported Clio split schema")
    if payload["dataset_id"] != dataset["dataset_id"]:
        raise ValueError("split dataset id mismatch")
    development = payload["development_scenes"]
    held_out = payload["held_out_scenes"]
    if development != ["apartment"] or held_out != ["cubicle"]:
        raise ValueError("D16 scene roles are not frozen")
    if set(development) & set(held_out):
        raise ValueError("development and held-out scenes overlap")
    role_by_scene = {
        str(item["scene_id"]): str(item["role"]) for item in dataset["scenes"]
    }
    if any(role_by_scene.get(item) != "development" for item in development):
        raise ValueError("development scene role mismatch")
    if any(role_by_scene.get(item) != "held-out" for item in held_out):
        raise ValueError("held-out scene role mismatch")
    if payload["query_status"] != "PENDING_DATA_METADATA":
        raise ValueError("query list cannot be frozen without dataset metadata")
    if payload["query_manifest_sha256"] is not None or payload["queries"] != []:
        raise ValueError("D16 must not fabricate Clio queries")
    return deepcopy(dict(payload))


def _scene_decision(
    scene: Mapping[str, Any], *, license_verified: bool,
    maximum_peak_bytes: int
) -> dict[str, Any]:
    sizes = [
        scene["archive_bytes"], scene["extracted_bytes"],
        scene["temporary_bytes"],
    ]
    peak_bytes = None if any(value is None for value in sizes) else sum(sizes)
    reasons: list[str] = []
    if peak_bytes is None:
        reasons.append("DATA_DOWNLOAD_BLOCKED_SIZE_UNKNOWN")
    if not license_verified:
        reasons.append("DATA_DOWNLOAD_BLOCKED_LICENSE_UNVERIFIED")
    if scene["sha256"] is None:
        reasons.append("DATA_DOWNLOAD_BLOCKED_CHECKSUM_UNVERIFIED")
    if peak_bytes is not None and peak_bytes > maximum_peak_bytes:
        reasons.append("DATA_DOWNLOAD_BLOCKED_INSUFFICIENT_SPACE")
    priority = (
        "DATA_DOWNLOAD_BLOCKED_SIZE_UNKNOWN",
        "DATA_DOWNLOAD_BLOCKED_LICENSE_UNVERIFIED",
        "DATA_DOWNLOAD_BLOCKED_CHECKSUM_UNVERIFIED",
        "DATA_DOWNLOAD_BLOCKED_INSUFFICIENT_SPACE",
    )
    status = next((item for item in priority if item in reasons), "READY_TO_DOWNLOAD")
    if status != scene["declared_download_status"]:
        raise ValueError(
            f"scene {scene['scene_id']} declared download status is stale"
        )
    return {
        "scene_id": scene["scene_id"], "role": scene["role"],
        "peak_bytes": peak_bytes, "maximum_peak_bytes": maximum_peak_bytes,
        "download_allowed": status == "READY_TO_DOWNLOAD",
        "download_status": status, "blocking_reasons": reasons,
    }


def audit_clio_feasibility(
    dataset_payload: Mapping[str, Any],
    split_payload: Mapping[str, Any],
    *, available_bytes: int, checked_at: str
) -> dict[str, Any]:
    dataset = validate_dataset_manifest(dataset_payload)
    split = validate_split_manifest(split_payload, dataset)
    if isinstance(available_bytes, bool) or available_bytes < 0:
        raise ValueError("available_bytes must be non-negative")
    reserve = int(dataset["download_policy"]["reserve_bytes"])
    maximum_peak = max(0, int(available_bytes) - reserve)
    license_verified = (
        dataset["dataset_license"]["status"] == "VERIFIED_FOR_DATASET"
    )
    scene_results = [
        _scene_decision(
            scene, license_verified=license_verified,
            maximum_peak_bytes=maximum_peak,
        )
        for scene in dataset["scenes"]
    ]
    statuses = [item["download_status"] for item in scene_results]
    overall = (
        "READY_TO_DOWNLOAD"
        if all(item == "READY_TO_DOWNLOAD" for item in statuses)
        else statuses[0]
    )
    return {
        "schema_version": CLIO_SCHEMA_VERSION, "status": "PASS",
        "stage": "D16", "completion": "CPU_COMPLETE",
        "dataset_download_status": overall,
        "dataset_license_status": dataset["dataset_license"]["status"],
        "available_bytes": int(available_bytes), "reserve_bytes": reserve,
        "maximum_peak_bytes": maximum_peak,
        "dataset_manifest_sha256": canonical_sha256(dataset),
        "split_manifest_sha256": canonical_sha256(split),
        "query_status": split["query_status"], "scenes": scene_results,
        "side_effects": {
            "download_started": False, "files_deleted": False,
            "clio_installed": False, "gpu_used": False,
        },
        "checked_at": str(checked_at),
    }


def audit_clio_filesystem(
    dataset_payload: Mapping[str, Any],
    split_payload: Mapping[str, Any],
    *, filesystem_path: str | Path, checked_at: str
) -> dict[str, Any]:
    available = shutil.disk_usage(Path(filesystem_path)).free
    return audit_clio_feasibility(
        dataset_payload, split_payload, available_bytes=available,
        checked_at=checked_at,
    )
