"""Auditable local acquisition receipt for a public Clio apartment subset."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
import json
import zipfile


SCHEMA_VERSION = "0.1"
STAGE = "D16.1-clio-apartment-acquisition"
STATUS = "LOCAL_DEVELOPMENT_DATA_READY"
TOP_FIELDS = (
    "schema_version", "stage", "status", "source", "split_guard",
    "materialization_scope", "archive", "extraction", "usage_boundary",
    "checked_at",
)
SOURCE_FIELDS = (
    "official_readme_url", "official_shared_folder_url",
    "apartment_folder_url", "access", "author_approval_required",
)
SPLIT_FIELDS = (
    "scene_id", "role", "held_out_scene", "held_out_downloaded",
)
SCOPE_FIELDS = (
    "kind", "included_modalities", "excluded_modalities",
    "rgb_frame_count", "task_metadata_file_count", "full_scene_claimed",
)
ARCHIVE_FIELDS = (
    "path", "bytes", "sha256", "zip_entry_count", "file_entry_count",
    "uncompressed_bytes", "top_level_entries", "integrity",
)
EXTRACTION_FIELDS = ("path", "status", "file_count", "bytes")
USAGE_FIELDS = (
    "dataset_license_status", "local_research_use", "redistribution_allowed",
    "code_license_assumed_for_data",
)


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict(payload: Mapping[str, Any], fields: tuple[str, ...], name: str) -> None:
    if set(payload) != set(fields):
        raise ValueError(f"{name} fields are not frozen")


def _relative(value: Any, name: str) -> str:
    path = Path(str(value))
    if not str(value) or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be repository-relative")
    return path.as_posix()


def _https(value: Any, name: str) -> str:
    text = str(value)
    if not text.startswith("https://"):
        raise ValueError(f"{name} must use https")
    return text


def _sha256(value: Any, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return text


def _nonnegative(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def validate_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    _strict(payload, TOP_FIELDS, "Clio acquisition receipt")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported Clio acquisition schema")
    if payload["stage"] != STAGE or payload["status"] != STATUS:
        raise ValueError("Clio acquisition stage/status changed")
    source = payload["source"]
    split = payload["split_guard"]
    archive = payload["archive"]
    extraction = payload["extraction"]
    scope = payload["materialization_scope"]
    usage = payload["usage_boundary"]
    for value, fields, name in (
        (source, SOURCE_FIELDS, "source"),
        (split, SPLIT_FIELDS, "split guard"),
        (scope, SCOPE_FIELDS, "materialization scope"),
        (archive, ARCHIVE_FIELDS, "archive"),
        (extraction, EXTRACTION_FIELDS, "extraction"),
        (usage, USAGE_FIELDS, "usage boundary"),
    ):
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be an object")
        _strict(value, fields, name)
    for key in (
        "official_readme_url", "official_shared_folder_url",
        "apartment_folder_url",
    ):
        _https(source[key], key)
    if source["access"] != "PUBLIC_SHARED_LINK" or source[
        "author_approval_required"
    ] is not False:
        raise ValueError("public local access boundary changed")
    if split != {
        "scene_id": "apartment", "role": "development",
        "held_out_scene": "cubicle", "held_out_downloaded": False,
    }:
        raise ValueError("apartment/cubicle split guard changed")
    if scope != {
        "kind": "RGB_TASK_METADATA_DEVELOPMENT_SUBSET",
        "included_modalities": ["rgb_images", "task_metadata"],
        "excluded_modalities": [
            "colmap_dense", "colmap_sparse", "depth_images", "rosbag",
        ],
        "rgb_frame_count": 24,
        "task_metadata_file_count": 3,
        "full_scene_claimed": False,
    }:
        raise ValueError("Clio subset materialization scope changed")
    _relative(archive["path"], "archive path")
    _relative(extraction["path"], "extraction path")
    for key in (
        "bytes", "zip_entry_count", "file_entry_count", "uncompressed_bytes",
    ):
        _nonnegative(archive[key], f"archive {key}")
    for key in ("file_count", "bytes"):
        _nonnegative(extraction[key], f"extraction {key}")
    _sha256(archive["sha256"], "archive sha256")
    tops = archive["top_level_entries"]
    if not isinstance(tops, list) or not tops or any(
        not isinstance(item, str) or not item for item in tops
    ) or tops != sorted(set(tops)):
        raise ValueError("top-level ZIP entries must be sorted unique strings")
    if archive["integrity"] != "PASS" or extraction["status"] != "PASS":
        raise ValueError("archive and extraction must pass")
    if archive["file_entry_count"] != extraction["file_count"]:
        raise ValueError("archive/extraction file count mismatch")
    if archive["uncompressed_bytes"] != extraction["bytes"]:
        raise ValueError("archive/extraction byte count mismatch")
    if usage != {
        "dataset_license_status": "DATA_LICENSE_UNVERIFIED",
        "local_research_use": True,
        "redistribution_allowed": False,
        "code_license_assumed_for_data": False,
    }:
        raise ValueError("dataset usage boundary changed")
    if not str(payload["checked_at"]).strip():
        raise ValueError("checked_at is required")
    return json.loads(json.dumps(payload))


def _safe_zip_infos(archive_path: Path) -> tuple[list[zipfile.ZipInfo], list[str]]:
    with zipfile.ZipFile(archive_path) as handle:
        infos = handle.infolist()
        if not infos:
            raise ValueError("Clio apartment ZIP is empty")
        tops: set[str] = set()
        for info in infos:
            path = Path(info.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Clio ZIP contains a path escape")
            if path.parts:
                tops.add(path.parts[0])
        if handle.testzip() is not None:
            raise ValueError("Clio apartment ZIP integrity test failed")
    return infos, sorted(tops)


def _extraction_inventory(root: Path) -> tuple[int, int]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = [path for path in root.rglob("*") if path.is_file()]
    return len(files), sum(path.stat().st_size for path in files)


def build_receipt(
    *, project_root: str | Path, archive_path: str | Path,
    extraction_path: str | Path, apartment_folder_url: str, checked_at: str,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    archive = Path(archive_path).resolve()
    extraction = Path(extraction_path).resolve()
    try:
        archive_ref = archive.relative_to(root).as_posix()
        extraction_ref = extraction.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("Clio acquisition paths must stay inside project root") from exc
    infos, tops = _safe_zip_infos(archive)
    file_infos = [info for info in infos if not info.is_dir()]
    file_count, extracted_bytes = _extraction_inventory(extraction)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "status": STATUS,
        "source": {
            "official_readme_url": "https://github.com/MIT-SPARK/Clio#datasets",
            "official_shared_folder_url": (
                "https://www.dropbox.com/scl/fo/5bkv8rsa2xvwmvom6bmza/"
                "AOc8VW71kuZCgQjcw_REbWA?rlkey="
                "wx1njghufcxconm1znidc1hgw&dl=0"
            ),
            "apartment_folder_url": apartment_folder_url,
            "access": "PUBLIC_SHARED_LINK",
            "author_approval_required": False,
        },
        "split_guard": {
            "scene_id": "apartment", "role": "development",
            "held_out_scene": "cubicle", "held_out_downloaded": False,
        },
        "materialization_scope": {
            "kind": "RGB_TASK_METADATA_DEVELOPMENT_SUBSET",
            "included_modalities": ["rgb_images", "task_metadata"],
            "excluded_modalities": [
                "colmap_dense", "colmap_sparse", "depth_images", "rosbag",
            ],
            "rgb_frame_count": 24,
            "task_metadata_file_count": 3,
            "full_scene_claimed": False,
        },
        "archive": {
            "path": archive_ref, "bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
            "zip_entry_count": len(infos),
            "file_entry_count": len(file_infos),
            "uncompressed_bytes": sum(info.file_size for info in file_infos),
            "top_level_entries": tops, "integrity": "PASS",
        },
        "extraction": {
            "path": extraction_ref, "status": "PASS",
            "file_count": file_count, "bytes": extracted_bytes,
        },
        "usage_boundary": {
            "dataset_license_status": "DATA_LICENSE_UNVERIFIED",
            "local_research_use": True, "redistribution_allowed": False,
            "code_license_assumed_for_data": False,
        },
        "checked_at": checked_at,
    }
    return validate_receipt(receipt)
