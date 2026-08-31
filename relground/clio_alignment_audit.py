"""Read-only Clio COLMAP/pose readiness audit for evaluator-only Sim(3)."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Mapping

from .clio_pose_alignment import read_colmap_world_from_camera


ALIGNMENT_AUDIT_SCHEMA_VERSION = "0.1"
ALIGNMENT_AUDIT_STAGE = "D16.2-clio-alignment-readiness"
RGB_PATTERN = re.compile(r"rgb_(\d+)\.(?:jpg|jpeg|png)", re.IGNORECASE)
SPARSE_REQUIRED = ("cameras.bin", "images.bin", "points3D.bin")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"alignment artifact is outside project root: {path}") from error


def _open_read_only(database: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)


def _database_inventory(database: Path) -> dict[str, Any]:
    if not database.is_file():
        raise FileNotFoundError(database)
    with _open_read_only(database) as connection:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        if quick_check != "ok":
            raise ValueError(f"COLMAP database quick_check failed: {quick_check}")
        names = [str(row[0]) for row in connection.execute("SELECT name FROM images")]
        camera_rows = list(connection.execute("SELECT model, width, height FROM cameras"))
        pose_count = int(connection.execute(
            "SELECT COUNT(*) FROM images WHERE prior_qw IS NOT NULL OR prior_qx IS NOT NULL OR prior_qy IS NOT NULL OR prior_qz IS NOT NULL OR prior_tx IS NOT NULL OR prior_ty IS NOT NULL OR prior_tz IS NOT NULL"
        ).fetchone()[0])
    ids: list[int] = []
    invalid_names: list[str] = []
    for name in names:
        match = RGB_PATTERN.fullmatch(name)
        if match is None:
            invalid_names.append(name)
        else:
            ids.append(int(match.group(1)))
    if invalid_names or len(ids) != len(set(ids)) or not ids:
        raise ValueError("COLMAP image names must be unique rgb_<index>.jpg")
    dimensions = sorted({(int(row[1]), int(row[2])) for row in camera_rows})
    models = sorted({int(row[0]) for row in camera_rows})
    return {
        "integrity": "PASS",
        "image_count": len(ids),
        "camera_count": len(camera_rows),
        "image_id_min": min(ids),
        "image_id_max": max(ids),
        "camera_models": models,
        "image_dimensions": [list(item) for item in dimensions],
        "images_with_prior_pose": pose_count,
        "image_ids": set(ids),
    }


def build_alignment_readiness(
    *,
    project_root: Path,
    database_path: Path,
    rgb_root: Path,
    sparse_root: Path,
    rosbag_path: Path,
    created_at: str | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    database_path = database_path.resolve()
    rgb_root = rgb_root.resolve()
    sparse_root = sparse_root.resolve()
    rosbag_path = rosbag_path.resolve()
    database = _database_inventory(database_path)
    database_ids = database.pop("image_ids")
    local_ids: set[int] = set()
    duplicate_local_ids: set[int] = set()
    invalid_local: list[str] = []
    if rgb_root.is_dir():
        for path in rgb_root.iterdir():
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            match = RGB_PATTERN.fullmatch(path.name)
            if match is None:
                invalid_local.append(path.name)
            else:
                frame_id = int(match.group(1))
                if frame_id in local_ids:
                    duplicate_local_ids.add(frame_id)
                local_ids.add(frame_id)
    if invalid_local:
        raise ValueError("local RGB directory contains unexpected image names")
    if duplicate_local_ids:
        raise ValueError(f"local RGB directory has duplicate frame ids: {sorted(duplicate_local_ids)}")
    sparse_files = {
        name: (sparse_root / name).is_file() for name in SPARSE_REQUIRED
    }
    sparse_ready = all(sparse_files.values())
    registered_ids: set[int] = set()
    if sparse_ready:
        registered = read_colmap_world_from_camera(sparse_root / "images.bin")
        for frame_id in registered:
            match = re.fullmatch(r"rgb_(\d+)", frame_id)
            if match is None:
                raise ValueError("sparse image names must be rgb_<index>")
            registered_ids.add(int(match.group(1)))
    comparison_ids = registered_ids if sparse_ready else database_ids
    comparison_universe = (
        "colmap_sparse_registered_images"
        if sparse_ready
        else "colmap_database_candidates"
    )
    missing = sorted(comparison_ids - local_ids)
    extra = sorted(local_ids - comparison_ids)
    rosbag_present = rosbag_path.is_file()
    database_pose_ready = database["images_with_prior_pose"] == database["image_count"]
    alignment_ready = sparse_ready or rosbag_present or database_pose_ready
    readiness = (
        "READY_FOR_EVALUATOR_ONLY_SIM3"
        if alignment_ready
        else "BLOCKED_MISSING_SPARSE_OR_ROSBAG_POSES"
    )
    status = "PASS" if alignment_ready else "PASS_WITH_ALIGNMENT_INPUTS_PENDING"
    return {
        "schema_version": ALIGNMENT_AUDIT_SCHEMA_VERSION,
        "status": status,
        "stage": ALIGNMENT_AUDIT_STAGE,
        "scene_id": "apartment",
        "split_role": "development",
        "source": {
            "database": _relative(project_root, database_path),
            "database_sha256": sha256_file(database_path),
            "database_bytes": database_path.stat().st_size,
            "rgb_root": _relative(project_root, rgb_root),
            "sparse_root": _relative(project_root, sparse_root),
            "rosbag": _relative(project_root, rosbag_path),
        },
        "database": database,
        "local_rgb": {
            "comparison_universe": comparison_universe,
            "available_count": len(local_ids & comparison_ids),
            "missing_count": len(missing),
            "extra_count": len(extra),
            "missing_id_ranges": _compress_ranges(missing),
            "extra_ids": extra,
            "complete": not missing and not extra,
        },
        "pose_sources": {
            "database_prior_poses_complete": database_pose_ready,
            "sparse_required_files": sparse_files,
            "sparse_model_complete": sparse_ready,
            "sparse_registered_image_count": len(registered_ids),
            "rosbag_present": rosbag_present,
        },
        "alignment": {
            "method": "evaluator_only_sim3_from_matched_camera_poses",
            "readiness": readiness,
            "main_inference_may_read_gt": False,
            "cubicle_accessed": False,
        },
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }


def _compress_ranges(values: list[int]) -> list[list[int]]:
    if not values:
        return []
    output: list[list[int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            output.append([start, previous])
            start = value
        previous = value
    output.append([start, previous])
    return output


def validate_alignment_readiness(
    payload: Mapping[str, Any], *, project_root: Path
) -> dict[str, Any]:
    failures: list[str] = []
    try:
        if payload.get("schema_version") != ALIGNMENT_AUDIT_SCHEMA_VERSION:
            raise ValueError("unsupported Clio alignment audit schema")
        source = payload["source"]
        root = project_root.resolve()

        def resolve(reference: str) -> Path:
            path = Path(reference)
            if path.is_absolute():
                raise ValueError("alignment references must be relative")
            resolved = (root / path).resolve()
            try:
                resolved.relative_to(root)
            except ValueError as error:
                raise ValueError("alignment reference escapes project root") from error
            return resolved

        recomputed = build_alignment_readiness(
            project_root=root,
            database_path=resolve(str(source["database"])),
            rgb_root=resolve(str(source["rgb_root"])),
            sparse_root=resolve(str(source["sparse_root"])),
            rosbag_path=resolve(str(source["rosbag"])),
            created_at=str(payload["created_at"]),
        )
        if recomputed != dict(payload):
            raise ValueError("alignment readiness differs from deterministic replay")
    except (KeyError, TypeError, ValueError, OSError, sqlite3.DatabaseError) as error:
        failures.append(str(error))
    return {
        "schema_version": "0.1",
        "status": "PASS" if not failures else "FAIL",
        "stage": "D16.2-clio-alignment-validation",
        "checks": {
            "database_integrity": not failures,
            "image_universe_recomputed": not failures,
            "pose_readiness_recomputed": not failures,
            "split_guard": not failures,
        },
        "failures": failures,
    }
