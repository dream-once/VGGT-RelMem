"""Deterministic scene geometry and object-centric viewpoint auditing.

This module deliberately contains no rendering backend.  It prepares validated,
portable NumPy/dict payloads which the D15.5 CLI can render with matplotlib,
Viser, or another viewer without changing the geometric contract here.
"""

from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


VIEWPOINT_AUDIT_VERSION = "0.1"
OBB_EDGE_INDICES = np.asarray(
    (
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7), (6, 7),
    ),
    dtype=np.int64,
)


def _require_finite(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _safe_reference(root: str | Path, reference: str | Path) -> Path:
    """Resolve an artifact reference without permitting absolute/parent escape."""

    base = Path(root).resolve()
    raw = str(reference)
    relative = Path(raw)
    if not raw or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe artifact reference: {raw!r}")
    resolved = (base / relative).resolve()
    if resolved == base or base not in resolved.parents:
        raise ValueError(f"artifact reference escapes cache root: {raw!r}")
    if not resolved.is_file():
        raise FileNotFoundError(f"referenced artifact does not exist: {raw}")
    return resolved


def _scalar_text(value: np.ndarray) -> str:
    item = np.asarray(value)
    if item.size != 1:
        raise ValueError("identity metadata must contain one scalar value")
    raw = item.reshape(-1)[0]
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def load_observation_points(
    cache_root: str | Path,
    points_ref: str | Path,
    *,
    expected_obs_id: str | None = None,
    expected_frame_id: str | None = None,
) -> np.ndarray:
    """Load a D7 point reference while enforcing containment and identities.

    ``.npz`` files may store the point array as ``points`` or as their only
    payload array.  Optional ``obs_id``/``frame_id`` scalars are checked when
    present.  Pickled arrays are never enabled.
    """

    path = _safe_reference(cache_root, points_ref)
    if path.suffix == ".npy":
        raw = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            point_keys = [key for key in archive.files if key == "points"]
            if not point_keys:
                payload_keys = [
                    key for key in archive.files if key not in {"obs_id", "frame_id"}
                ]
                if len(payload_keys) != 1:
                    raise ValueError(
                        f"point archive must contain a unique points payload: {points_ref}"
                    )
                point_keys = payload_keys
            raw = np.asarray(archive[point_keys[0]])
            if expected_obs_id is not None and "obs_id" in archive.files:
                actual = _scalar_text(archive["obs_id"])
                if actual != str(expected_obs_id):
                    raise ValueError(
                        f"point artifact obs_id mismatch: {actual} != {expected_obs_id}"
                    )
            if expected_frame_id is not None and "frame_id" in archive.files:
                actual = _scalar_text(archive["frame_id"])
                if actual != str(expected_frame_id):
                    raise ValueError(
                        "point artifact frame_id mismatch: "
                        f"{actual} != {expected_frame_id}"
                    )
    else:
        raise ValueError(f"unsupported point artifact extension: {path.suffix}")

    points = _require_finite("observation points", raw).astype(np.float64, copy=False)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("observation points must have shape [N, 3] with N > 0")
    return np.ascontiguousarray(points)


def _first_array(
    archive: Mapping[str, np.ndarray],
    names: Sequence[str],
    *,
    required: bool = True,
) -> np.ndarray | None:
    for name in names:
        if name in archive:
            return np.asarray(archive[name])
    if required:
        raise ValueError(f"geometry is missing one of the arrays: {list(names)}")
    return None


def _manifest_frames(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("frames", "keyframes", "frame_records"):
        value = payload.get(key)
        if isinstance(value, list) and all(isinstance(item, Mapping) for item in value):
            return list(value)
    frame_ids = payload.get("frame_ids")
    if isinstance(frame_ids, list):
        return [
            {"frame_id": str(frame_id), "geometry_index": index}
            for index, frame_id in enumerate(frame_ids)
        ]
    raise ValueError("geometry manifest has no frame records")


def _frame_id(record: Mapping[str, Any]) -> str:
    for key in ("frame_id", "id", "name"):
        if key in record and str(record[key]).strip():
            return str(record[key])
    raise ValueError("manifest frame record has no frame_id")


def _geometry_index(record: Mapping[str, Any], default: int) -> int:
    for key in ("geometry_index", "pointmap_index", "index"):
        if key in record:
            index = int(record[key])
            if index < 0:
                raise ValueError("geometry_index cannot be negative")
            return index
    return default


def _transform_points_sl4(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply an SL(4) homogeneous transform, including the projective w divide."""

    xyz = _require_finite("camera points", points).astype(np.float64, copy=False)
    matrix = _require_finite("world_from_camera", transform).astype(
        np.float64, copy=False
    )
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("camera points must have shape [N, 3]")
    if matrix.shape != (4, 4):
        raise ValueError("world_from_camera must have shape [4, 4]")
    determinant = float(np.linalg.det(matrix))
    if not np.isfinite(determinant) or abs(determinant) < 1e-12:
        raise ValueError("world_from_camera is singular")
    homogeneous = np.concatenate(
        (xyz, np.ones((len(xyz), 1), dtype=np.float64)), axis=1
    )
    transformed = homogeneous @ matrix.T
    w = transformed[:, 3]
    valid_w = np.isfinite(w) & (np.abs(w) > 1e-12)
    result = np.full((len(xyz), 3), np.nan, dtype=np.float64)
    result[valid_w] = transformed[valid_w, :3] / w[valid_w, None]
    return result


def _as_frame_stack(name: str, array: np.ndarray, frame_count: int) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim < 2 or value.shape[0] < frame_count:
        raise ValueError(f"{name} does not cover every manifest geometry index")
    return value


def _uint8_colors(colors: np.ndarray) -> np.ndarray:
    value = _require_finite("geometry colors", colors)
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError("geometry colors must have shape [N, 3]")
    if np.issubdtype(value.dtype, np.floating) and len(value):
        if float(np.max(value)) <= 1.0 + 1e-6:
            value = value * 255.0
    return np.clip(np.rint(value), 0, 255).astype(np.uint8)


def _resolve_image_path(reference: str, manifest_path: Path) -> Path:
    path = Path(reference)
    if path.is_absolute() and path.is_file():
        return path
    candidates = [Path.cwd() / path, manifest_path.parent / path]
    candidates.extend(parent / path for parent in manifest_path.parents)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"cannot resolve manifest image: {reference}")


def _load_preprocessed_colors(
    records: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    required_count: int,
    manifest_path: Path,
    upstream_root: Path,
) -> np.ndarray:
    """Load the exact VGGT resize/crop/pad grid used by the point maps."""
    ordered: list[Path | None] = [None] * required_count
    for record, geometry_index in zip(records, indices):
        reference = record.get("image_path")
        if reference is None:
            raise ValueError("geometry manifest frame has no image_path for RGB coloring")
        ordered[geometry_index] = _resolve_image_path(str(reference), manifest_path)
    if any(path is None for path in ordered):
        raise ValueError("geometry manifest does not cover every RGB frame index")
    checkout = upstream_root.resolve() / "third_party" / "vggt"
    if not checkout.is_dir():
        raise FileNotFoundError(f"missing VGGT source checkout: {checkout}")
    checkout_text = str(checkout)
    if checkout_text not in sys.path:
        sys.path.insert(0, checkout_text)
    from vggt.utils.load_fn import load_and_preprocess_images

    tensor = load_and_preprocess_images([str(path) for path in ordered])
    colors = tensor.permute(0, 2, 3, 1).cpu().numpy()
    return _require_finite("VGGT-preprocessed RGB", colors)


def build_world_cloud(
    geometry_path: str | Path,
    manifest_path: str | Path,
    *,
    upstream_root: str | Path = "third_party/VGGT-SLAM",
    max_points: int = 120_000,
    seed: int = 0,
    min_confidence: float | None = None,
) -> dict[str, Any]:
    """Build a deterministic sampled RGB world cloud from camera point maps.

    The function intentionally requires a per-frame SL(4) transform.  Camera
    pointmaps are never mislabeled as world coordinates.
    """

    if int(max_points) <= 0:
        raise ValueError("max_points must be positive")
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("geometry manifest root must be an object")
    records = _manifest_frames(manifest)
    ids = [_frame_id(record) for record in records]
    if len(ids) == 0 or len(ids) != len(set(ids)):
        raise ValueError("geometry manifest frame ids must be non-empty and unique")
    indices = [_geometry_index(record, index) for index, record in enumerate(records)]
    if len(indices) != len(set(indices)):
        raise ValueError("geometry manifest geometry indices must be unique")
    required_count = max(indices) + 1

    with np.load(Path(geometry_path), allow_pickle=False) as archive:
        points_stack = _first_array(
            archive,
            ("points", "pointmaps", "point_maps", "camera_points", "points_camera"),
        )
        colors_stack = _first_array(
            archive,
            ("colors", "images", "rgb", "rgbs", "image_colors"),
            required=False,
        )
        transforms = _first_array(
            archive,
            (
                "world_from_camera",
                "world_from_cameras",
                "transforms",
                "keyframe_transforms",
                "T_world_camera",
            ),
        )
        confidence_stack = _first_array(
            archive,
            ("valid_masks", "valid_mask", "confidence_maps", "raw_confidence_maps", "raw_confidence", "confidences", "confidence"),
            required=False,
        )
        points_stack = _as_frame_stack("point maps", points_stack, required_count)
        if colors_stack is None:
            colors_stack = _load_preprocessed_colors(
                records,
                indices,
                required_count,
                manifest_path,
                Path(upstream_root),
            )
        colors_stack = _as_frame_stack("colors", colors_stack, required_count)
        transforms = _as_frame_stack("world_from_camera", transforms, required_count)
        if confidence_stack is not None:
            confidence_stack = _as_frame_stack(
                "confidence", confidence_stack, required_count
            )

        frame_payloads: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        valid_source_count = 0
        for frame_id, geometry_index in zip(ids, indices):
            camera_points = np.asarray(points_stack[geometry_index]).reshape(-1, 3)
            colors = np.asarray(colors_stack[geometry_index]).reshape(-1, 3)
            if len(camera_points) != len(colors):
                raise ValueError(f"point/color length mismatch for {frame_id}")
            world = _transform_points_sl4(
                camera_points, np.asarray(transforms[geometry_index])
            )
            valid = np.all(np.isfinite(camera_points), axis=1)
            valid &= np.all(np.isfinite(world), axis=1)
            valid &= np.all(np.isfinite(colors), axis=1)
            if confidence_stack is not None:
                confidence = np.asarray(confidence_stack[geometry_index]).reshape(-1)
                if len(confidence) != len(camera_points):
                    raise ValueError(f"point/confidence length mismatch for {frame_id}")
                valid &= np.isfinite(confidence)
                threshold = 0.0 if min_confidence is None else float(min_confidence)
                valid &= confidence > threshold
            source_indices = np.flatnonzero(valid).astype(np.int64)
            valid_source_count += len(source_indices)
            frame_payloads.append(
                (
                    frame_id,
                    world[valid],
                    _uint8_colors(colors[valid]),
                    source_indices,
                )
            )

    if valid_source_count == 0:
        raise ValueError("geometry contains no finite valid points")
    target = min(int(max_points), valid_source_count)
    counts = np.asarray([len(item[1]) for item in frame_payloads], dtype=np.int64)
    exact = counts.astype(np.float64) * (target / float(valid_source_count))
    quotas = np.floor(exact).astype(np.int64)
    remainder = target - int(np.sum(quotas))
    if remainder:
        order = sorted(
            range(len(frame_payloads)),
            key=lambda index: (-(exact[index] - quotas[index]), ids[index]),
        )
        for index in order[:remainder]:
            quotas[index] += 1

    rng = np.random.default_rng(int(seed))
    sampled_points: list[np.ndarray] = []
    sampled_colors: list[np.ndarray] = []
    sampled_frames: list[np.ndarray] = []
    sampled_source_indices: list[np.ndarray] = []
    per_frame_counts: dict[str, int] = {}
    for (frame_id, points, colors, source_indices), quota in zip(
        frame_payloads, quotas
    ):
        if quota == 0:
            per_frame_counts[frame_id] = 0
            continue
        if quota == len(points):
            chosen = np.arange(len(points), dtype=np.int64)
        else:
            chosen = np.sort(rng.choice(len(points), size=int(quota), replace=False))
        sampled_points.append(points[chosen])
        sampled_colors.append(colors[chosen])
        sampled_frames.append(np.full(len(chosen), frame_id, dtype=np.str_))
        sampled_source_indices.append(source_indices[chosen])
        per_frame_counts[frame_id] = len(chosen)

    points_out = np.ascontiguousarray(np.concatenate(sampled_points, axis=0))
    colors_out = np.ascontiguousarray(np.concatenate(sampled_colors, axis=0))
    frames_out = np.ascontiguousarray(np.concatenate(sampled_frames, axis=0))
    source_out = np.ascontiguousarray(
        np.concatenate(sampled_source_indices, axis=0)
    )
    if len(points_out) != target or not np.all(np.isfinite(points_out)):
        raise AssertionError("deterministic world-cloud sampling invariant failed")
    return {
        "points": points_out,
        "colors": colors_out,
        "frame_ids": frames_out,
        "source_indices": source_out,
        "source_valid_points": int(valid_source_count),
        "sampled_points": int(target),
        "per_frame_counts": per_frame_counts,
        "seed": int(seed),
        "min_confidence": (
            None if min_confidence is None else float(min_confidence)
        ),
    }


def _matrix4(value: Any, name: str) -> np.ndarray:
    matrix = _require_finite(name, value).astype(np.float64, copy=False)
    if matrix.size == 16 and matrix.shape != (4, 4):
        matrix = matrix.reshape(4, 4)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape [4, 4]")
    return matrix


def _pose_from_record(record: Mapping[str, Any]) -> np.ndarray:
    for key in (
        "world_from_camera",
        "camera_to_world",
        "anchor_pose",
        "pose",
        "matrix",
        "transform",
    ):
        if key in record:
            return _matrix4(record[key], key)
    raise ValueError("camera record has no rigid world_from_camera pose")


def _anchor_records(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list) and all(isinstance(item, Mapping) for item in payload):
        return list(payload)
    if not isinstance(payload, Mapping):
        raise ValueError("anchor pose root must be an object or list")
    for key in ("anchors", "poses", "frames", "camera_poses"):
        value = payload.get(key)
        if isinstance(value, list) and all(isinstance(item, Mapping) for item in value):
            return list(value)
    # A frame-id -> matrix mapping is also unambiguous and portable.
    candidates = []
    for key, value in payload.items():
        try:
            matrix = _matrix4(value, f"pose[{key}]")
        except (TypeError, ValueError):
            continue
        candidates.append({"frame_id": str(key), "world_from_camera": matrix})
    if candidates:
        return candidates
    raise ValueError("anchor pose payload has no camera records")


def load_anchor_cameras(
    anchor_poses_path: str | Path,
    *,
    expected_frame_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Load rigid camera anchors in deterministic frame order."""

    payload = json.loads(Path(anchor_poses_path).read_text(encoding="utf-8"))
    records = _anchor_records(payload)
    cameras: dict[str, dict[str, Any]] = {}
    for record in records:
        frame_id = _frame_id(record)
        if frame_id in cameras:
            raise ValueError(f"duplicate camera anchor: {frame_id}")
        pose = _pose_from_record(record)
        rotation = pose[:3, :3]
        if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            raise ValueError(f"camera anchor is not affine rigid: {frame_id}")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
            raise ValueError(f"camera anchor rotation is not orthonormal: {frame_id}")
        determinant = float(np.linalg.det(rotation))
        if abs(determinant - 1.0) > 1e-4:
            raise ValueError(f"camera anchor rotation is not proper: {frame_id}")
        forward = rotation @ np.asarray([0.0, 0.0, 1.0])
        cameras[frame_id] = {
            "frame_id": frame_id,
            "world_from_camera": pose.copy(),
            "center": pose[:3, 3].copy(),
            "forward": forward / np.linalg.norm(forward),
        }
    if expected_frame_ids is None:
        order = sorted(cameras)
    else:
        order = [str(frame_id) for frame_id in expected_frame_ids]
        if len(order) != len(set(order)):
            raise ValueError("expected frame ids must be unique")
        missing = [frame_id for frame_id in order if frame_id not in cameras]
        if missing:
            raise ValueError(f"missing camera anchors: {missing}")
    return [cameras[frame_id] for frame_id in order]


def obb_edges(
    center: Sequence[float],
    rotation: Sequence[Sequence[float]],
    extents: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return eight OBB corners and the canonical twelve wireframe edges."""

    center_value = _require_finite("OBB center", center).astype(np.float64)
    rotation_value = _require_finite("OBB rotation", rotation).astype(np.float64)
    extent_value = _require_finite("OBB extents", extents).astype(np.float64)
    if center_value.shape != (3,) or extent_value.shape != (3,):
        raise ValueError("OBB center/extents must have shape [3]")
    if rotation_value.shape != (3, 3):
        raise ValueError("OBB rotation must have shape [3, 3]")
    if np.any(extent_value < 0):
        raise ValueError("OBB extents cannot be negative")
    if not np.allclose(rotation_value.T @ rotation_value, np.eye(3), atol=1e-4):
        raise ValueError("OBB rotation must be orthonormal")
    signs = np.asarray(
        [
            (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
            (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
        ],
        dtype=np.float64,
    )
    local = signs * (extent_value[None, :] / 2.0)
    corners = local @ rotation_value.T + center_value[None, :]
    return corners, OBB_EDGE_INDICES.copy()


def _mapping_get(data: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _observation_center(
    observation: Mapping[str, Any], cache_root: Path
) -> np.ndarray:
    center_paths = (
        ("obb", "center"),
        ("obb_3d", "center"),
        ("box3d", "center"),
        ("obb_center",),
        ("center",),
    )
    for path in center_paths:
        value = _mapping_get(observation, path)
        if value is not None:
            center = _require_finite("observation center", value).astype(np.float64)
            if center.shape != (3,):
                raise ValueError("observation center must have shape [3]")
            return center
    point_reference = None
    for key in ("points_ref", "point_ref", "points_path"):
        if key in observation and observation[key]:
            point_reference = str(observation[key])
            break
    if point_reference is None:
        raise ValueError(
            f"observation {observation.get('obs_id')} has neither center nor points_ref"
        )
    points = load_observation_points(
        cache_root,
        point_reference,
        expected_obs_id=str(observation.get("obs_id", "")),
        expected_frame_id=str(observation.get("frame_id", "")),
    )
    return np.median(points, axis=0)


def _quality_components(observation: Mapping[str, Any]) -> dict[str, float]:
    candidates: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = (
        (
            "retrieval",
            (("retrieval_score",), ("scores", "retrieval"), ("quality", "retrieval")),
        ),
        (
            "sam",
            (("sam_score",), ("mask_score",), ("scores", "sam"), ("quality", "sam")),
        ),
        (
            "valid_points",
            (
                ("valid_point_ratio",),
                ("point_quality",),
                ("scores", "valid_point_ratio"),
                ("quality", "valid_points"),
            ),
        ),
    )
    output: dict[str, float] = {}
    for label, paths in candidates:
        for path in paths:
            raw = _mapping_get(observation, path)
            if raw is None:
                continue
            value = float(raw)
            if not np.isfinite(value):
                raise ValueError(
                    f"observation {observation.get('obs_id')} has non-finite {label}"
                )
            output[label] = float(np.clip(value, 0.0, 1.0))
            break
    for path in (("quality",), ("observation_quality",)):
        raw = _mapping_get(observation, path)
        if raw is not None and not isinstance(raw, Mapping):
            value = float(raw)
            if not np.isfinite(value):
                raise ValueError("observation quality is non-finite")
            output.setdefault("aggregate", float(np.clip(value, 0.0, 1.0)))
    if not output:
        # Missing quality is explicit rather than silently treated as perfect.
        output["unavailable"] = 0.0
    return output


def _object_records(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    roots: list[Mapping[str, Any]] = [payload]
    for key in ("prediction", "result", "association"):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            roots.append(nested)
    for root in roots:
        for key in ("objects", "promoted_objects", "clusters"):
            value = root.get(key)
            if isinstance(value, list) and all(
                isinstance(item, Mapping) for item in value
            ):
                return list(value)
    raise ValueError("association payload has no object/cluster records")


def _object_id(record: Mapping[str, Any]) -> str:
    for key in ("object_id", "cluster_id", "id"):
        if key in record and str(record[key]).strip():
            return str(record[key])
    raise ValueError("association object has no id")


def _member_ids(record: Mapping[str, Any]) -> list[str]:
    for key in (
        "observation_ids",
        "member_observation_ids",
        "members",
        "obs_ids",
        "observations",
    ):
        raw = record.get(key)
        if isinstance(raw, list):
            values: list[str] = []
            for item in raw:
                if isinstance(item, Mapping):
                    member = item.get("obs_id", item.get("observation_id"))
                else:
                    member = item
                if member is None or not str(member).strip():
                    raise ValueError("association object contains an empty member id")
                values.append(str(member))
            if not values or len(values) != len(set(values)):
                raise ValueError("association member ids must be non-empty and unique")
            return values
    raise ValueError(f"association object {_object_id(record)} has no members")


def _identity_value(payload: Mapping[str, Any], key: str) -> str | None:
    for root in (payload, payload.get("metadata"), payload.get("prediction")):
        if isinstance(root, Mapping) and key in root and root[key] is not None:
            return str(root[key])
    return None


def _pair_metric(
    first: Mapping[str, Any], second: Mapping[str, Any], audit_center: np.ndarray
) -> dict[str, Any]:
    first_center = np.asarray(first["center"], dtype=np.float64)
    second_center = np.asarray(second["center"], dtype=np.float64)
    ray_first = audit_center - first_center
    ray_second = audit_center - second_center
    depth_first = float(np.linalg.norm(ray_first))
    depth_second = float(np.linalg.norm(ray_second))
    if depth_first <= 1e-12 or depth_second <= 1e-12:
        raise ValueError("audit center coincides with a camera center")
    cosine = float(
        np.clip(
            np.dot(ray_first / depth_first, ray_second / depth_second), -1.0, 1.0
        )
    )
    angle = float(np.degrees(np.arccos(cosine)))
    baseline = float(np.linalg.norm(first_center - second_center))
    mean_depth = float((depth_first + depth_second) / 2.0)
    return {
        "frame_a": str(first["frame_id"]),
        "frame_b": str(second["frame_id"]),
        "angle_deg": angle,
        "baseline": baseline,
        "mean_depth": mean_depth,
        "baseline_depth_ratio": baseline / mean_depth,
    }


def audit_object_viewpoints(
    scene_cache_path: str | Path,
    association_path: str | Path,
    anchor_poses_path: str | Path,
    *,
    strict_angle_deg: float = 15.0,
    strict_ratio: float = 0.2,
    diagnostic_angle_deg: float = 8.0,
    diagnostic_ratio: float = 0.1,
) -> dict[str, Any]:
    """Audit object-centric, rather than camera-heading, multiview evidence.

    A frame center is the component-wise median of all member observation
    centers in that frame.  The audit center is the component-wise median of
    those frame centers, so duplicate masks cannot overweight one frame.
    Artifact validity (parsing, identity, finiteness) remains independent of
    the evidence classification.
    """

    thresholds = (
        strict_angle_deg,
        strict_ratio,
        diagnostic_angle_deg,
        diagnostic_ratio,
    )
    if not all(np.isfinite(value) and value >= 0 for value in thresholds):
        raise ValueError("viewpoint thresholds must be finite and non-negative")

    cache_path = Path(scene_cache_path)
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    association = json.loads(Path(association_path).read_text(encoding="utf-8"))
    if not isinstance(cache, Mapping) or not isinstance(association, Mapping):
        raise ValueError("cache and association roots must be objects")
    scene_id = str(cache.get("scene_id", "")).strip()
    query = str(cache.get("query", "")).strip()
    frame_ids_raw = cache.get("frame_ids")
    observations_raw = cache.get("observations")
    if not scene_id or not query:
        raise ValueError("scene cache requires scene_id and query")
    if not isinstance(frame_ids_raw, list) or len(frame_ids_raw) != len(
        set(map(str, frame_ids_raw))
    ):
        raise ValueError("scene cache frame_ids must be a unique list")
    frame_ids = [str(value) for value in frame_ids_raw]
    if not isinstance(observations_raw, list) or not observations_raw:
        raise ValueError("scene cache observations must be a non-empty list")
    observations: dict[str, Mapping[str, Any]] = {}
    for raw in observations_raw:
        if not isinstance(raw, Mapping):
            raise ValueError("scene observation must be an object")
        obs_id = str(raw.get("obs_id", "")).strip()
        frame_id = str(raw.get("frame_id", "")).strip()
        if not obs_id or obs_id in observations:
            raise ValueError("scene observation ids must be non-empty and unique")
        if frame_id not in frame_ids:
            raise ValueError(f"observation {obs_id} uses unknown frame {frame_id}")
        class_text = raw.get("class_text", raw.get("query"))
        if class_text is not None and str(class_text) != query:
            raise ValueError(f"observation {obs_id} query disagrees with scene")
        observations[obs_id] = raw

    for key, expected in (("scene_id", scene_id), ("query", query)):
        actual = _identity_value(association, key)
        if actual is not None and actual != expected:
            raise ValueError(
                f"association {key} disagrees with cache: {actual} != {expected}"
            )
    cameras = {
        camera["frame_id"]: camera
        for camera in load_anchor_cameras(anchor_poses_path)
    }
    records = sorted(_object_records(association), key=_object_id)
    object_ids = [_object_id(record) for record in records]
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("association object ids must be unique")

    consumed: set[str] = set()
    audited_objects: list[dict[str, Any]] = []
    for record in records:
        object_id = _object_id(record)
        members = _member_ids(record)
        unknown = [obs_id for obs_id in members if obs_id not in observations]
        if unknown:
            raise ValueError(f"object {object_id} references unknown observations: {unknown}")
        duplicate_membership = sorted(set(members) & consumed)
        if duplicate_membership:
            raise ValueError(
                f"observations belong to multiple objects: {duplicate_membership}"
            )
        consumed.update(members)

        by_frame: dict[str, list[Mapping[str, Any]]] = {}
        for obs_id in members:
            observation = observations[obs_id]
            by_frame.setdefault(str(observation["frame_id"]), []).append(observation)
        missing_cameras = sorted(set(by_frame) - set(cameras))
        if missing_cameras:
            raise ValueError(
                f"object {object_id} member frames lack anchors: {missing_cameras}"
            )

        frame_evidence: list[dict[str, Any]] = []
        frame_centers: list[np.ndarray] = []
        for frame_id in sorted(by_frame):
            frame_observations = sorted(
                by_frame[frame_id], key=lambda value: str(value["obs_id"])
            )
            centers = np.stack(
                [
                    _observation_center(observation, cache_path.parent)
                    for observation in frame_observations
                ],
                axis=0,
            )
            center = np.median(centers, axis=0)
            frame_centers.append(center)
            component_rows = [
                _quality_components(observation)
                for observation in frame_observations
            ]
            component_values = [
                value for row in component_rows for value in row.values()
            ]
            quality = float(min(component_values))
            frame_evidence.append(
                {
                    "frame_id": frame_id,
                    "observation_ids": [
                        str(observation["obs_id"])
                        for observation in frame_observations
                    ],
                    "frame_center": center.tolist(),
                    "quality": quality,
                    "quality_aggregation": "minimum_over_available_components_and_observations",
                    "quality_components": component_rows,
                }
            )
        audit_center = np.median(np.stack(frame_centers, axis=0), axis=0)
        if not np.all(np.isfinite(audit_center)):
            raise ValueError(f"object {object_id} audit center is non-finite")
        pairs = [
            _pair_metric(cameras[first], cameras[second], audit_center)
            for first, second in combinations(sorted(by_frame), 2)
        ]
        strict_pairs = [
            pair
            for pair in pairs
            if pair["angle_deg"] >= float(strict_angle_deg)
            and pair["baseline_depth_ratio"] >= float(strict_ratio)
        ]
        strict_covered = sorted(
            {
                frame_id
                for pair in strict_pairs
                for frame_id in (pair["frame_a"], pair["frame_b"])
            }
        )
        diagnostic_pairs = [
            pair
            for pair in pairs
            if pair["angle_deg"] >= float(diagnostic_angle_deg)
            and pair["baseline_depth_ratio"] >= float(diagnostic_ratio)
        ]
        strict = (
            len(by_frame) >= 3
            and len(strict_pairs) >= 2
            and len(strict_covered) >= 3
        )
        diagnostic = len(by_frame) >= 2 and bool(diagnostic_pairs)
        if strict:
            evidence_status = "STRICT_MULTIVIEW"
        elif diagnostic:
            evidence_status = "DIAGNOSTIC_PARALLAX"
        else:
            evidence_status = "WEAK_OR_SINGLE_VIEW"
        audited_objects.append(
            {
                "object_id": object_id,
                "observation_ids": members,
                "distinct_frame_count": len(by_frame),
                "audit_center": audit_center.tolist(),
                "frame_evidence": frame_evidence,
                "pair_metrics": pairs,
                "strict_qualifying_pairs": [
                    [pair["frame_a"], pair["frame_b"]] for pair in strict_pairs
                ],
                "strict_covered_frames": strict_covered,
                "diagnostic_qualifying_pairs": [
                    [pair["frame_a"], pair["frame_b"]]
                    for pair in diagnostic_pairs
                ],
                "evidence_status": evidence_status,
            }
        )

    status_counts = {
        status: sum(
            item["evidence_status"] == status for item in audited_objects
        )
        for status in (
            "STRICT_MULTIVIEW",
            "DIAGNOSTIC_PARALLAX",
            "WEAK_OR_SINGLE_VIEW",
        )
    }
    if status_counts["STRICT_MULTIVIEW"]:
        evidence_status = "STRONG_OBJECT_CENTRIC_MULTIVIEW"
    elif status_counts["DIAGNOSTIC_PARALLAX"]:
        evidence_status = "DIAGNOSTIC_OBJECT_CENTRIC_PARALLAX"
    else:
        evidence_status = "WEAK_OBJECT_CENTRIC_EVIDENCE"
    return {
        "schema_version": VIEWPOINT_AUDIT_VERSION,
        "artifact_status": "PASS",
        "evidence_status": evidence_status,
        "scene_id": scene_id,
        "query": query,
        "thresholds": {
            "strict_min_distinct_frames": 3,
            "strict_min_qualifying_pairs": 2,
            "strict_min_covered_frames": 3,
            "strict_min_angle_deg": float(strict_angle_deg),
            "strict_min_baseline_depth_ratio": float(strict_ratio),
            "diagnostic_min_distinct_frames": 2,
            "diagnostic_min_qualifying_pairs": 1,
            "diagnostic_min_angle_deg": float(diagnostic_angle_deg),
            "diagnostic_min_baseline_depth_ratio": float(diagnostic_ratio),
        },
        "object_count": len(audited_objects),
        "status_counts": status_counts,
        "objects": audited_objects,
    }


__all__ = [
    "VIEWPOINT_AUDIT_VERSION",
    "OBB_EDGE_INDICES",
    "load_observation_points",
    "build_world_cloud",
    "load_anchor_cameras",
    "obb_edges",
    "audit_object_viewpoints",
]
