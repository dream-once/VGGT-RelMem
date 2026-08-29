"""Render a D15.5 scene-memory audit as an offline preview/video or Viser scene.

The camera frusta drawn by this tool are deliberately schematic: the exported
geometry contract does not retain calibrated intrinsics.  All distances shown
by this tool are therefore in VGGT-SLAM reconstruction units.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from relground.scene_visualization import (
    audit_object_viewpoints,
    build_world_cloud,
    load_anchor_cameras,
    load_observation_points,
    obb_edges,
)


SCHEMA_VERSION = "d15.5-scene-memory-visualization/0.1"
AXIS_NOTICE = "schematic camera frusta; distances are reconstruction units"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, root: Path, kind: str, status: str = "available") -> dict[str, Any]:
    item: dict[str, Any] = {
        "kind": kind,
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "status": status,
    }
    if status == "available":
        item.update({"bytes": path.stat().st_size, "sha256": _sha256(path)})
    return item


def _safe_output(root: Path, name: str) -> Path:
    root = root.resolve()
    output = (root / name).resolve()
    if output.parent != root:
        raise ValueError(f"output escapes output directory: {name}")
    return output


def _finite_array(value: Any, *, shape: tuple[int, ...] | None = None, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _nested(mapping: Mapping[str, Any], *paths: str) -> Any:
    for path in paths:
        value: Any = mapping
        ok = True
        for part in path.split("."):
            if not isinstance(value, Mapping) or part not in value:
                ok = False
                break
            value = value[part]
        if ok:
            return value
    return None


def _records(payload: Mapping[str, Any], keys: Sequence[str]) -> list[dict[str, Any]]:
    for key in keys:
        value = _nested(payload, key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, Mapping)]
        if isinstance(value, Mapping):
            result: list[dict[str, Any]] = []
            for record_id, item in value.items():
                if isinstance(item, Mapping):
                    copied = dict(item)
                    copied.setdefault("id", str(record_id))
                    result.append(copied)
            if result:
                return result
    return []


def _find_scene_cache(observation_root: Path) -> Path:
    candidates = (
        observation_root / "observations.json",
        observation_root / "scene_cache.json",
        observation_root / "object_observations.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no scene_cache.json or observations.json under {observation_root}"
    )


def _camera_record(item: Mapping[str, Any]) -> dict[str, Any]:
    frame_id = str(_nested(item, "frame_id", "id", "name") or "")
    pose_value = _nested(item, "world_from_camera", "pose", "transform", "matrix")
    center_value = _nested(item, "center", "position", "camera_center")
    direction_value = _nested(item, "direction", "forward", "view_direction")
    pose: np.ndarray | None = None
    if pose_value is not None:
        pose = _finite_array(pose_value, shape=(4, 4), name=f"camera {frame_id} pose")
    if center_value is not None:
        center = _finite_array(center_value, shape=(3,), name=f"camera {frame_id} center")
    elif pose is not None:
        center = pose[:3, 3]
    else:
        raise ValueError(f"camera {frame_id!r} has no center or pose")
    if direction_value is not None:
        direction = _finite_array(direction_value, shape=(3,), name=f"camera {frame_id} direction")
    elif pose is not None:
        direction = pose[:3, 2]
    else:
        direction = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"camera {frame_id!r} has a degenerate view direction")
    direction = direction / norm
    return {"frame_id": frame_id, "center": center, "direction": direction, "pose": pose}


def _extract_obb(item: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    obb = _nested(item, "fused_obb", "obb", "obb_3d", "bbox_3d", "geometry.obb")
    source = obb if isinstance(obb, Mapping) else item
    center = _nested(source, "center", "centroid")
    rotation = _nested(source, "rotation", "axes", "orientation")
    extents = _nested(source, "extents", "extent", "size", "dimensions")
    if center is None or extents is None:
        return None
    center_a = _finite_array(center, shape=(3,), name="OBB center")
    extents_a = _finite_array(extents, shape=(3,), name="OBB extents")
    if np.any(extents_a < 0):
        raise ValueError("OBB extents must be non-negative")
    if rotation is None:
        rotation_a = np.eye(3, dtype=np.float64)
    else:
        rotation_a = _finite_array(rotation, shape=(3, 3), name="OBB rotation")
    return center_a, rotation_a, extents_a


def _object_id(item: Mapping[str, Any], fallback: str) -> str:
    return str(
        _nested(item, "object_id", "cluster_id", "memory_id", "id", "track_id")
        or fallback
    )


def _object_label(item: Mapping[str, Any]) -> str:
    return str(
        _nested(item, "class_text", "label", "class_name", "category", "query", "semantic_label")
        or "object"
    )


def _observation_ids(item: Mapping[str, Any]) -> list[str]:
    value = _nested(
        item,
        "observation_ids",
        "member_observation_ids",
        "members",
        "observations",
    )
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for member in value:
        if isinstance(member, Mapping):
            member = _nested(member, "obs_id", "observation_id", "id")
        if member is not None:
            result.append(str(member))
    return result


def _status(item: Mapping[str, Any]) -> str:
    value = str(_nested(item, "status", "state", "association_status") or "").lower()
    if value in {"pending", "provisional", "rejected", "unassociated"}:
        return "pending"
    promoted = _nested(item, "promoted", "is_permanent", "accepted")
    return "predicted" if promoted is not False else "pending"


def _stable_color(identifier: str) -> np.ndarray:
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], "big") / 65535.0
    import colorsys

    return np.asarray(colorsys.hsv_to_rgb(hue, 0.72, 0.95), dtype=np.float64)


def _scene_records(memory_path: Path, scene_cache_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    memory = _read_json(memory_path)
    cache = _read_json(scene_cache_path)
    objects = _records(
        memory,
        (
            "objects",
            "predicted_objects",
            "permanent_objects",
            "object_memory.objects",
            "result.objects",
            "clusters",
            "association.clusters",
        ),
    )
    observations = _records(
        cache,
        (
            "observations",
            "object_observations",
            "scene.observations",
            "records",
        ),
    )
    observation_by_id = {
        str(_nested(item, "obs_id", "observation_id", "id") or f"observation-{index:04d}"): item
        for index, item in enumerate(observations)
    }
    normalized_objects: list[dict[str, Any]] = []
    claimed: set[str] = set()
    for index, item in enumerate(objects):
        identifier = _object_id(item, f"object-{index:04d}")
        members = _observation_ids(item)
        claimed.update(members)
        normalized_objects.append(
            {
                "id": identifier,
                "label": _object_label(item),
                "status": _status(item),
                "obb": _extract_obb(item),
                "observation_ids": members,
                "color": _stable_color(identifier),
                "raw": item,
            }
        )
    normalized_observations: list[dict[str, Any]] = []
    for index, item in enumerate(observations):
        identifier = str(_nested(item, "obs_id", "observation_id", "id") or f"observation-{index:04d}")
        normalized_observations.append(
            {
                "id": identifier,
                "frame_id": str(_nested(item, "frame_id", "source_frame_id") or ""),
                "status": "associated" if identifier in claimed else "pending",
                "obb": _extract_obb(item),
                "points_ref": _nested(item, "points_ref", "point_ref", "artifacts.points"),
                "raw": item,
            }
        )
    return normalized_objects, normalized_observations


def _frustum_segments(camera: Mapping[str, Any], scale: float) -> np.ndarray:
    center = np.asarray(camera["center"], dtype=np.float64)
    forward = np.asarray(camera["direction"], dtype=np.float64)
    helper = np.array((0.0, 1.0, 0.0), dtype=np.float64)
    if abs(float(np.dot(helper, forward))) > 0.9:
        helper = np.array((1.0, 0.0, 0.0), dtype=np.float64)
    right = np.cross(forward, helper)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    up = np.cross(right, forward)
    tip = center + forward * scale
    corners = np.stack(
        [tip + sx * right * scale * 0.45 + sy * up * scale * 0.30 for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    )
    edges: list[np.ndarray] = []
    for corner in corners:
        edges.append(np.stack((center, corner)))
    for index in range(4):
        edges.append(np.stack((corners[index], corners[(index + 1) % 4])))
    return np.stack(edges)


def _box_segments(obb: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    corners, edges = obb_edges(*obb)
    corners_a = _finite_array(corners, name="OBB corners")
    edges_a = np.asarray(edges)
    if edges_a.ndim == 3 and edges_a.shape[1:] == (2, 3):
        segments = edges_a.astype(np.float64)
    elif edges_a.ndim == 2 and edges_a.shape[1] == 2:
        segments = corners_a[edges_a.astype(np.int64)]
    else:
        raise ValueError(f"unexpected OBB edge shape: {edges_a.shape}")
    if not np.isfinite(segments).all():
        raise ValueError("OBB segments contain non-finite values")
    return segments


def _scene_bounds(points: np.ndarray, trajectory: np.ndarray, objects: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, float]:
    samples = [points, trajectory]
    for item in objects:
        if item["obb"] is not None:
            samples.append(_box_segments(item["obb"]).reshape(-1, 3))
    combined = np.concatenate([value for value in samples if len(value)], axis=0)
    if not len(combined) or not np.isfinite(combined).all():
        raise ValueError("scene has no finite geometry")
    lower = np.quantile(combined, 0.01, axis=0)
    upper = np.quantile(combined, 0.99, axis=0)
    center = (lower + upper) * 0.5
    radius = max(float(np.max(upper - lower) * 0.55), 1e-3)
    return center, radius


def _draw_segments(axis: Any, segments: np.ndarray, color: Any, *, width: float, alpha: float, linestyle: str = "-") -> None:
    for segment in segments:
        axis.plot(
            segment[:, 0], segment[:, 2], segment[:, 1],
            color=color, linewidth=width, alpha=alpha, linestyle=linestyle,
        )


def _draw_scene(
    axis: Any,
    points: np.ndarray,
    colors: np.ndarray,
    trajectory: np.ndarray,
    cameras: Sequence[Mapping[str, Any]],
    selected_ids: set[str],
    objects: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    center: np.ndarray,
    radius: float,
    *,
    point_size: float = 0.25,
) -> None:
    axis.scatter(points[:, 0], points[:, 2], points[:, 1], c=colors, s=point_size, linewidths=0, alpha=0.55)
    axis.plot(trajectory[:, 0], trajectory[:, 2], trajectory[:, 1], color="#e53935", linewidth=2.0, alpha=0.95)
    frustum_scale = max(radius * 0.055, 0.02)
    for camera in cameras:
        selected = camera["frame_id"] in selected_ids
        if selected:
            _draw_segments(axis, _frustum_segments(camera, frustum_scale), "#ffd54f", width=1.1, alpha=0.9)
    for item in observations:
        if item["obb"] is not None:
            color = "#90a4ae" if item["status"] == "pending" else "#80cbc4"
            _draw_segments(axis, _box_segments(item["obb"]), color, width=0.55, alpha=0.38, linestyle="--")
    for item in objects:
        if item["obb"] is not None:
            pending = item["status"] == "pending"
            _draw_segments(
                axis, _box_segments(item["obb"]), item["color"],
                width=2.5 if not pending else 1.3,
                alpha=0.98 if not pending else 0.55,
                linestyle="--" if pending else "-",
            )
            obb_center = item["obb"][0]
            axis.text(obb_center[0], obb_center[2], obb_center[1], f"{item['label']}\n{item['id']}", color=item["color"], fontsize=7)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[2] - radius, center[2] + radius)
    axis.set_zlim(center[1] - radius, center[1] + radius)
    axis.set_xlabel("X")
    axis.set_ylabel("Z (forward)")
    axis.set_zlabel("Y (up)")
    axis.set_box_aspect((1, 1, 1))


def _save_overview(
    output: Path,
    points: np.ndarray,
    colors: np.ndarray,
    trajectory: np.ndarray,
    cameras: Sequence[Mapping[str, Any]],
    selected_ids: set[str],
    objects: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    center: np.ndarray,
    radius: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 6), facecolor="#101216")
    views = ((22, -62, "Perspective"), (90, -90, "Top"), (4, -90, "Front"))
    for position, (elevation, azimuth, title) in enumerate(views, start=1):
        axis = fig.add_subplot(1, 3, position, projection="3d", facecolor="#101216")
        _draw_scene(axis, points, colors, trajectory, cameras, selected_ids, objects, observations, center, radius)
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_title(title, color="white")
        axis.tick_params(colors="#b0bec5")
    fig.suptitle(
        "D15.5 scene memory — RGB map | red trajectory | yellow selected cameras | solid fused OBB | dashed pending\n"
        + AXIS_NOTICE,
        color="white",
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_video(
    output: Path,
    points: np.ndarray,
    colors: np.ndarray,
    trajectory: np.ndarray,
    cameras: Sequence[Mapping[str, Any]],
    selected_ids: set[str],
    objects: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    center: np.ndarray,
    radius: float,
    seconds: float,
    fps: int,
) -> int:
    if seconds <= 0 or fps <= 0:
        raise ValueError("video seconds and fps must be positive")
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame_count = max(2, int(round(seconds * fps)))
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#090b0f")
    axis = fig.add_subplot(111, projection="3d", facecolor="#090b0f")
    _draw_scene(
        axis, points, colors, trajectory, cameras, selected_ids, objects,
        observations, center, radius, point_size=0.35,
    )
    axis.tick_params(colors="#b0bec5")
    fig.tight_layout()
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        plt.close(fig)
        raise RuntimeError(f"could not open video writer for {output}")
    try:
        for index in range(frame_count):
            phase = index / frame_count
            azimuth = -70.0 + phase * 360.0
            elevation = 18.0 + 10.0 * math.sin(phase * math.tau)
            axis.view_init(elev=elevation, azim=azimuth)
            axis.set_title(
                f"Object-centric orbit {index + 1}/{frame_count}  |  azimuth {azimuth:.1f}°\n{AXIS_NOTICE}",
                color="white",
            )
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            writer.write(bgr)
    finally:
        writer.release()
        plt.close(fig)
    return frame_count


def _load_observation_clouds(
    observation_root: Path,
    observations: Sequence[Mapping[str, Any]],
    max_points_each: int = 3000,
) -> dict[str, np.ndarray]:
    clouds: dict[str, np.ndarray] = {}
    for item in observations:
        points_ref = item.get("points_ref")
        if not points_ref:
            continue
        try:
            points = np.asarray(load_observation_points(observation_root, points_ref), dtype=np.float64)
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            continue
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"observation {item['id']} points must have shape (N,3)")
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) > max_points_each:
            seed = int.from_bytes(hashlib.sha256(item["id"].encode()).digest()[:8], "big")
            indices = np.random.default_rng(seed).choice(len(points), max_points_each, replace=False)
            points = points[indices]
        if len(points):
            clouds[item["id"]] = points
    return clouds


def _save_ply(output: Path, points: np.ndarray, colors: np.ndarray, observation_clouds: Mapping[str, np.ndarray]) -> tuple[str, int]:
    all_points = [points]
    all_colors = [np.clip(colors, 0.0, 1.0)]
    for identifier, cloud in sorted(observation_clouds.items()):
        all_points.append(cloud)
        color = _stable_color(identifier)
        all_colors.append(np.broadcast_to(color, cloud.shape))
    fused_points = np.concatenate(all_points, axis=0)
    fused_colors = np.concatenate(all_colors, axis=0)
    if not np.isfinite(fused_points).all() or not np.isfinite(fused_colors).all():
        raise ValueError("PLY data contains non-finite values")
    try:
        import open3d as o3d
    except ImportError:
        return "dependency_unavailable", 0
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(fused_points)
    cloud.colors = o3d.utility.Vector3dVector(fused_colors)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(output), cloud, write_ascii=False, compressed=True):
        raise RuntimeError(f"failed to write {output}")
    return "available", len(fused_points)


def _serve(
    points: np.ndarray,
    colors: np.ndarray,
    trajectory: np.ndarray,
    cameras: Sequence[Mapping[str, Any]],
    selected_ids: set[str],
    objects: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    observation_clouds: Mapping[str, np.ndarray],
    host: str,
    port: int,
    radius: float,
) -> None:
    import viser

    server = viser.ViserServer(host=host, port=port)
    server.scene.add_point_cloud(
        "/scene/rgb_geometry",
        points=points.astype(np.float32),
        colors=(np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8),
        point_size=max(radius * 0.0015, 0.001),
        point_shape="circle",
    )
    if len(trajectory) >= 2:
        server.scene.add_spline_catmull_rom(
            "/scene/camera_trajectory",
            positions=trajectory.astype(np.float32),
            color=(244, 67, 54),
            line_width=3.0,
        )
    scale = max(radius * 0.055, 0.02)
    for camera in cameras:
        if camera["frame_id"] not in selected_ids:
            continue
        segments = _frustum_segments(camera, scale).astype(np.float32)
        server.scene.add_line_segments(
            f"/selected_cameras/{camera['frame_id']}_schematic",
            points=segments,
            colors=np.broadcast_to(np.array((255, 213, 79), dtype=np.uint8), segments.shape),
            line_width=2.0,
        )
    for item in observations:
        cloud = observation_clouds.get(item["id"])
        color = np.array((144, 164, 174) if item["status"] == "pending" else (128, 203, 196), dtype=np.uint8)
        if cloud is not None:
            server.scene.add_point_cloud(
                f"/observations/{item['status']}/{item['id']}/points",
                points=cloud.astype(np.float32),
                colors=np.broadcast_to(color, cloud.shape),
                point_size=max(radius * 0.002, 0.0015),
                point_shape="circle",
            )
        if item["obb"] is not None:
            segments = _box_segments(item["obb"]).astype(np.float32)
            server.scene.add_line_segments(
                f"/observations/{item['status']}/{item['id']}/obb",
                points=segments,
                colors=np.broadcast_to(color, segments.shape),
                line_width=1.0,
            )
    for item in objects:
        if item["obb"] is None:
            continue
        rgb = (np.clip(item["color"], 0.0, 1.0) * 255).astype(np.uint8)
        segments = _box_segments(item["obb"]).astype(np.float32)
        server.scene.add_line_segments(
            f"/objects/{item['status']}/{item['id']}/obb",
            points=segments,
            colors=np.broadcast_to(rgb, segments.shape),
            line_width=4.0 if item["status"] == "predicted" else 2.0,
        )
        server.scene.add_label(
            f"/objects/{item['status']}/{item['id']}/label",
            text=f"{item['label']} [{item['status']}]",
            position=item["obb"][0].astype(np.float32),
        )
    print(json.dumps({"viser_url": f"http://{host}:{port}", "notice": AXIS_NOTICE}, indent=2))
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        return


def _input_record(path: Path, output_root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display = resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        display = resolved.name
    return {"path": display, "bytes": resolved.stat().st_size, "sha256": _sha256(resolved)}


def _require_input_path(
    path: Path,
    *,
    label: str,
    directory: bool = False,
) -> None:
    valid = path.is_dir() if directory else path.is_file()
    if valid:
        return
    expected = "directory" if directory else "file"
    project_root = Path(__file__).resolve().parents[1]
    message = (
        f"required {label} {expected} not found: {path}. "
        "Relative input paths are resolved from the current working "
        f"directory ({Path.cwd().resolve()}). Run "
        f"'cd {project_root}' before the documented command, or pass "
        "absolute paths for every input."
    )
    if path.exists() and directory:
        raise NotADirectoryError(message)
    raise FileNotFoundError(message)


def run(args: argparse.Namespace) -> dict[str, Any]:
    geometry = Path(args.geometry)
    geometry_manifest = Path(args.geometry_manifest)
    anchor_poses = Path(args.anchor_poses)
    memory = Path(args.memory)
    observation_root = Path(args.observation_root)
    output_root = Path(args.output_dir).resolve()
    upstream = Path(args.upstream)
    for path, label in (
        (geometry, "geometry"),
        (geometry_manifest, "geometry manifest"),
        (anchor_poses, "anchor poses"),
        (memory, "object memory"),
    ):
        _require_input_path(path, label=label)
    _require_input_path(
        observation_root, label="observation root", directory=True
    )
    _require_input_path(upstream, label="upstream checkout", directory=True)
    if args.max_background_points <= 0:
        raise ValueError("--max-background-points must be positive")
    if not (1 <= args.port <= 65535):
        raise ValueError("--port must be between 1 and 65535")
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root.is_symlink():
        raise ValueError("output directory may not be a symbolic link")

    scene_cache_path = _find_scene_cache(observation_root)
    cloud = build_world_cloud(
        geometry,
        geometry_manifest,
        upstream_root=upstream,
        max_points=args.max_background_points,
        seed=0,
    )
    points = _finite_array(cloud["points"], name="background points")
    colors = _finite_array(cloud["colors"], name="background colors")
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError("world cloud points/colors must both have shape (N,3)")
    if not len(points):
        raise ValueError("world cloud is empty")
    if len(colors) and float(np.max(colors)) > 1.0 + 1e-6:
        colors = colors / 255.0
    colors = np.clip(colors, 0.0, 1.0)
    camera_values = load_anchor_cameras(anchor_poses)
    cameras = [_camera_record(item) for item in camera_values]
    if not cameras:
        raise ValueError("anchor pose file contains no cameras")
    trajectory = np.stack([item["center"] for item in cameras])
    objects, observations = _scene_records(memory, scene_cache_path)
    selected_ids = {item["frame_id"] for item in observations if item["frame_id"]}
    observation_clouds = _load_observation_clouds(observation_root, observations)
    center, radius = _scene_bounds(points, trajectory, objects)

    audit = audit_object_viewpoints(scene_cache_path, memory, anchor_poses)
    if not isinstance(audit, Mapping):
        raise ValueError("viewpoint audit must return a JSON object")
    audit_output = _safe_output(output_root, "viewpoint_audit.json")
    _write_json(audit_output, dict(audit))

    ply_output = _safe_output(output_root, "scene_memory.ply")
    ply_status, ply_point_count = _save_ply(ply_output, points, colors, observation_clouds)
    overview_output = _safe_output(output_root, "overview.png")
    _save_overview(
        overview_output, points, colors, trajectory, cameras, selected_ids,
        objects, observations, center, radius,
    )
    video_output = _safe_output(output_root, "object_parallax.mp4")
    video_frames = _save_video(
        video_output, points, colors, trajectory, cameras, selected_ids,
        objects, observations, center, radius, args.video_seconds, args.video_fps,
    )

    artifacts = [
        _artifact(audit_output, output_root, "viewpoint_audit"),
        _artifact(overview_output, output_root, "three_view_overview"),
        _artifact(video_output, output_root, "dynamic_object_parallax_video"),
    ]
    if ply_status == "available":
        artifacts.append(_artifact(ply_output, output_root, "colored_scene_point_cloud"))
    else:
        artifacts.append(
            {"kind": "colored_scene_point_cloud", "path": "scene_memory.ply", "status": ply_status}
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "artifact_status": "PASS",
        "viewpoint_evidence_status": audit["evidence_status"],
        "scene_id": audit["scene_id"],
        "query": audit["query"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "coordinate_contract": {
            "units": "VGGT-SLAM reconstruction units",
            "camera_frusta": "schematic (intrinsics are not retained)",
            "finite_values_required": True,
        },
        "inputs": {
            "geometry": _input_record(geometry, output_root),
            "geometry_manifest": _input_record(geometry_manifest, output_root),
            "anchor_poses": _input_record(anchor_poses, output_root),
            "memory": _input_record(memory, output_root),
            "scene_cache": _input_record(scene_cache_path, output_root),
        },
        "counts": {
            "background_points": int(len(points)),
            "ply_points": int(ply_point_count),
            "anchor_cameras": len(cameras),
            "selected_cameras": len(selected_ids),
            "predicted_objects": sum(item["status"] == "predicted" for item in objects),
            "pending_objects": sum(item["status"] == "pending" for item in objects),
            "observations": len(observations),
            "observation_clouds": len(observation_clouds),
            "video_frames": video_frames,
        },
        "render": {
            "max_background_points": args.max_background_points,
            "video_seconds": args.video_seconds,
            "video_fps": args.video_fps,
            "orbit_degrees": 360.0,
            "dynamic_camera": True,
        },
        "source_counts": cloud.get("source_counts", {}),
        "artifacts": artifacts,
        "manifest_inventory_note": "visualization_manifest.json omits its own recursive hash",
    }
    manifest_output = _safe_output(output_root, "visualization_manifest.json")
    _write_json(manifest_output, manifest)

    if args.serve:
        _serve(
            points, colors, trajectory, cameras, selected_ids, objects,
            observations, observation_clouds, args.host, args.port, radius,
        )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create an honest scene-level D15.5 3D memory visualization."
    )
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--geometry-manifest", required=True)
    parser.add_argument("--anchor-poses", required=True)
    parser.add_argument("--memory", required=True, help="D12 object_memory.json or A2 result")
    parser.add_argument("--observation-root", required=True, help="D7 artifact directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--upstream", default="third_party/VGGT-SLAM")
    parser.add_argument("--max-background-points", type=int, default=120_000)
    parser.add_argument("--video-seconds", type=float, default=12.0)
    parser.add_argument("--video-fps", type=int, default=24)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    return parser


def main() -> None:
    manifest = run(build_parser().parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
