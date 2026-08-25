"""Export an in-memory official VGGT-SLAM 2.0 Solver to NPZ/JSON."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np

from .geometry import GEOMETRY_SCHEMA_VERSION, save_geometry_npz


@dataclass(frozen=True)
class VGGTFrameRecord:
    frame_id: str
    image_path: str
    submap_id: int
    submap_frame_index: int
    graph_node_id: int
    confidence_threshold: float


@dataclass(frozen=True)
class VGGTExportSummary:
    geometry_path: str
    manifest_path: str
    anchor_poses_path: str
    frame_count: int
    submap_count: int
    skipped_loop_closure_submaps: int
    skipped_duplicate_frames: int
    point_map_shape: tuple[int, int, int]
    estimated_uncompressed_bytes: int


def validate_upstream_layout(path: str | Path) -> Path:
    root = Path(path).resolve()
    required = (
        root / "main.py",
        root / "vggt_slam/solver.py",
        root / "vggt_slam/submap.py",
    )
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise FileNotFoundError(
            f"invalid VGGT-SLAM checkout; missing: {missing}"
        )
    return root


def _array(value: Any) -> np.ndarray:
    for method in ("detach", "cpu"):
        if hasattr(value, method):
            value = getattr(value, method)()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _frame_id(
    image_path: str,
    submap_id: int,
    index: int,
    used: set[str],
) -> str:
    stem = Path(image_path).stem or f"frame_{submap_id}_{index}"
    candidate = (
        stem
        if stem not in used
        else f"{stem}__s{submap_id}_i{index}"
    )
    serial = 2
    while candidate in used:
        candidate = f"{stem}__s{submap_id}_i{index}_{serial}"
        serial += 1
    used.add(candidate)
    return candidate


def export_solver_geometry(
    solver: Any,
    geometry_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    anchor_poses_path: str | Path | None = None,
    source_commit: str = "unknown",
    deduplicate_images: bool = True,
) -> VGGTExportSummary:
    """Export non-loop submaps without modifying upstream.

    Point maps stay in VGGT submap coordinates. world_from_camera stores the
    optimized SL(4) global transform and may be projective. Both raw VGGT
    confidence and the upstream percentile-threshold valid mask are preserved.
    The legacy confidence_maps field remains a binary alias for consumers.
    Rigid anchor poses are written separately.
    """

    output = Path(geometry_path)
    manifest = (
        Path(manifest_path)
        if manifest_path
        else output.with_suffix(".manifest.json")
    )
    poses_path = (
        Path(anchor_poses_path)
        if anchor_poses_path
        else output.with_suffix(".anchor_poses.json")
    )
    points: list[np.ndarray] = []
    confidence: list[np.ndarray] = []
    raw_confidence: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    transforms: list[np.ndarray] = []
    records: list[VGGTFrameRecord] = []
    anchor_poses: dict[str, list[list[float]]] = {}
    seen_images: set[str] = set()
    used_ids: set[str] = set()
    submap_count = skipped_lc = skipped_duplicates = 0
    submaps = solver.map.ordered_submaps_by_key()

    for submap in submaps:
        if bool(submap.get_lc_status()):
            skipped_lc += 1
            continue
        submap_count += 1
        submap_id = int(submap.get_id())
        local_points = _array(submap.pointclouds)
        raw_conf = _array(submap.conf)
        if local_points.ndim != 4 or local_points.shape[-1] != 3:
            raise ValueError(
                f"submap {submap_id} pointcloud shape is invalid"
            )
        if raw_conf.shape != local_points.shape[:-1]:
            raise ValueError(
                f"submap {submap_id} confidence shape is invalid"
            )
        names = list(getattr(submap, "img_names", []))
        if len(names) != len(local_points):
            names = [
                f"submap_{submap_id}_frame_{i}"
                for i in range(len(local_points))
            ]
        rigid_poses = _array(
            submap.get_all_poses_world(solver.graph)
        )
        if rigid_poses.shape != (len(local_points), 4, 4):
            raise ValueError(
                f"submap {submap_id} anchor pose shape is invalid"
            )
        threshold = float(submap.get_conf_threshold())

        for index, name in enumerate(names):
            image_key = str(Path(name).resolve(strict=False))
            if deduplicate_images and image_key in seen_images:
                skipped_duplicates += 1
                continue
            seen_images.add(image_key)
            frame_id = _frame_id(
                str(name),
                submap_id,
                index,
                used_ids,
            )
            node_id = submap_id + index
            transform = _array(
                solver.graph.get_homography(node_id)
            ).astype(np.float64)
            if (
                transform.shape != (4, 4)
                or not np.all(np.isfinite(transform))
            ):
                raise ValueError(
                    f"graph node {node_id} homography is invalid"
                )
            if abs(transform[-1, -1]) > 1e-12:
                transform /= transform[-1, -1]
            frame_raw_confidence = raw_conf[index].astype(
                np.float32,
                copy=False,
            )
            frame_valid_mask = frame_raw_confidence > threshold
            points.append(
                local_points[index].astype(np.float32, copy=False)
            )
            raw_confidence.append(frame_raw_confidence)
            valid_masks.append(frame_valid_mask)
            confidence.append(frame_valid_mask.astype(np.float32))
            transforms.append(transform)
            anchor_poses[frame_id] = (
                rigid_poses[index].astype(float).tolist()
            )
            records.append(
                VGGTFrameRecord(
                    frame_id,
                    str(name),
                    submap_id,
                    index,
                    node_id,
                    threshold,
                )
            )

    if not points:
        raise ValueError("solver contains no exportable frames")
    if len({item.shape for item in points}) != 1:
        raise ValueError(
            "all exported point maps must share one resolution"
        )
    point_stack = np.stack(points)
    confidence_stack = np.stack(confidence)
    raw_confidence_stack = np.stack(raw_confidence)
    valid_mask_stack = np.stack(valid_masks)
    transform_stack = np.stack(transforms)
    save_geometry_npz(
        output,
        frame_ids=[item.frame_id for item in records],
        point_maps=point_stack,
        confidence_maps=confidence_stack,
        world_from_camera=transform_stack,
        raw_confidence_maps=raw_confidence_stack,
        valid_masks=valid_mask_stack,
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": GEOMETRY_SCHEMA_VERSION,
                "source": "MIT-SPARK/VGGT-SLAM",
                "source_commit": source_commit,
                "point_coordinates": (
                    "VGGT submap canonical coordinates"
                ),
                "point_transform": (
                    "optimized SL(4) global_from_submap; "
                    "homogeneous divide required"
                ),
                "confidence_encoding": {
                    "raw_confidence_maps": (
                        "unmodified VGGT confidence values"
                    ),
                    "valid_masks": (
                        "boolean raw_confidence > upstream "
                        "percentile threshold"
                    ),
                    "confidence_maps": (
                        "legacy float32 alias of valid_masks"
                    ),
                },
                "frames": [asdict(item) for item in records],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    poses_path.parent.mkdir(parents=True, exist_ok=True)
    poses_path.write_text(
        json.dumps(
            anchor_poses,
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    size = (
        point_stack.nbytes
        + confidence_stack.nbytes
        + raw_confidence_stack.nbytes
        + valid_mask_stack.nbytes
        + transform_stack.nbytes
    )
    return VGGTExportSummary(
        str(output),
        str(manifest),
        str(poses_path),
        len(records),
        submap_count,
        skipped_lc,
        skipped_duplicates,
        tuple(point_stack.shape[1:]),
        size,
    )
