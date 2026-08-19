"""Create static and interactive previews of an exported geometry bundle."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
import time

import numpy as np

from adapters.geometry import load_geometry_npz


def _prepare_vggt_import(upstream: Path) -> None:
    candidate = upstream / "third_party/vggt"
    if candidate.exists():
        sys.path.insert(0, str(candidate))


def _world_points_and_colors(
    geometry_path: Path,
    upstream: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geometry = load_geometry_npz(geometry_path)
    manifest_path = geometry_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    image_paths = [str(item["image_path"]) for item in manifest["frames"]]
    if len(image_paths) != len(geometry.frame_ids):
        raise ValueError("manifest and geometry frame counts differ")

    _prepare_vggt_import(upstream)
    from vggt.utils.load_fn import load_and_preprocess_images

    images = load_and_preprocess_images(image_paths).permute(0, 2, 3, 1).numpy()
    if images.shape[:3] != geometry.point_maps.shape[:3]:
        raise ValueError(
            f"preprocessed RGB shape {images.shape[:3]} does not match "
            f"point maps {geometry.point_maps.shape[:3]}"
        )

    points: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    for index in range(len(geometry.frame_ids)):
        local = geometry.point_maps[index].reshape(-1, 3).astype(np.float64)
        confidence = geometry.confidence_maps[index].reshape(-1)
        rgb = images[index].reshape(-1, 3)
        valid = np.isfinite(local).all(axis=1) & np.isfinite(confidence) & (confidence > 0)
        local_h = np.column_stack((local[valid], np.ones(np.count_nonzero(valid))))
        world_h = (geometry.world_from_camera[index] @ local_h.T).T
        valid_w = np.isfinite(world_h).all(axis=1) & (np.abs(world_h[:, 3]) > 1e-12)
        points.append(world_h[valid_w, :3] / world_h[valid_w, 3:4])
        colors.append(np.clip(rgb[valid][valid_w], 0.0, 1.0))

    anchor_path = geometry_path.with_suffix(".anchor_poses.json")
    anchor_data = json.loads(anchor_path.read_text())
    trajectory = np.asarray(
        [anchor_data[frame_id] for frame_id in geometry.frame_ids], dtype=np.float64
    )[:, :3, 3]
    return np.concatenate(points), np.concatenate(colors), trajectory


def _save_ply(points: np.ndarray, colors: np.ndarray, output: Path, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    if voxel > 0:
        cloud = cloud.voxel_down_sample(voxel)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(output), cloud, write_ascii=False, compressed=True):
        raise RuntimeError(f"failed to write {output}")
    return np.asarray(cloud.points), np.asarray(cloud.colors)


def _save_preview(
    points: np.ndarray,
    colors: np.ndarray,
    trajectory: np.ndarray,
    output: Path,
    max_points: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(points) > max_points:
        indices = np.random.default_rng(0).choice(len(points), max_points, replace=False)
        points = points[indices]
        colors = colors[indices]

    lower = np.quantile(points, 0.01, axis=0)
    upper = np.quantile(points, 0.99, axis=0)
    inside = np.all((points >= lower) & (points <= upper), axis=1)
    points = points[inside]
    colors = colors[inside]
    center = (lower + upper) / 2
    radius = float(np.max(upper - lower) / 2)

    fig = plt.figure(figsize=(18, 6), facecolor="white")
    views = ((18, -65, "Perspective"), (90, -90, "Top"), (0, -90, "Front"))
    for position, (elevation, azimuth, title) in enumerate(views, start=1):
        axis = fig.add_subplot(1, 3, position, projection="3d")
        axis.scatter(points[:, 0], points[:, 2], points[:, 1], c=colors, s=0.25, linewidths=0)
        axis.plot(
            trajectory[:, 0], trajectory[:, 2], trajectory[:, 1],
            color="red", linewidth=2.0, marker="o", markersize=2,
        )
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[2] - radius, center[2] + radius)
        axis.set_zlim(center[1] - radius, center[1] + radius)
        axis.set_xlabel("X")
        axis.set_ylabel("Z (forward)")
        axis.set_zlabel("Y (up)")
        axis.set_title(title)
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_box_aspect((1, 1, 1))
    fig.suptitle("VGGT-SLAM geometry — red line is the camera trajectory")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _serve(
    points: np.ndarray,
    colors: np.ndarray,
    trajectory: np.ndarray,
    host: str,
    port: int,
) -> None:
    import viser

    server = viser.ViserServer(host=host, port=port)
    server.scene.add_point_cloud(
        "/geometry", points=points.astype(np.float32),
        colors=(np.clip(colors, 0, 1) * 255).astype(np.uint8),
        point_size=0.002, point_shape="circle",
    )
    server.scene.add_spline_catmull_rom(
        "/camera_trajectory", positions=trajectory.astype(np.float32),
        color=(255, 0, 0), line_width=3.0,
    )
    print(f"Viser is running on http://{host}:{port}")
    print("Forward this port in VS Code, then open the forwarded local URL. Ctrl-C stops it.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("geometry")
    parser.add_argument("--upstream", default="third_party/VGGT-SLAM")
    parser.add_argument("--ply")
    parser.add_argument("--preview")
    parser.add_argument("--voxel", type=float, default=0.005)
    parser.add_argument("--max-preview-points", type=int, default=80000)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    geometry_path = Path(args.geometry)
    output_ply = Path(args.ply) if args.ply else geometry_path.parent / "point_cloud.ply"
    output_preview = (
        Path(args.preview) if args.preview else geometry_path.parent / "point_cloud_preview.png"
    )
    points, colors, trajectory = _world_points_and_colors(
        geometry_path, Path(args.upstream)
    )
    points, colors = _save_ply(points, colors, output_ply, args.voxel)
    _save_preview(
        points, colors, trajectory, output_preview, args.max_preview_points
    )
    print(json.dumps({
        "point_count": len(points),
        "ply": str(output_ply),
        "preview": str(output_preview),
        "voxel": args.voxel,
    }, indent=2))
    if args.serve:
        _serve(points, colors, trajectory, args.host, args.port)


if __name__ == "__main__":
    main()
