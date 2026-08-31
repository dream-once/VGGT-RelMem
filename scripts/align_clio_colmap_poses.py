"""Align retained VGGT anchor poses to the Clio COLMAP reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relground.clio_pose_alignment import (
    build_vggt_to_colmap_alignment,
    validate_vggt_to_colmap_alignment,
)
from relground.clio_world_alignment import (
    build_vggt_to_clio_world_alignment,
    validate_vggt_to_clio_world_alignment,
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--anchor-poses", default="runs/clio-apartment-gpu/geometry.anchor_poses.json")
    parser.add_argument("--colmap-images", default="data/clio/apartment/sparse/0/images.bin")
    parser.add_argument("--output", default="runs/clio-apartment-gpu/d21_1-pillow-audit/vggt_to_colmap_alignment.json")
    parser.add_argument("--scene-transform", default="configs/clio_scene_transforms.json")
    parser.add_argument("--scene-id", default="apartment")
    parser.add_argument("--world-output", default="runs/clio-apartment-gpu/d21_1-pillow-audit/vggt_to_clio_world_alignment.json")
    parser.add_argument("--max-rmse-m", type=float, default=0.15)
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    scene_transform_path = (project_root / args.scene_transform).resolve()
    scene_transform = json.loads(scene_transform_path.read_text(encoding="utf-8"))
    scene = scene_transform["scenes"][args.scene_id]
    scene_scale = float(scene["scale"])
    payload = build_vggt_to_colmap_alignment(
        project_root=project_root,
        anchor_poses_path=project_root / args.anchor_poses,
        colmap_images_path=project_root / args.colmap_images,
        max_rmse_colmap_units=args.max_rmse_m / scene_scale,
        scene_id=args.scene_id,
        split_role=str(scene["split_role"]),
    )
    report = validate_vggt_to_colmap_alignment(payload, project_root=project_root)
    if report["status"] != "PASS":
        raise ValueError("VGGT-to-COLMAP alignment validation failed")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    world_payload = build_vggt_to_clio_world_alignment(
        project_root=project_root,
        colmap_alignment_path=output.resolve(),
        scene_transform_path=scene_transform_path,
        scene_id=args.scene_id,
        max_rmse_m=args.max_rmse_m,
    )
    world_report = validate_vggt_to_clio_world_alignment(world_payload, project_root=project_root)
    if world_report["status"] != "PASS":
        raise ValueError("VGGT-to-Clio-world alignment validation failed")
    world_output = Path(args.world_output)
    world_output.parent.mkdir(parents=True, exist_ok=True)
    world_output.write_text(json.dumps(world_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "colmap_alignment": payload,
        "colmap_validation": report,
        "world_alignment": world_payload,
        "world_validation": world_report,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
