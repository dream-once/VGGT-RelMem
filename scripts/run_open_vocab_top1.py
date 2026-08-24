"""Run the historical D4 robust single-view path.

This command is kept to reproduce artifacts that were originally labelled B0.
It uses original-resolution SAM masks followed by resizing and robust lifting,
so new controlled comparisons must use scripts.run_single_view_baselines.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

import numpy as np

from adapters.geometry import load_geometry_npz
from adapters.masks import MaskRecord, save_mask_manifest
from adapters.open_vocab import (
    PE_SOURCE_COMMIT,
    SAM3_SOURCE_COMMIT,
    PerceptionEncoderBackend,
    Sam3Backend,
    load_frame_sources,
    resize_mask_nearest,
    select_top1,
    validate_source_checkout,
)
from relground.observations import LifterConfig, LiftingError, Robust3DLifter
from relground.schemas import RunManifest
from relground.single_view import B1_ROBUST_SINGLE_VIEW


def git_commit(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def require_pinned_sources(pe_root: Path, sam3_root: Path) -> dict[str, str]:
    validate_source_checkout(
        pe_root,
        ("core/vision_encoder/pe.py", "core/vision_encoder/transforms.py"),
    )
    validate_source_checkout(
        sam3_root,
        ("sam3/model_builder.py", "sam3/model/sam3_image_processor.py"),
    )
    commits = {"perception_models": git_commit(pe_root), "sam3": git_commit(sam3_root)}
    expected = {"perception_models": PE_SOURCE_COMMIT, "sam3": SAM3_SOURCE_COMMIT}
    mismatches = {
        name: {"actual": commits[name], "expected": expected[name]}
        for name in expected
        if commits[name] != expected[name]
    }
    if mismatches:
        raise RuntimeError(f"open-vocabulary source commit mismatch: {mismatches}")
    return commits


def local_sam3_checkpoint(checkpoint: str | None) -> tuple[Path | None, bool, str]:
    if checkpoint:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SAM 3 checkpoint does not exist: {path}")
        return path, True, "explicit local checkpoint"
    try:
        from huggingface_hub import get_token, hf_hub_download

        try:
            cached = hf_hub_download(
                repo_id="facebook/sam3",
                filename="sam3.pt",
                local_files_only=True,
            )
            return Path(cached), True, "cached Hugging Face checkpoint"
        except Exception:
            token_present = get_token() is not None
            return None, token_present, (
                "Hugging Face token is configured; gated download can be attempted"
                if token_present
                else "accept facebook/sam3 terms and run `hf auth login`"
            )
    except ImportError:
        return None, False, "huggingface_hub is not installed"


def check_imports(pe_root: Path, sam3_root: Path) -> dict[str, Any]:
    for root in (sam3_root, pe_root):
        value = str(root)
        if value not in sys.path:
            sys.path.insert(0, value)
    import torch
    import torchvision
    import timm
    import core.vision_encoder.pe  # noqa: F401
    import sam3.model_builder  # noqa: F401
    import sam3.model.sam3_image_processor  # noqa: F401

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def save_preview(
    image_path: Path,
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    output_path: Path,
    query: str,
) -> None:
    from PIL import Image, ImageDraw

    palette = [(239, 68, 68), (59, 130, 246), (34, 197, 94), (234, 179, 8)]
    with Image.open(image_path) as source:
        image = source.convert("RGBA")
    for index, mask in enumerate(masks):
        color = palette[index % len(palette)]
        alpha = Image.fromarray((np.asarray(mask, dtype=np.uint8) * 105), mode="L")
        overlay = Image.new("RGBA", image.size, color + (0,))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(image)
    for index, (box, score) in enumerate(zip(boxes, scores)):
        color = palette[index % len(palette)]
        coordinates = tuple(float(value) for value in box)
        draw.rectangle(coordinates, outline=color + (255,), width=5)
        draw.text((coordinates[0] + 4, max(0.0, coordinates[1] - 16)), f"{query} {score:.3f}", fill=color + (255,))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)


def check_only(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    pe_root = Path(args.pe_root).resolve()
    sam3_root = Path(args.sam3_root).resolve()
    commits = require_pinned_sources(pe_root, sam3_root)
    geometry = load_geometry_npz(args.geometry)
    sources = load_frame_sources(
        args.geometry_manifest,
        geometry.frame_ids,
        project_root=project_root,
    )
    imports = check_imports(pe_root, sam3_root)
    checkpoint, inference_ready, checkpoint_note = local_sam3_checkpoint(args.sam3_checkpoint)
    payload = {
        "status": "READY" if imports["cuda_available"] and inference_ready else "SOURCE_READY",
        "baseline_id": B1_ROBUST_SINGLE_VIEW,
        "legacy_label": "B0",
        "inference_ready": bool(imports["cuda_available"] and inference_ready),
        "frames": len(sources),
        "point_map_shape": list(geometry.point_maps.shape[1:]),
        "source_commits": commits,
        "environment": imports,
        "sam3_checkpoint": None if checkpoint is None else str(checkpoint),
        "checkpoint_note": checkpoint_note,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    project_root = Path(args.project_root).resolve()
    pe_root = Path(args.pe_root).resolve()
    sam3_root = Path(args.sam3_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    commits = require_pinned_sources(pe_root, sam3_root)

    geometry = load_geometry_npz(args.geometry)
    sources = load_frame_sources(
        args.geometry_manifest,
        geometry.frame_ids,
        project_root=project_root,
    )
    if args.max_frames is not None:
        sources = sources[: args.max_frames]
    if not sources:
        raise ValueError("no geometry frames selected")

    cached_checkpoint, can_download, checkpoint_note = local_sam3_checkpoint(args.sam3_checkpoint)
    if args.sam3_checkpoint is None and cached_checkpoint is None and not can_download:
        raise RuntimeError(
            "SAM 3 gated checkpoint is unavailable: " + checkpoint_note
        )

    pe_backend = PerceptionEncoderBackend(
        pe_root,
        config=args.pe_config,
        checkpoint_path=args.pe_checkpoint,
        device=args.device,
    )
    image_embeddings = pe_backend.encode_images(
        [source.image_path for source in sources],
        batch_size=args.batch_size,
    )
    text_embedding = pe_backend.encode_text(args.query)
    match = select_top1(
        [source.frame_id for source in sources],
        image_embeddings,
        text_embedding,
    )
    selected = sources[match.index]
    pe_backend.close()
    del pe_backend
    gc.collect()

    from PIL import Image

    with Image.open(selected.image_path) as source_image:
        image = source_image.convert("RGB")
    sam_backend = Sam3Backend(
        sam3_root,
        checkpoint_path=args.sam3_checkpoint,
        confidence_threshold=args.sam_threshold,
        device=args.device,
    )
    segmentation = sam_backend.segment(image, args.query)

    frame = geometry.get(selected.frame_id)
    target_shape = frame.point_map.shape[:2]
    masks_dir = output_dir / "masks"
    points_dir = output_dir / "points"
    masks_dir.mkdir(parents=True, exist_ok=True)
    points_dir.mkdir(parents=True, exist_ok=True)
    lifter = Robust3DLifter(
        LifterConfig(
            confidence_threshold=args.geometry_confidence_threshold,
            min_points=args.min_points,
            outlier_mad_scale=args.outlier_mad_scale,
        )
    )
    mask_records: list[MaskRecord] = []
    observations: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    resized_masks: list[np.ndarray] = []

    for index, (full_mask, box, sam_score) in enumerate(
        zip(segmentation.masks, segmentation.boxes_xyxy, segmentation.scores)
    ):
        obs_id = f"b1_legacy_{selected.frame_id}_{index:03d}"
        resized = resize_mask_nearest(full_mask, target_shape)
        resized_masks.append(resized)
        mask_name = f"{obs_id}.npy"
        points_name = f"{obs_id}.npz"
        np.save(masks_dir / mask_name, resized)
        mask_ref = f"masks/{mask_name}"
        points_ref = f"points/{points_name}"
        mask_records.append(
            MaskRecord(
                obs_id=obs_id,
                frame_id=selected.frame_id,
                class_text=args.query,
                mask_ref=mask_ref,
                retrieval_score=match.score,
                sam_score=float(sam_score),
            )
        )
        try:
            observation, points = lifter.make_observation(
                obs_id=obs_id,
                class_text=args.query,
                frame_id=selected.frame_id,
                mask=resized,
                point_map=frame.point_map,
                confidence_map=frame.confidence_map,
                world_from_camera=frame.world_from_camera,
                retrieval_score=match.score,
                sam_score=float(sam_score),
                mask_ref=mask_ref,
                points_ref=points_ref,
            )
            observation.metadata.update(
                {
                    "box_xyxy": np.asarray(box, dtype=float).tolist(),
                    "baseline_id": B1_ROBUST_SINGLE_VIEW,
                    "legacy_label": "B0",
                    "source_mask_shape": list(full_mask.shape),
                    "geometry_mask_shape": list(resized.shape),
                }
            )
            np.savez_compressed(points_dir / points_name, points=points.astype(np.float32))
            observations.append(observation.to_dict())
        except LiftingError as error:
            rejected.append(
                {
                    "obs_id": obs_id,
                    "sam_score": float(sam_score),
                    "box_xyxy": np.asarray(box, dtype=float).tolist(),
                    "reason": str(error),
                }
            )

    mask_manifest_path = output_dir / "masks.json"
    observations_path = output_dir / "observations.json"
    preview_path = output_dir / "preview.png"
    # Preserve the historical filename so already-produced D4 runs still load.
    result_path = output_dir / "b0_result.json"
    save_mask_manifest(mask_manifest_path, mask_records)
    observations_path.write_text(
        json.dumps(
            {"schema_version": "0.1", "observations": observations},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    save_preview(
        selected.image_path,
        segmentation.masks,
        segmentation.boxes_xyxy,
        segmentation.scores,
        preview_path,
        args.query,
    )

    status = "PASS" if observations else "NO_VALID_3D_OBSERVATION"
    result = {
        "schema_version": "0.1",
        "status": status,
        "baseline_id": B1_ROBUST_SINGLE_VIEW,
        "legacy_label": "B0",
        "backend": (
            "legacy top-1 + original-resolution SAM 3 + resized mask "
            "+ robust lifting"
        ),
        "query": args.query,
        "top1": {
            "frame_id": selected.frame_id,
            "geometry_index": selected.geometry_index,
            "image_path": str(selected.image_path),
            "retrieval_score": match.score,
            "retrieval_cosine": match.cosine,
        },
        "sam_instances": len(segmentation.masks),
        "lifted_instances": len(observations),
        "rejected_instances": rejected,
        "source_commits": commits,
        "artifacts": {
            "mask_manifest": mask_manifest_path.name,
            "observations": observations_path.name,
            "preview": preview_path.name,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    peak_vram = None
    try:
        import torch

        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    except ImportError:
        pass
    manifest = RunManifest(
        git_sha=git_commit(project_root),
        env_lock="scripts/bootstrap_open_vocab.sh",
        dataset_split=Path(args.geometry).parent.name,
        seed=0,
        config={
            "pipeline": result["backend"],
            "query": args.query,
            "geometry": str(Path(args.geometry)),
            "geometry_manifest": str(Path(args.geometry_manifest)),
            "pe_config": args.pe_config,
            "sam_threshold": args.sam_threshold,
            "source_commits": commits,
            "max_frames": args.max_frames,
            "checkpoint_note": checkpoint_note,
        },
        command=shlex.join([sys.executable, "-m", "scripts.run_open_vocab_top1", *sys.argv[1:]]),
        runtime_seconds=time.perf_counter() - started,
        peak_vram_mb=peak_vram,
    )
    manifest.save(output_dir / "run_manifest.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 2


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    upstream = root / "third_party" / "VGGT-SLAM" / "third_party"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", default="runs/office-loop/geometry.npz")
    parser.add_argument("--geometry-manifest", default="runs/office-loop/geometry.manifest.json")
    parser.add_argument("--query", default="printer")
    parser.add_argument(
        "--output-dir",
        default="runs/office-loop-b1-legacy-printer",
    )
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--pe-root", default=str(upstream / "perception_models"))
    parser.add_argument("--sam3-root", default=str(upstream / "sam3"))
    parser.add_argument("--pe-config", default="PE-Core-L14-336")
    parser.add_argument("--pe-checkpoint")
    parser.add_argument("--sam3-checkpoint")
    parser.add_argument("--sam-threshold", type=float, default=0.5)
    parser.add_argument("--geometry-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--min-points", type=int, default=30)
    parser.add_argument("--outlier-mad-scale", type=float, default=3.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--check-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.query.strip():
        raise ValueError("query must not be empty")
    if not 0.0 <= args.sam_threshold <= 1.0:
        raise ValueError("sam-threshold must be in [0, 1]")
    return check_only(args) if args.check_only else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
