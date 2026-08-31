"""Run D6: D5 top-K frames -> SAM 3 masks -> robust world-space 3D OBBs."""

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
from typing import Any, Sequence

import numpy as np

from adapters.geometry import load_geometry_npz
from adapters.masks import MaskRecord, save_mask_manifest
from adapters.open_vocab import (
    SAM3_SOURCE_COMMIT,
    FrameSource,
    Sam3Backend,
    load_frame_sources,
    validate_source_checkout,
)
from relground.observation_cache import sha256_file
from relground.observations import LifterConfig, LiftingError, Robust3DLifter
from relground.schemas import RunManifest
from relground.single_view import load_vggt_sam_image


def git_commit(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def require_pinned_sam3_source(sam3_root: Path) -> str:
    validate_source_checkout(
        sam3_root,
        ("sam3/model_builder.py", "sam3/model/sam3_image_processor.py"),
    )
    actual = git_commit(sam3_root)
    if actual != SAM3_SOURCE_COMMIT:
        raise RuntimeError(
            f"SAM 3 source commit mismatch: actual={actual} "
            f"expected={SAM3_SOURCE_COMMIT}"
        )
    return actual


def require_local_checkpoint(value: str | None) -> Path:
    if value is None:
        raise ValueError(
            "--sam3-checkpoint is required; D6 deliberately uses an explicit "
            "local checkpoint and never downloads during a run"
        )
    path = Path(value).expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"SAM 3 checkpoint is missing or empty: {path}")
    return path


def _bounded_score(row: dict[str, Any], name: str) -> float:
    score = float(row[name])
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return score


def load_d5_selection(
    path: str | Path,
    sources: Sequence[FrameSource],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selection_path = Path(path)
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    if payload.get("stage") != "D5":
        raise ValueError("selection artifact stage must be D5")
    query = str(payload.get("query", "")).strip()
    if not query:
        raise ValueError("selection query must not be empty")
    rows = payload.get("frames")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("D6 requires at least two D5-selected frames")
    if int(payload.get("selected_count", -1)) != len(rows):
        raise ValueError("selection selected_count does not match frames")
    requested_k = int(payload.get("requested_k", 0))
    if requested_k < len(rows):
        raise ValueError("selection requested_k is smaller than selected_count")

    sources_by_id = {source.frame_id: source for source in sources}
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for expected_rank, raw in enumerate(rows, start=1):
        if not isinstance(raw, dict):
            raise ValueError("selection frame records must be objects")
        rank = int(raw.get("rank", -1))
        if rank != expected_rank:
            raise ValueError("selection ranks must be contiguous and start at 1")
        frame_id = str(raw.get("frame_id", ""))
        if not frame_id or frame_id in seen:
            raise ValueError("selection frame ids must be non-empty and unique")
        if frame_id not in sources_by_id:
            raise ValueError(f"selected frame is absent from geometry: {frame_id}")
        source = sources_by_id[frame_id]
        if int(raw.get("geometry_index", -1)) != source.geometry_index:
            raise ValueError(f"geometry index mismatch for selected frame {frame_id}")
        seen.add(frame_id)
        retrieval_cosine = float(raw["retrieval_cosine"])
        if not np.isfinite(retrieval_cosine):
            raise ValueError("retrieval_cosine must be finite")
        normalized.append(
            {
                "rank": rank,
                "frame_id": frame_id,
                "geometry_index": source.geometry_index,
                "image_path": str(source.image_path),
                "submap_id": source.submap_id,
                "submap_frame_index": source.submap_frame_index,
                "retrieval_score": _bounded_score(raw, "retrieval_score"),
                "retrieval_cosine": retrieval_cosine,
            }
        )
    return payload, normalized


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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
        alpha = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 105, mode="L")
        overlay = Image.new("RGBA", image.size, color + (0,))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(image)
    for index, (box, score) in enumerate(zip(boxes, scores)):
        color = palette[index % len(palette)]
        coordinates = tuple(float(value) for value in box)
        draw.rectangle(coordinates, outline=color + (255,), width=5)
        draw.text(
            (coordinates[0] + 4, max(0.0, coordinates[1] - 16)),
            f"{query} {float(score):.3f}",
            fill=color + (255,),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)


def _load_inputs(
    args: argparse.Namespace,
) -> tuple[Any, list[FrameSource], dict[str, Any], list[dict[str, Any]]]:
    project_root = Path(args.project_root).resolve()
    geometry = load_geometry_npz(args.geometry)
    sources = load_frame_sources(
        args.geometry_manifest,
        geometry.frame_ids,
        project_root=project_root,
    )
    selection, selected = load_d5_selection(args.selection, sources)
    return geometry, sources, selection, selected


def check_only(args: argparse.Namespace) -> int:
    sam3_root = Path(args.sam3_root).resolve()
    source_commit = require_pinned_sam3_source(sam3_root)
    checkpoint = require_local_checkpoint(args.sam3_checkpoint)
    geometry, _sources, selection, selected = _load_inputs(args)
    retrieval_query = str(selection["query"]).strip()
    segmentation_query = (
        args.sam_query.strip() if args.sam_query is not None else retrieval_query
    )
    preprocess = []
    for row in selected:
        frame = geometry.get(row["frame_id"])
        image, transform = load_vggt_sam_image(
            row["image_path"],
            frame.point_map.shape[:2],
        )
        preprocess.append(
            {
                "frame_id": row["frame_id"],
                "sam_input_size": list(image.size),
                "transform": transform.to_dict(),
            }
        )
    payload = {
        "status": "SOURCE_READY",
        "stage": "D6",
        "inference_executed": False,
        "query": segmentation_query,
        "retrieval_query": retrieval_query,
        "selected_frames": [row["frame_id"] for row in selected],
        "point_map_shape": list(geometry.point_maps.shape[1:]),
        "preprocess": preprocess,
        "baseline_id": "B2-topk-multiframe",
        "mask_resizing_after_sam": False,
        "source_commits": {"sam3": source_commit},
        "sam3_checkpoint": str(checkpoint),
        "sam3_checkpoint_bytes": checkpoint.stat().st_size,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    project_root = Path(args.project_root).resolve()
    sam3_root = Path(args.sam3_root).resolve()
    source_commit = require_pinned_sam3_source(sam3_root)
    checkpoint = require_local_checkpoint(args.sam3_checkpoint)
    geometry, _sources, selection, selected = _load_inputs(args)
    retrieval_query = str(selection["query"]).strip()
    query = args.sam_query.strip() if args.sam_query is not None else retrieval_query

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; choose a new directory"
        )
    masks_dir = output_dir / "masks"
    points_dir = output_dir / "points"
    previews_dir = output_dir / "previews"
    sam_inputs_dir = output_dir / "sam_inputs"
    preprocess_dir = output_dir / "preprocess"
    for directory in (
        masks_dir,
        points_dir,
        previews_dir,
        sam_inputs_dir,
        preprocess_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    lifter_config = LifterConfig(
        confidence_threshold=args.geometry_confidence_threshold,
        min_points=args.min_points,
        outlier_mad_scale=args.outlier_mad_scale,
    )
    lifter = Robust3DLifter(lifter_config)
    mask_records: list[MaskRecord] = []
    observations: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    per_frame: list[dict[str, Any]] = []

    backend = Sam3Backend(
        sam3_root,
        checkpoint_path=checkpoint,
        confidence_threshold=args.sam_threshold,
        device=args.device,
    )
    torch = backend.torch
    try:
        for row in selected:
            frame_id = row["frame_id"]
            image_path = Path(row["image_path"])
            frame = geometry.get(frame_id)
            target_shape = frame.point_map.shape[:2]
            image, transform = load_vggt_sam_image(
                image_path,
                target_shape,
            )
            sam_input_ref = f"sam_inputs/{frame_id}.png"
            preprocess_ref = f"preprocess/{frame_id}.json"
            image.save(output_dir / sam_input_ref)
            save_json(
                output_dir / preprocess_ref,
                {
                    "schema_version": transform.schema_version,
                    "frame_id": frame_id,
                    "source_image": str(image_path),
                    "sam_input": sam_input_ref,
                    "sam_input_sha256": sha256_file(
                        output_dir / sam_input_ref
                    ),
                    "transform": transform.to_dict(),
                },
            )
            segmentation = backend.segment(image, query)
            if tuple(segmentation.masks.shape[1:]) != tuple(target_shape):
                raise RuntimeError(
                    "SAM masks must directly match the VGGT point grid: "
                    f"{segmentation.masks.shape[1:]} != {target_shape}"
                )
            preview_ref = f"previews/{frame_id}.png"
            save_preview(
                output_dir / sam_input_ref,
                segmentation.masks,
                segmentation.boxes_xyxy,
                segmentation.scores,
                output_dir / preview_ref,
                query,
            )

            lifted_count = 0
            rejected_count = 0
            for index, (mask, box, sam_score) in enumerate(
                zip(
                    segmentation.masks,
                    segmentation.boxes_xyxy,
                    segmentation.scores,
                )
            ):
                obs_id = f"d6_{frame_id}_{index:03d}"
                mask_ref = f"masks/{obs_id}.npy"
                points_ref = f"points/{obs_id}.npz"
                np.save(output_dir / mask_ref, np.asarray(mask, dtype=bool))
                mask_records.append(
                    MaskRecord(
                        obs_id=obs_id,
                        frame_id=frame_id,
                        class_text=query,
                        mask_ref=mask_ref,
                        retrieval_score=row["retrieval_score"],
                        sam_score=float(sam_score),
                    )
                )
                try:
                    observation, points = lifter.make_observation(
                        obs_id=obs_id,
                        class_text=query,
                        frame_id=frame_id,
                        mask=mask,
                        point_map=frame.point_map,
                        confidence_map=frame.confidence_map,
                        world_from_camera=frame.world_from_camera,
                        retrieval_score=row["retrieval_score"],
                        sam_score=float(sam_score),
                        mask_ref=mask_ref,
                        points_ref=points_ref,
                    )
                    observation.metadata.update(
                        {
                            "selected_rank": row["rank"],
                            "retrieval_cosine": row["retrieval_cosine"],
                            "box_xyxy": np.asarray(box, dtype=float).tolist(),
                            "baseline_id": "B2-topk-multiframe",
                            "sam_mask_shape": list(mask.shape),
                            "mask_resizing_after_sam": False,
                            "sam_input_ref": sam_input_ref,
                            "preprocess_ref": preprocess_ref,
                        }
                    )
                    np.savez_compressed(
                        output_dir / points_ref,
                        points=points.astype(np.float32),
                    )
                    observations.append(observation.to_dict())
                    lifted_count += 1
                except LiftingError as error:
                    rejected.append(
                        {
                            "obs_id": obs_id,
                            "frame_id": frame_id,
                            "selected_rank": row["rank"],
                            "sam_score": float(sam_score),
                            "box_xyxy": np.asarray(box, dtype=float).tolist(),
                            "reason": str(error),
                        }
                    )
                    rejected_count += 1

            per_frame.append(
                {
                    "rank": row["rank"],
                    "frame_id": frame_id,
                    "geometry_index": row["geometry_index"],
                    "retrieval_score": row["retrieval_score"],
                    "sam_instances": len(segmentation.masks),
                    "lifted_instances": lifted_count,
                    "rejected_instances": rejected_count,
                    "preview": preview_ref,
                    "sam_input": sam_input_ref,
                    "preprocess": preprocess_ref,
                    "mask_resizing_after_sam": False,
                }
            )
    finally:
        backend.close()
        del backend
        gc.collect()

    mask_manifest_path = output_dir / "masks.json"
    observations_path = output_dir / "observations.json"
    selection_snapshot_path = output_dir / "selection.json"
    save_mask_manifest(mask_manifest_path, mask_records)
    save_json(
        observations_path,
        {"schema_version": "0.1", "observations": observations},
    )
    save_json(selection_snapshot_path, selection)

    frames_with_masks = [
        row["frame_id"] for row in per_frame if row["sam_instances"] > 0
    ]
    frames_with_lifted = [
        row["frame_id"] for row in per_frame if row["lifted_instances"] > 0
    ]
    status = (
        "PASS"
        if len(frames_with_lifted) >= 2
        else "INSUFFICIENT_MULTIFRAME_3D_EVIDENCE"
    )
    result = {
        "schema_version": "0.1",
        "status": status,
        "stage": "D6",
        "baseline_id": "B2-topk-multiframe",
        "backend": (
            "D5 top-K + VGGT-preprocessed SAM 3 + Robust3DLifter"
        ),
        "mask_resizing_after_sam": False,
        "query": query,
        "retrieval_query": retrieval_query,
        "selection_source": str(Path(args.selection)),
        "requested_k": int(selection["requested_k"]),
        "selected_frames": selected,
        "processed_frames": per_frame,
        "sam_instances": len(mask_records),
        "lifted_instances": len(observations),
        "rejected_instances": rejected,
        "frames_with_masks": frames_with_masks,
        "frames_with_lifted_observations": frames_with_lifted,
        "lifter_config": {
            "confidence_threshold": lifter_config.confidence_threshold,
            "min_points": lifter_config.min_points,
            "outlier_mad_scale": lifter_config.outlier_mad_scale,
        },
        "sam_threshold": args.sam_threshold,
        "source_commits": {"sam3": source_commit},
        "artifacts": {
            "selection": selection_snapshot_path.name,
            "mask_manifest": mask_manifest_path.name,
            "observations": observations_path.name,
            "previews": previews_dir.name,
            "sam_inputs": sam_inputs_dir.name,
            "preprocess": preprocess_dir.name,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_json(output_dir / "d6_result.json", result)

    peak_vram = None
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    RunManifest(
        git_sha=git_commit(project_root),
        env_lock="scripts/bootstrap_open_vocab.sh",
        dataset_split=Path(args.geometry).parent.name,
        seed=0,
        config={
            "pipeline": result["backend"],
            "query": query,
            "retrieval_query": retrieval_query,
            "selection": str(Path(args.selection)),
            "geometry": str(Path(args.geometry)),
            "geometry_manifest": str(Path(args.geometry_manifest)),
            "sam_threshold": args.sam_threshold,
            "lifter_config": result["lifter_config"],
            "source_commits": result["source_commits"],
            "sam3_checkpoint": str(checkpoint),
        },
        command=shlex.join(
            [sys.executable, "-m", "scripts.run_sam_topk_lifting", *sys.argv[1:]]
        ),
        runtime_seconds=time.perf_counter() - started,
        peak_vram_mb=peak_vram,
    ).save(output_dir / "run_manifest.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 2


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    sam3_root = root / "third_party" / "VGGT-SLAM" / "third_party" / "sam3"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        default="runs/office-loop-d5-trash-can/topk_5.json",
    )
    parser.add_argument("--geometry", default="runs/office-loop/geometry.npz")
    parser.add_argument(
        "--geometry-manifest",
        default="runs/office-loop/geometry.manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/office-loop-d6-controlled-trash-can",
    )
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--sam3-root", default=str(sam3_root))
    parser.add_argument("--sam3-checkpoint")
    parser.add_argument("--sam-query")
    parser.add_argument("--sam-threshold", type=float, default=0.5)
    parser.add_argument("--geometry-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--min-points", type=int, default=30)
    parser.add_argument("--outlier-mad-scale", type=float, default=3.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--check-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 <= args.sam_threshold <= 1.0:
        raise ValueError("sam-threshold must be in [0, 1]")
    if args.sam_query is not None and not args.sam_query.strip():
        raise ValueError("sam-query must not be empty")
    LifterConfig(
        confidence_threshold=args.geometry_confidence_threshold,
        min_points=args.min_points,
        outlier_mad_scale=args.outlier_mad_scale,
    )
    return check_only(args) if args.check_only else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
