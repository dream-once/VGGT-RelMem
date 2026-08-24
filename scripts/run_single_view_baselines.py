"""Run controlled D4 B0/B1 baselines from one PE/SAM inference pass."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shlex
import sys
import time
from typing import Any

import numpy as np

from adapters.geometry import load_geometry_npz
from adapters.masks import MaskRecord, save_mask_manifest
from adapters.open_vocab import (
    PerceptionEncoderBackend,
    Sam3Backend,
    load_frame_sources,
    select_top1,
)
from relground.observation_cache import sha256_file
from relground.observations import LifterConfig, LiftingError, Robust3DLifter
from relground.schemas import ObjectObservation, RunManifest
from relground.single_view import (
    BASELINE_SCHEMA_VERSION,
    B0_OFFICIAL,
    B1_ROBUST_SINGLE_VIEW,
    SUPPORTED_SINGLE_VIEW_BASELINES,
    load_vggt_sam_image,
    make_official_observation,
)
from scripts.run_open_vocab_top1 import (
    check_imports,
    git_commit,
    local_sam3_checkpoint,
    require_pinned_sources,
    save_preview,
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def observation_payload(
    baseline_id: str,
    observations: list[ObjectObservation],
) -> dict[str, Any]:
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "baseline_id": baseline_id,
        "observations": [value.to_dict() for value in observations],
    }


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
    if args.max_frames is not None:
        sources = sources[: args.max_frames]
    if not sources:
        raise ValueError("no geometry frames selected")

    preprocess = []
    for source in sources:
        frame = geometry.get(source.frame_id)
        image, transform = load_vggt_sam_image(
            source.image_path,
            frame.point_map.shape[:2],
        )
        preprocess.append(
            {
                "frame_id": source.frame_id,
                "sam_input_size": list(image.size),
                "transform": transform.to_dict(),
            }
        )
    imports = check_imports(pe_root, sam3_root)
    checkpoint, inference_ready, checkpoint_note = local_sam3_checkpoint(
        args.sam3_checkpoint
    )
    payload = {
        "status": (
            "READY"
            if imports["cuda_available"] and inference_ready
            else "SOURCE_READY"
        ),
        "inference_ready": bool(
            imports["cuda_available"] and inference_ready
        ),
        "controlled_baselines": list(SUPPORTED_SINGLE_VIEW_BASELINES),
        "shared_inputs": [
            "PE top-1 frame",
            "VGGT-preprocessed SAM image",
            "SAM masks",
            "VGGT point map and world transform",
        ],
        "frames": len(sources),
        "point_map_shape": list(geometry.point_maps.shape[1:]),
        "preprocess": preprocess,
        "source_commits": commits,
        "environment": imports,
        "sam3_checkpoint": None if checkpoint is None else str(checkpoint),
        "checkpoint_note": checkpoint_note,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def add_metadata(
    observation: ObjectObservation,
    *,
    instance_id: str,
    box: np.ndarray,
    mask_shape: tuple[int, int],
    baseline_id: str,
    robust_config: LifterConfig | None = None,
) -> None:
    observation.metadata.update(
        {
            "baseline_id": baseline_id,
            "shared_instance_id": instance_id,
            "box_xyxy": np.asarray(box, dtype=float).tolist(),
            "sam_mask_shape": list(mask_shape),
        }
    )
    if robust_config is not None:
        observation.metadata["lifting"] = {
            "mask_indexing": "direct",
            "confidence_threshold": robust_config.confidence_threshold,
            "min_points": robust_config.min_points,
            "outlier_filter": "radial-mad",
            "outlier_mad_scale": robust_config.outlier_mad_scale,
            "obb": "robust-pca",
        }


def reject(
    rejected: list[dict[str, Any]],
    *,
    instance_id: str,
    sam_score: float,
    box: np.ndarray,
    error: Exception,
) -> None:
    rejected.append(
        {
            "shared_instance_id": instance_id,
            "sam_score": float(sam_score),
            "box_xyxy": np.asarray(box, dtype=float).tolist(),
            "reason": str(error),
        }
    )


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

    checkpoint, can_infer, checkpoint_note = local_sam3_checkpoint(
        args.sam3_checkpoint
    )
    if checkpoint is None and not can_infer:
        raise RuntimeError(
            "SAM 3 gated checkpoint is unavailable: " + checkpoint_note
        )

    pe_backend = PerceptionEncoderBackend(
        pe_root,
        config=args.pe_config,
        checkpoint_path=args.pe_checkpoint,
        device=args.device,
    )
    embeddings = pe_backend.encode_images(
        [source.image_path for source in sources],
        batch_size=args.batch_size,
    )
    text_embedding = pe_backend.encode_text(args.query)
    match = select_top1(
        [source.frame_id for source in sources],
        embeddings,
        text_embedding,
    )
    selected = sources[match.index]
    pe_backend.close()
    del pe_backend
    gc.collect()

    frame = geometry.get(selected.frame_id)
    sam_image, transform = load_vggt_sam_image(
        selected.image_path,
        frame.point_map.shape[:2],
    )
    sam_input_path = output_dir / "sam_input.png"
    preprocess_path = output_dir / "preprocess.json"
    sam_image.save(sam_input_path)
    write_json(
        preprocess_path,
        {
            "schema_version": transform.schema_version,
            "frame_id": selected.frame_id,
            "source_image": str(selected.image_path),
            "transform": transform.to_dict(),
        },
    )

    sam_backend = Sam3Backend(
        sam3_root,
        checkpoint_path=checkpoint,
        confidence_threshold=args.sam_threshold,
        device=args.device,
    )
    segmentation = sam_backend.segment(sam_image, args.query)
    sam_backend.close()
    del sam_backend
    gc.collect()

    target_shape = frame.point_map.shape[:2]
    if tuple(segmentation.masks.shape[1:]) != tuple(target_shape):
        raise RuntimeError(
            "controlled SAM masks must directly match the VGGT point grid: "
            f"{segmentation.masks.shape[1:]} != {target_shape}"
        )

    masks_dir = output_dir / "masks"
    b0_points_dir = output_dir / "b0_official" / "points"
    b1_points_dir = output_dir / "b1_robust_single_view" / "points"
    for directory in (masks_dir, b0_points_dir, b1_points_dir):
        directory.mkdir(parents=True, exist_ok=True)

    robust_config = LifterConfig(
        confidence_threshold=args.geometry_confidence_threshold,
        min_points=args.min_points,
        outlier_mad_scale=args.outlier_mad_scale,
    )
    robust_lifter = Robust3DLifter(robust_config)
    mask_records: list[MaskRecord] = []
    b0_observations: list[ObjectObservation] = []
    b1_observations: list[ObjectObservation] = []
    b0_rejected: list[dict[str, Any]] = []
    b1_rejected: list[dict[str, Any]] = []

    for index, (mask, box, sam_score) in enumerate(
        zip(
            segmentation.masks,
            segmentation.boxes_xyxy,
            segmentation.scores,
        )
    ):
        instance_id = f"sam_{selected.frame_id}_{index:03d}"
        mask_name = f"{instance_id}.npy"
        mask_ref = f"masks/{mask_name}"
        np.save(masks_dir / mask_name, np.asarray(mask, dtype=bool))
        mask_records.append(
            MaskRecord(
                obs_id=instance_id,
                frame_id=selected.frame_id,
                class_text=args.query,
                mask_ref=mask_ref,
                retrieval_score=match.score,
                sam_score=float(sam_score),
            )
        )

        b0_obs_id = f"b0_{selected.frame_id}_{index:03d}"
        b0_points_ref = f"b0_official/points/{b0_obs_id}.npz"
        try:
            observation, points = make_official_observation(
                obs_id=b0_obs_id,
                class_text=args.query,
                frame_id=selected.frame_id,
                mask=mask,
                point_map=frame.point_map,
                global_from_submap=frame.world_from_camera,
                retrieval_score=match.score,
                sam_score=float(sam_score),
                mask_ref=mask_ref,
                points_ref=b0_points_ref,
            )
            add_metadata(
                observation,
                instance_id=instance_id,
                box=box,
                mask_shape=tuple(mask.shape),
                baseline_id=B0_OFFICIAL,
            )
            np.savez_compressed(
                output_dir / b0_points_ref,
                points=points.astype(np.float32),
            )
            b0_observations.append(observation)
        except (LiftingError, ValueError, np.linalg.LinAlgError) as error:
            reject(
                b0_rejected,
                instance_id=instance_id,
                sam_score=float(sam_score),
                box=box,
                error=error,
            )

        b1_obs_id = f"b1_{selected.frame_id}_{index:03d}"
        b1_points_ref = (
            f"b1_robust_single_view/points/{b1_obs_id}.npz"
        )
        try:
            observation, points = robust_lifter.make_observation(
                obs_id=b1_obs_id,
                class_text=args.query,
                frame_id=selected.frame_id,
                mask=mask,
                point_map=frame.point_map,
                confidence_map=frame.confidence_map,
                world_from_camera=frame.world_from_camera,
                retrieval_score=match.score,
                sam_score=float(sam_score),
                mask_ref=mask_ref,
                points_ref=b1_points_ref,
            )
            add_metadata(
                observation,
                instance_id=instance_id,
                box=box,
                mask_shape=tuple(mask.shape),
                baseline_id=B1_ROBUST_SINGLE_VIEW,
                robust_config=robust_config,
            )
            np.savez_compressed(
                output_dir / b1_points_ref,
                points=points.astype(np.float32),
            )
            b1_observations.append(observation)
        except (LiftingError, ValueError, np.linalg.LinAlgError) as error:
            reject(
                b1_rejected,
                instance_id=instance_id,
                sam_score=float(sam_score),
                box=box,
                error=error,
            )

    mask_manifest_path = output_dir / "masks.json"
    b0_observations_path = output_dir / "b0_official" / "observations.json"
    b1_observations_path = (
        output_dir / "b1_robust_single_view" / "observations.json"
    )
    preview_path = output_dir / "preview.png"
    result_path = output_dir / "single_view_result.json"
    save_mask_manifest(mask_manifest_path, mask_records)
    write_json(
        b0_observations_path,
        observation_payload(B0_OFFICIAL, b0_observations),
    )
    write_json(
        b1_observations_path,
        observation_payload(B1_ROBUST_SINGLE_VIEW, b1_observations),
    )
    save_preview(
        sam_input_path,
        segmentation.masks,
        segmentation.boxes_xyxy,
        segmentation.scores,
        preview_path,
        args.query,
    )

    baseline_records = {
        B0_OFFICIAL: {
            "method": "direct mask indexing + finite-only filtering + upstream PCA OBB",
            "observations": str(b0_observations_path.relative_to(output_dir)),
            "lifted_instances": len(b0_observations),
            "rejected_instances": b0_rejected,
        },
        B1_ROBUST_SINGLE_VIEW: {
            "method": "confidence filtering + radial MAD + minimum points + robust PCA OBB",
            "observations": str(b1_observations_path.relative_to(output_dir)),
            "lifted_instances": len(b1_observations),
            "rejected_instances": b1_rejected,
        },
    }
    status = (
        "PASS"
        if all(
            record["lifted_instances"] > 0
            for record in baseline_records.values()
        )
        else "NO_VALID_OBSERVATION_FOR_EVERY_BASELINE"
    )
    result = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "status": status,
        "stage": "D4-controlled-correction",
        "query": args.query,
        "top1": {
            "frame_id": selected.frame_id,
            "geometry_index": selected.geometry_index,
            "source_image": str(selected.image_path),
            "retrieval_score": match.score,
            "retrieval_cosine": match.cosine,
        },
        "controlled_inputs": {
            "sam_input": sam_input_path.name,
            "sam_input_sha256": sha256_file(sam_input_path),
            "preprocess": preprocess_path.name,
            "preprocess_sha256": sha256_file(preprocess_path),
            "mask_manifest": mask_manifest_path.name,
            "sam_mask_shape": list(target_shape),
            "sam_instances": len(mask_records),
            "mask_resizing_after_sam": False,
        },
        "baselines": baseline_records,
        "source_commits": commits,
        "artifacts": {
            "preview": preview_path.name,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(result_path, result)

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
            "pipeline": "controlled single-view B0/B1",
            "query": args.query,
            "geometry": str(Path(args.geometry)),
            "geometry_manifest": str(Path(args.geometry_manifest)),
            "pe_config": args.pe_config,
            "sam_threshold": args.sam_threshold,
            "robust_lifter": {
                "confidence_threshold": robust_config.confidence_threshold,
                "min_points": robust_config.min_points,
                "outlier_mad_scale": robust_config.outlier_mad_scale,
            },
            "source_commits": commits,
            "max_frames": args.max_frames,
            "checkpoint_note": checkpoint_note,
        },
        command=shlex.join(
            [
                sys.executable,
                "-m",
                "scripts.run_single_view_baselines",
                *sys.argv[1:],
            ]
        ),
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
    parser.add_argument(
        "--geometry-manifest",
        default="runs/office-loop/geometry.manifest.json",
    )
    parser.add_argument("--query", default="trash can")
    parser.add_argument(
        "--output-dir",
        default="runs/office-loop-single-view-trash-can",
    )
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument(
        "--pe-root",
        default=str(upstream / "perception_models"),
    )
    parser.add_argument("--sam3-root", default=str(upstream / "sam3"))
    parser.add_argument("--pe-config", default="PE-Core-L14-336")
    parser.add_argument("--pe-checkpoint")
    parser.add_argument("--sam3-checkpoint")
    parser.add_argument("--sam-threshold", type=float, default=0.5)
    parser.add_argument(
        "--geometry-confidence-threshold",
        type=float,
        default=0.5,
    )
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
