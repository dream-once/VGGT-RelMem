"""Run the D5 PE top-K frame retriever with temporal/viewpoint suppression."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np

from adapters.geometry import load_anchor_poses, load_geometry_npz
from adapters.open_vocab import (
    PE_SOURCE_COMMIT,
    FrameScore,
    FrameSource,
    PerceptionEncoderBackend,
    load_frame_sources,
    score_frames,
    select_top1,
    validate_source_checkout,
)
from relground.retrieval import (
    FrameCandidate,
    RetrievalConfig,
    TopKFrameRetriever,
    viewpoint_from_world_pose,
)
from relground.schemas import RunManifest


def git_commit(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def require_pinned_pe_source(pe_root: Path) -> str:
    validate_source_checkout(
        pe_root,
        ("core/vision_encoder/pe.py", "core/vision_encoder/transforms.py"),
    )
    actual = git_commit(pe_root)
    if actual != PE_SOURCE_COMMIT:
        raise RuntimeError(
            f"Perception Encoder source commit mismatch: "
            f"actual={actual} expected={PE_SOURCE_COMMIT}"
        )
    return actual


def normalized_k_values(values: Sequence[int]) -> list[int]:
    k_values = sorted(set(int(value) for value in values))
    if not k_values or any(value < 1 for value in k_values):
        raise ValueError("--k values must be positive")
    if 1 not in k_values:
        raise ValueError("--k must include 1 so D5 can verify B0 compatibility")
    return k_values


def retrieval_settings(args: argparse.Namespace) -> dict[str, Any]:
    settings = {
        "redundancy": args.redundancy,
        "min_frame_gap": args.min_frame_gap,
        "min_camera_distance": args.min_camera_distance,
        "min_view_angle_deg": args.min_view_angle_deg,
    }
    RetrievalConfig(top_k=max(normalized_k_values(args.k)), **settings)
    return settings


def build_candidates(
    sources: Sequence[FrameSource],
    scored: Sequence[FrameScore],
    anchor_poses: dict[str, np.ndarray],
) -> list[FrameCandidate]:
    if len(sources) != len(scored):
        raise ValueError("sources and frame scores must have the same length")
    candidates: list[FrameCandidate] = []
    for source, frame_score in zip(sources, scored):
        if source.frame_id != frame_score.frame_id:
            raise ValueError("frame score order does not match geometry sources")
        center, direction = viewpoint_from_world_pose(anchor_poses[source.frame_id])
        candidates.append(
            FrameCandidate(
                frame_id=source.frame_id,
                score=frame_score.score,
                index=source.geometry_index,
                camera_center=center,
                view_direction=direction,
                metadata={
                    "cosine": frame_score.cosine,
                    "image_path": str(source.image_path),
                    "submap_id": source.submap_id,
                    "submap_frame_index": source.submap_frame_index,
                },
            )
        )
    return candidates


def candidate_record(candidate: FrameCandidate, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "frame_id": candidate.frame_id,
        "geometry_index": candidate.index,
        "image_path": candidate.metadata["image_path"],
        "submap_id": candidate.metadata["submap_id"],
        "submap_frame_index": candidate.metadata["submap_frame_index"],
        "retrieval_score": candidate.score,
        "retrieval_cosine": candidate.metadata["cosine"],
        "camera_center": candidate.camera_center.tolist(),
        "view_direction": candidate.view_direction.tolist(),
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_preview(
    selected: Sequence[FrameCandidate],
    output_path: Path,
    query: str,
) -> None:
    from PIL import Image, ImageDraw

    cell_width, cell_height = 420, 260
    columns = min(2, max(1, len(selected)))
    rows = max(1, (len(selected) + columns - 1) // columns)
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, candidate in enumerate(selected):
        with Image.open(candidate.metadata["image_path"]) as source:
            image = source.convert("RGB")
        image.thumbnail((cell_width, cell_height - 34))
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height + 34
        canvas.paste(image, (x + (cell_width - image.width) // 2, y))
        label = (
            f"#{index + 1} {candidate.frame_id} "
            f"score={candidate.score:.6f} query={query}"
        )
        draw.rectangle((x, y - 34, x + cell_width, y), fill="black")
        draw.text((x + 6, y - 25), label, fill="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def check_only(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    pe_root = Path(args.pe_root).resolve()
    source_commit = require_pinned_pe_source(pe_root)
    k_values = normalized_k_values(args.k)
    settings = retrieval_settings(args)
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
    poses = load_anchor_poses(
        args.anchor_poses,
        required_frame_ids=[source.frame_id for source in sources],
    )
    centers = [
        viewpoint_from_world_pose(poses[source.frame_id])[0]
        for source in sources
    ]
    payload = {
        "status": "SOURCE_READY",
        "stage": "D5",
        "inference_executed": False,
        "frames": len(sources),
        "k_values": k_values,
        "retrieval_config": settings,
        "source_commits": {"perception_models": source_commit},
        "camera_path_extent": (
            np.ptp(np.stack(centers), axis=0).tolist()
            if len(centers) > 1
            else [0.0, 0.0, 0.0]
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    k_values = normalized_k_values(args.k)
    settings = retrieval_settings(args)
    project_root = Path(args.project_root).resolve()
    pe_root = Path(args.pe_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    source_commit = require_pinned_pe_source(pe_root)

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
    anchor_poses = load_anchor_poses(
        args.anchor_poses,
        required_frame_ids=[source.frame_id for source in sources],
    )

    backend = PerceptionEncoderBackend(
        pe_root,
        config=args.pe_config,
        checkpoint_path=args.pe_checkpoint,
        device=args.device,
    )
    image_embeddings = backend.encode_images(
        [source.image_path for source in sources],
        batch_size=args.batch_size,
    )
    text_embedding = backend.encode_text(args.query)
    scored = score_frames(
        [source.frame_id for source in sources],
        image_embeddings,
        text_embedding,
    )
    upstream_top1 = select_top1(
        [source.frame_id for source in sources],
        image_embeddings,
        text_embedding,
    )
    peak_vram = None
    if backend.torch.cuda.is_available():
        peak_vram = backend.torch.cuda.max_memory_allocated() / (1024**2)
    backend.close()
    del backend
    gc.collect()

    candidates = build_candidates(sources, scored, anchor_poses)
    raw_config = RetrievalConfig(top_k=len(candidates), redundancy="none")
    raw_ranking = TopKFrameRetriever(raw_config).retrieve(candidates)
    raw_records = [
        candidate_record(candidate, rank)
        for rank, candidate in enumerate(raw_ranking, start=1)
    ]

    selections: dict[str, dict[str, Any]] = {}
    selection_ids: dict[int, list[str]] = {}
    artifacts: dict[str, str] = {}
    for k in k_values:
        config = RetrievalConfig(top_k=k, **settings)
        selected = TopKFrameRetriever(config).retrieve(candidates)
        frame_ids = [candidate.frame_id for candidate in selected]
        selection_ids[k] = frame_ids
        name = f"topk_{k}.json"
        artifact = {
            "schema_version": "0.1",
            "stage": "D5",
            "query": args.query.strip(),
            "requested_k": k,
            "selected_count": len(selected),
            "exhausted_nonredundant_candidates": len(selected) < min(k, len(candidates)),
            "retrieval_config": asdict(config),
            "frames": [
                candidate_record(candidate, rank)
                for rank, candidate in enumerate(selected, start=1)
            ],
        }
        save_json(output_dir / name, artifact)
        selections[str(k)] = {
            "artifact": name,
            "selected_count": len(selected),
            "frame_ids": frame_ids,
        }
        artifacts[f"topk_{k}"] = name

    prefix_consistent = True
    previous: list[str] = []
    for k in k_values:
        current = selection_ids[k]
        if previous != current[: len(previous)]:
            prefix_consistent = False
            break
        previous = current
    top1_compatible = bool(
        selection_ids[1] and selection_ids[1][0] == upstream_top1.frame_id
    )
    status = "PASS" if prefix_consistent and top1_compatible else "FAIL"

    preview_path = output_dir / "topk_preview.png"
    max_selected = TopKFrameRetriever(
        RetrievalConfig(top_k=max(k_values), **settings)
    ).retrieve(candidates)
    save_preview(max_selected, preview_path, args.query)
    artifacts["preview"] = preview_path.name

    result = {
        "schema_version": "0.1",
        "status": status,
        "stage": "D5",
        "backend": f"Perception Encoder {args.pe_config}",
        "query": args.query.strip(),
        "searched_frames": len(candidates),
        "source_commits": {"perception_models": source_commit},
        "pose_source": str(Path(args.anchor_poses)),
        "pose_convention": "world_from_anchor; camera +z is forward",
        "retrieval_config": settings,
        "k_values": k_values,
        "upstream_top1": {
            "frame_id": upstream_top1.frame_id,
            "geometry_index": upstream_top1.index,
            "retrieval_score": upstream_top1.score,
            "retrieval_cosine": upstream_top1.cosine,
        },
        "top1_compatible": top1_compatible,
        "prefix_consistent": prefix_consistent,
        "raw_ranking": raw_records,
        "selections": selections,
        "artifacts": artifacts,
    }
    save_json(output_dir / "retrieval.json", result)

    RunManifest(
        git_sha=git_commit(project_root),
        env_lock="scripts/bootstrap_open_vocab.sh",
        dataset_split=Path(args.geometry).parent.name,
        seed=0,
        config={
            "pipeline": "D5 PE top-K frame retrieval",
            "query": args.query.strip(),
            "geometry": str(Path(args.geometry)),
            "geometry_manifest": str(Path(args.geometry_manifest)),
            "anchor_poses": str(Path(args.anchor_poses)),
            "pe_config": args.pe_config,
            "source_commits": result["source_commits"],
            "max_frames": args.max_frames,
            "k_values": k_values,
            **settings,
        },
        command=shlex.join(
            [sys.executable, "-m", "scripts.run_pe_topk", *sys.argv[1:]]
        ),
        runtime_seconds=time.perf_counter() - started,
        peak_vram_mb=peak_vram,
    ).save(output_dir / "run_manifest.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 2


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    pe_root = root / "third_party" / "VGGT-SLAM" / "third_party" / "perception_models"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", default="runs/office-loop/geometry.npz")
    parser.add_argument(
        "--geometry-manifest",
        default="runs/office-loop/geometry.manifest.json",
    )
    parser.add_argument(
        "--anchor-poses",
        default="runs/office-loop/geometry.anchor_poses.json",
    )
    parser.add_argument("--query", default="trash can")
    parser.add_argument("--output-dir", default="runs/office-loop-d5-trash-can")
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--pe-root", default=str(pe_root))
    parser.add_argument("--pe-config", default="PE-Core-L14-336")
    parser.add_argument("--pe-checkpoint")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument(
        "--redundancy",
        choices=("none", "temporal", "viewpoint", "hybrid"),
        default="hybrid",
    )
    parser.add_argument("--min-frame-gap", type=int, default=3)
    parser.add_argument("--min-camera-distance", type=float, default=0.15)
    parser.add_argument("--min-view-angle-deg", type=float, default=12.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.query.strip():
        raise ValueError("query must not be empty")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("max-frames must be positive")
    normalized_k_values(args.k)
    retrieval_settings(args)
    return check_only(args) if args.check_only else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
