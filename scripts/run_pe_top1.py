"""Run and record the real Perception Encoder top-1 half of the D4 baseline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shlex
import sys
import time

from adapters.geometry import load_geometry_npz
from adapters.open_vocab import PerceptionEncoderBackend, load_frame_sources, select_top1
from relground.schemas import RunManifest
from scripts.run_open_vocab_top1 import git_commit, require_pinned_sources


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    upstream = root / "third_party" / "VGGT-SLAM" / "third_party"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", default="runs/office-loop/geometry.npz")
    parser.add_argument("--geometry-manifest", default="runs/office-loop/geometry.manifest.json")
    parser.add_argument("--query", default="printer")
    parser.add_argument("--output-dir", default="runs/office-loop-pe-printer")
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--pe-root", default=str(upstream / "perception_models"))
    parser.add_argument("--sam3-root", default=str(upstream / "sam3"))
    parser.add_argument("--pe-config", default="PE-Core-L14-336")
    parser.add_argument("--pe-checkpoint")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.query.strip():
        raise ValueError("query must not be empty")

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
    match = select_top1(
        [source.frame_id for source in sources],
        image_embeddings,
        text_embedding,
    )
    selected = sources[match.index]
    backend.close()
    del backend
    gc.collect()

    from PIL import Image

    preview_path = output_dir / "top1_frame.jpg"
    with Image.open(selected.image_path) as source:
        preview = source.convert("RGB")
    preview.thumbnail((1280, 1280))
    preview.save(preview_path, quality=90)
    result = {
        "schema_version": "0.1",
        "status": "PASS",
        "backend": f"Perception Encoder {args.pe_config}",
        "query": args.query.strip(),
        "searched_frames": len(sources),
        "top1": {
            "frame_id": selected.frame_id,
            "geometry_index": selected.geometry_index,
            "image_path": str(selected.image_path),
            "retrieval_score": match.score,
            "retrieval_cosine": match.cosine,
        },
        "source_commits": commits,
        "artifacts": {"top1_preview": preview_path.name},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    result_path = output_dir / "retrieval.json"
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
    RunManifest(
        git_sha=git_commit(project_root),
        env_lock="scripts/bootstrap_open_vocab.sh",
        dataset_split=Path(args.geometry).parent.name,
        seed=0,
        config={
            "pipeline": result["backend"],
            "query": args.query,
            "geometry": str(Path(args.geometry)),
            "geometry_manifest": str(Path(args.geometry_manifest)),
            "source_commits": commits,
            "max_frames": args.max_frames,
        },
        command=shlex.join([sys.executable, "-m", "scripts.run_pe_top1", *sys.argv[1:]]),
        runtime_seconds=time.perf_counter() - started,
        peak_vram_mb=peak_vram,
    ).save(output_dir / "run_manifest.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
