"""Run official VGGT-SLAM 2.0 geometry and export the project contract."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import glob
import json
import os
import shlex
import subprocess
import sys
import time
import types

from adapters.vggt_slam import export_solver_geometry, validate_upstream_layout
from relground.schemas import RunManifest

DEFAULT_UPSTREAM = "third_party/VGGT-SLAM"
DEFAULT_MODEL_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


def _select_image_names(
    names: list[str],
    *,
    frame_start: int = 0,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> list[str]:
    """Select a deterministic strided subsequence after numeric sorting."""

    if frame_start < 0:
        raise ValueError("frame_start must be non-negative")
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive when provided")
    selected = names[frame_start::frame_stride]
    if max_frames is not None:
        selected = selected[:max_frames]
    return selected


def _commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False, capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _prepare_imports(upstream: Path) -> None:
    for path in (upstream, upstream / "third_party/vggt", upstream / "third_party/salad"):
        if path.exists():
            sys.path.insert(0, str(path))


def _runtime_check(upstream: Path) -> dict[str, str]:
    """Import one dependency per process to limit peak RAM in CPU-only mode."""

    search_paths = [
        str(upstream),
        str(upstream / "third_party/vggt"),
        str(upstream / "third_party/salad"),
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        search_paths + [environment.get("PYTHONPATH", "")]
    )
    status: dict[str, str] = {}
    modules = (
        "numpy",
        "torch",
        "torchvision",
        "cv2",
        "gtsam",
        "open3d",
        "vggt.models.vggt",
        "salad.eval",
        "vggt_slam.graph",
    )
    code = (
        "import importlib, json, sys; "
        "m=importlib.import_module(sys.argv[1]); "
        "print(json.dumps(str(getattr(m, '__version__', 'ok'))))"
    )
    for module in modules:
        result = subprocess.run(
            [sys.executable, "-c", code, module],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        if result.returncode == 0:
            try:
                status[module] = json.loads(result.stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError):
                status[module] = "ok"
        else:
            detail = (result.stderr or result.stdout).strip().splitlines()
            message = detail[-1] if detail else "no output"
            status[module] = f"ERROR (exit {result.returncode}): {message}"
    return status


def _install_headless_viewer_stub() -> None:
    """Avoid starting a Viser server when this runner never visualizes."""

    viewer_module = types.ModuleType("vggt_slam.viewer")

    class HeadlessViewer:
        def __init__(self) -> None:
            pass

    viewer_module.Viewer = HeadlessViewer
    sys.modules["vggt_slam.viewer"] = viewer_module


class _DisabledImageRetrieval:
    """No-op replacement for upstream SALAD when loop closure is disabled."""

    def get_all_submap_embeddings(self, _submap):
        return []

    def find_loop_closures(self, _map, _submap, **_kwargs):
        return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", default=DEFAULT_UPSTREAM)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--image-folder")
    parser.add_argument("--output", default="runs/vggt-geometry/geometry.npz")
    parser.add_argument("--submap-size", type=int, default=16)
    parser.add_argument("--overlap", type=int, default=1)
    parser.add_argument("--max-loops", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--min-disparity", type=float, default=50.0)
    parser.add_argument("--conf-percentile", type=float, default=25.0)
    parser.add_argument("--disable-flow-filter", action="store_true")
    parser.add_argument("--model-url", default=DEFAULT_MODEL_URL)
    args = parser.parse_args()

    upstream = validate_upstream_layout(args.upstream)
    if args.check_only:
        modules = _runtime_check(upstream)
        print(json.dumps({
            "upstream": str(upstream),
            "commit": _commit(upstream),
            "cuda_inference_run": False,
            "modules": modules,
        }, indent=2))
        if any(value.startswith("ERROR") for value in modules.values()):
            raise SystemExit(1)
        return
    if not args.image_folder:
        parser.error("--image-folder is required unless --check-only is used")
    if (
        args.submap_size < 1
        or args.overlap not in (0, 1)
        or args.max_loops < 0
        or args.frame_start < 0
        or args.frame_stride < 1
        or (args.max_frames is not None and args.max_frames < 1)
    ):
        parser.error(
            "--submap-size must be positive, --overlap must be 0 or 1, "
            "--max-loops/--frame-start non-negative, and "
            "--frame-stride/--max-frames positive"
        )

    _prepare_imports(upstream)
    _install_headless_viewer_stub()
    import cv2
    import torch
    from vggt.models.vggt import VGGT
    import vggt_slam.solver as solver_module
    import vggt_slam.slam_utils as utils

    if not torch.cuda.is_available():
        raise RuntimeError("VGGT-SLAM geometry run requires CUDA")
    if args.max_loops == 0:
        solver_module.ImageRetrieval = _DisabledImageRetrieval
    Solver = solver_module.Solver
    source_commit = _commit(upstream)
    names = [
        item for item in glob.glob(os.path.join(args.image_folder, "*"))
        if not any(token in os.path.basename(item).lower() for token in ("depth", "txt", "db"))
    ]
    names = utils.sort_images_by_number(names)
    names = _select_image_names(
        names,
        frame_start=args.frame_start,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )
    if len(names) < 2:
        raise ValueError("at least two numbered images are required")

    started_at = time.perf_counter()
    solver = Solver(init_conf_threshold=args.conf_percentile, vis_imgs=False)
    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url(args.model_url))
    model.eval().to(torch.bfloat16).to("cuda")

    pending: list[str] = []
    for position, image_name in enumerate(names):
        keep = True
        if not args.disable_flow_filter:
            image = cv2.imread(image_name)
            if image is None:
                raise ValueError(f"failed to read image: {image_name}")
            keep = bool(solver.flow_tracker.compute_disparity(image, args.min_disparity, False))
        if keep:
            pending.append(image_name)
        last = position == len(names) - 1
        if len(pending) == args.submap_size + args.overlap or (last and len(pending) >= 2):
            predictions = solver.run_predictions(pending, model, args.max_loops, None, None)
            solver.add_points(predictions)
            solver.graph.optimize()
            pending = pending[-args.overlap:] if args.overlap else []

    summary = export_solver_geometry(
        solver, Path(args.output), source_commit=source_commit
    )
    output = Path(args.output)
    manifest = RunManifest(
        git_sha=source_commit,
        env_lock="scripts/bootstrap_vggt_geom.sh",
        dataset_split=Path(args.image_folder).name,
        seed=0,
        config={
            "pipeline": "MIT-SPARK/VGGT-SLAM 2.0 geometry",
            "upstream_path": str(upstream),
            "upstream_commit": source_commit,
            "project_git_sha": _commit(Path.cwd()),
            "salad_commit": _commit(upstream / "third_party/salad"),
            "vggt_commit": _commit(upstream / "third_party/vggt"),
            "model_url": args.model_url,
            "submap_size": args.submap_size,
            "overlap": args.overlap,
            "max_loops": args.max_loops,
            "max_frames": args.max_frames,
            "frame_start": args.frame_start,
            "frame_stride": args.frame_stride,
            "selected_images": [Path(name).name for name in names],
            "min_disparity": args.min_disparity,
            "conf_percentile": args.conf_percentile,
            "flow_filter": not args.disable_flow_filter,
            "frame_count": summary.frame_count,
            "geometry_path": str(output),
        },
        command=shlex.join(
            [sys.executable, "-m", "scripts.run_vggt_geometry", *sys.argv[1:]]
        ),
        runtime_seconds=time.perf_counter() - started_at,
        peak_vram_mb=torch.cuda.max_memory_allocated() / (1024 * 1024),
    )
    manifest.save(output.parent / "run_manifest.json")
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
