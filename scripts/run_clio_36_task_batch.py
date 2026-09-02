"""Materialize the frozen 18+18 Clio task set with the existing pipeline.

This is a thin subprocess orchestrator.  It does not change retrieval,
segmentation, lifting, or association policy.  Use --dry-run to audit the
complete command plan without data, model weights, or a GPU.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

from relground.clio_retrieval_evaluation import slugify_task


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    query_manifest: Path
    image_folder: Path
    run_root: Path


@dataclass(frozen=True)
class TaskLayout:
    retrieval: Path
    d6: Path
    d7: Path
    d8: Path
    a2_prediction: Path


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_queries(root: Path, spec: SceneSpec) -> list[dict[str, str]]:
    payload = json.loads((root / spec.query_manifest).read_text(encoding="utf-8"))
    if payload.get("scene_id") != spec.scene_id:
        raise ValueError(f"query manifest scene mismatch: {spec.scene_id}")
    role = str(payload["role"])
    queries = [
        {"task": str(item["task"]), "sam_query": str(item["sam_query"])}
        for item in payload["queries"]
        if item["split"] == role
    ]
    if len(queries) != 18:
        raise ValueError(f"{spec.scene_id} batch must contain exactly 18 tasks")
    if len({item["task"] for item in queries}) != 18:
        raise ValueError(f"{spec.scene_id} batch contains duplicate tasks")
    return queries


def _layout(spec: SceneSpec, task: str) -> TaskLayout:
    slug = slugify_task(task)
    prefix = ""
    if spec.scene_id == "apartment" and task != "bring me a pillow":
        prefix = "dev-"
    return TaskLayout(
        retrieval=spec.run_root / f"retrieval-{slug}",
        d6=spec.run_root / f"{prefix}d6-{slug}-k5",
        d7=spec.run_root / f"{prefix}d7-{slug}-k5",
        d8=spec.run_root / f"{prefix}d8-{slug}-k5",
        a2_prediction=spec.run_root / f"{prefix}a2-{slug}-k5" / "prediction",
    )


def _commands(
    *,
    root: Path,
    spec: SceneSpec,
    task: str,
    sam_query: str,
    open_vocab_env: Path,
    sam3_checkpoint: Path,
) -> tuple[list[list[str]], TaskLayout]:
    layout = _layout(spec, task)
    geometry = spec.run_root / "geometry.npz"
    geometry_manifest = spec.run_root / "geometry.manifest.json"
    anchor_poses = spec.run_root / "geometry.anchor_poses.json"
    gpu_python = [
        "conda", "run", "--no-capture-output", "-p",
        str(open_vocab_env), "python",
    ]
    cpu_python = [sys.executable]
    commands = [
        gpu_python + [
            "-m", "scripts.run_pe_topk",
            "--project-root", str(root),
            "--geometry", str(geometry),
            "--geometry-manifest", str(geometry_manifest),
            "--anchor-poses", str(anchor_poses),
            "--query", task,
            "--k", "1", "3", "5",
            "--redundancy", "hybrid",
            "--min-frame-gap", "2",
            "--min-camera-distance", "0.15",
            "--min-view-angle-deg", "3.0",
            "--output-dir", str(layout.retrieval),
        ],
        gpu_python + [
            "-m", "scripts.run_sam_topk_lifting",
            "--project-root", str(root),
            "--selection", str(layout.retrieval / "topk_5.json"),
            "--geometry", str(geometry),
            "--geometry-manifest", str(geometry_manifest),
            "--sam-query", sam_query,
            "--sam-threshold", "0.5",
            "--geometry-confidence-threshold", "0.5",
            "--min-points", "30",
            "--outlier-mad-scale", "3.5",
            "--sam3-checkpoint", str(sam3_checkpoint),
            "--output-dir", str(layout.d6),
        ],
        cpu_python + [
            "-m", "scripts.cache_scene_observations",
            "--project-root", str(root),
            "--d6-dir", str(layout.d6),
            "--geometry-manifest", str(geometry_manifest),
            "--image-folder", str(spec.image_folder),
            "--scene-id", f"clio-{spec.scene_id}-{slugify_task(task)}",
            "--output-dir", str(layout.d7),
        ],
        cpu_python + [
            "-m", "scripts.prepare_object_memory",
            "--project-root", str(root),
            "--cache", str(layout.d7 / "observations.json"),
            "--output-dir", str(layout.d8),
        ],
        cpu_python + [
            "-m", "scripts.run_a2_association",
            "--project-root", str(root),
            "--memory", str(layout.d8 / "object_memory.json"),
            "--output-dir", str(layout.a2_prediction),
        ],
    ]
    return commands, layout


def build_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = Path(args.project_root).resolve()
    specs = (
        SceneSpec(
            "apartment",
            Path("configs/clio_apartment_queries.json"),
            _resolve(root, args.apartment_images),
            _resolve(root, args.apartment_run_root),
        ),
        SceneSpec(
            "cubicle",
            Path("configs/clio_cubicle_queries.json"),
            _resolve(root, args.cubicle_images),
            _resolve(root, args.cubicle_run_root),
        ),
    )
    plan: list[dict[str, Any]] = []
    for spec in specs:
        for query in _load_queries(root, spec):
            commands, layout = _commands(
                root=root,
                spec=spec,
                task=query["task"],
                sam_query=query["sam_query"],
                open_vocab_env=_resolve(root, args.open_vocab_env),
                sam3_checkpoint=_resolve(root, args.sam3_checkpoint),
            )
            plan.append({
                "scene_id": spec.scene_id,
                "task": query["task"],
                "sam_query": query["sam_query"],
                "commands": commands,
                "downstream_condition": "run D7/D8/A2 only when D6 lifted_instances > 0",
                "d6_result": str(layout.d6 / "d6_result.json"),
            })
    if len(plan) != 36:
        raise AssertionError("Clio batch plan must contain exactly 36 tasks")
    return plan


def _preflight(args: argparse.Namespace, plan: Sequence[dict[str, Any]]) -> None:
    root = Path(args.project_root).resolve()
    required = [
        _resolve(root, args.open_vocab_env),
        _resolve(root, args.sam3_checkpoint),
        _resolve(root, args.apartment_images),
        _resolve(root, args.cubicle_images),
        _resolve(root, args.apartment_run_root) / "geometry.npz",
        _resolve(root, args.apartment_run_root) / "geometry.manifest.json",
        _resolve(root, args.apartment_run_root) / "geometry.anchor_poses.json",
        _resolve(root, args.cubicle_run_root) / "geometry.npz",
        _resolve(root, args.cubicle_run_root) / "geometry.manifest.json",
        _resolve(root, args.cubicle_run_root) / "geometry.anchor_poses.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing batch prerequisites: " + ", ".join(missing))
    collisions = []
    for row in plan:
        first_output = Path(row["commands"][0][-1])
        if first_output.exists():
            collisions.append(str(first_output))
    if collisions:
        raise FileExistsError(
            "batch requires fresh per-task output directories; first collisions: "
            + ", ".join(collisions[:5])
        )


def _run(command: Sequence[str], *, root: Path) -> None:
    subprocess.run(list(command), cwd=root, check=True)


def execute(args: argparse.Namespace, plan: Sequence[dict[str, Any]]) -> dict[str, int]:
    root = Path(args.project_root).resolve()
    _preflight(args, plan)
    materialized = 0
    downstream_skipped = 0
    for row in plan:
        commands = row["commands"]
        _run(commands[0], root=root)
        _run(commands[1], root=root)
        result = json.loads(Path(row["d6_result"]).read_text(encoding="utf-8"))
        if int(result.get("lifted_instances", 0)) == 0:
            downstream_skipped += 1
            continue
        for command in commands[2:]:
            _run(command, root=root)
        materialized += 1
    return {
        "task_count": len(plan),
        "tasks_with_d7_d8_a2": materialized,
        "tasks_without_lifted_observations": downstream_skipped,
    }


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--open-vocab-env", required=True)
    parser.add_argument("--sam3-checkpoint", required=True)
    parser.add_argument("--apartment-images", default="data/clio/apartment/images")
    parser.add_argument("--cubicle-images", default="data/clio/cubicle/images")
    parser.add_argument("--apartment-run-root", default="runs/clio-apartment-dev-v2-lc")
    parser.add_argument("--cubicle-run-root", default="runs/clio-cubicle-heldout-v1")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps({
            "status": "DRY_RUN",
            "task_count": len(plan),
            "gpu_command_count": 2 * len(plan),
            "conditional_cpu_command_count": 3 * len(plan),
            "tasks": plan,
        }, ensure_ascii=False, indent=2))
        return
    result = execute(args, plan)
    print(json.dumps({"status": "PASS", **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
