"""Extract label-free PE mask-crop scores for frozen Clio observations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from adapters.open_vocab import PE_SOURCE_COMMIT, PerceptionEncoderBackend
from relground.clio_retrieval_evaluation import slugify_task
from relground.pe_semantic_fusion import (
    PREDICTION_STAGE,
    SCHEMA_VERSION,
    build_crop_variants,
    mean_crop_query_score,
    observation_quality,
    select_semantic_representative,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("prediction source escapes project root") from error


def _rounded(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _rounded(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), 12)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _task_paths(
    scene_id: str,
    run_root: Path,
    task: str,
) -> tuple[Path, Path]:
    slug = slugify_task(task)
    prefix = ""
    if scene_id == "apartment" and task != "bring me a pillow":
        prefix = "dev-"
    return (
        run_root / f"{prefix}d6-{slug}-k5" / "observations.json",
        run_root / f"{prefix}a2-{slug}-k5" / "prediction" / "object_memory.json",
    )


def _encode_pil(
    backend: PerceptionEncoderBackend,
    images: Sequence[Image.Image],
    *,
    batch_size: int,
) -> np.ndarray:
    if not images:
        raise ValueError("at least one crop is required")
    outputs: list[np.ndarray] = []
    for offset in range(0, len(images), batch_size):
        tensors = [
            backend.preprocess(image.convert("RGB"))
            for image in images[offset : offset + batch_size]
        ]
        batch = backend.torch.stack(tensors).to(backend.device)
        with backend.torch.inference_mode():
            features = backend.model.encode_image(batch, normalize=True)
        outputs.append(
            features.detach().float().cpu().numpy().astype(np.float32, copy=False)
        )
    return np.concatenate(outputs, axis=0)


def _validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "post-D21-PE-mask-crop-representative-center-v1",
        "encoder": "PE-Core-L14-336",
        "pe_source_commit": PE_SOURCE_COMMIT,
        "text_source": "sam_query",
        "context_padding_fraction": 0.15,
        "min_padding_pixels": 2,
        "masked_background_value": 127,
        "semantic_score": "mean(context_crop_cosine, masked_crop_cosine)",
        "object_selection": "highest-confidence A2 permanent object unchanged",
        "center_rule": "center of highest-semantic-score observation in selected object",
        "fallback": "Q0 when no A2 permanent object",
        "learned_parameters": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"semantic-fusion config mismatch: {key}")


def build_prediction(
    *,
    project_root: Path,
    scene_id: str,
    query_manifest_path: Path,
    run_root: Path,
    config_path: Path,
    pe_source_root: Path,
    pe_checkpoint: Path,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    root = project_root.resolve()
    query_manifest = json.loads(query_manifest_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    if query_manifest.get("scene_id") != scene_id:
        raise ValueError("query manifest scene does not match --scene-id")
    role = str(query_manifest["role"])
    queries = [
        item for item in query_manifest["queries"]
        if item.get("split") == role
    ]
    if len(queries) != 18:
        raise ValueError("semantic-fusion prediction requires exactly 18 tasks")
    checkpoint_sha256 = _sha256(pe_checkpoint)
    if checkpoint_sha256 != config["pe_checkpoint_sha256"]:
        raise ValueError("PE checkpoint hash does not match frozen config")
    actual_source_commit = subprocess.check_output(
        ["git", "-C", str(pe_source_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if actual_source_commit != config["pe_source_commit"]:
        raise ValueError(
            "PE source checkout does not match frozen config: "
            f"{actual_source_commit}"
        )
    backend = PerceptionEncoderBackend(
        pe_source_root,
        config=str(config["encoder"]),
        checkpoint_path=pe_checkpoint,
        device=device,
    )
    task_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    try:
        for query in queries:
            task = str(query["task"])
            sam_query = str(query["sam_query"])
            observation_path, memory_path = _task_paths(
                scene_id,
                run_root,
                task,
            )
            if not observation_path.is_file():
                raise FileNotFoundError(
                    f"missing D6 observations for {task}: {observation_path}"
                )
            observation_payload = json.loads(
                observation_path.read_text(encoding="utf-8")
            )
            observations = sorted(
                observation_payload["observations"],
                key=lambda item: str(item["obs_id"]),
            )
            crops: list[Image.Image] = []
            crop_metadata: list[tuple[int, int, int, int]] = []
            source_metadata: list[tuple[str, str, str, str]] = []
            for observation in observations:
                image_path = (
                    observation_path.parent
                    / str(observation["metadata"]["sam_input_ref"])
                )
                mask_path = (
                    observation_path.parent / str(observation["mask_ref"])
                )
                if not image_path.is_file() or not mask_path.is_file():
                    raise FileNotFoundError(
                        f"missing crop source for {observation['obs_id']}"
                    )
                with Image.open(image_path) as image:
                    pixels = np.asarray(image.convert("RGB"))
                mask = np.load(mask_path, allow_pickle=False).astype(bool)
                context, masked, bounds = build_crop_variants(
                    pixels,
                    mask,
                    padding_fraction=float(
                        config["context_padding_fraction"]
                    ),
                    min_padding_pixels=int(config["min_padding_pixels"]),
                    background_value=int(
                        config["masked_background_value"]
                    ),
                )
                crops.extend([
                    Image.fromarray(context),
                    Image.fromarray(masked),
                ])
                crop_metadata.append(bounds)
                source_metadata.append((
                    _relative(root, image_path),
                    _relative(root, mask_path),
                    _sha256(image_path),
                    _sha256(mask_path),
                ))
            scores: list[tuple[float, float, float]] = []
            if crops:
                features = _encode_pil(
                    backend,
                    crops,
                    batch_size=batch_size,
                )
                text_embedding = backend.encode_text(sam_query)[0]
                for index in range(len(observations)):
                    context_score = float(
                        np.dot(features[2 * index], text_embedding)
                    )
                    masked_score = float(
                        np.dot(features[2 * index + 1], text_embedding)
                    )
                    scores.append((
                        context_score,
                        masked_score,
                        mean_crop_query_score(
                            features[2 * index],
                            features[2 * index + 1],
                            text_embedding,
                        ),
                    ))
            objects: list[dict[str, Any]] = []
            if memory_path.is_file():
                memory = json.loads(memory_path.read_text(encoding="utf-8"))
                objects = sorted(
                    memory.get("objects", []),
                    key=lambda item: (
                        -float(item["confidence"]),
                        str(item["object_id"]),
                    ),
                )
            selected_object = objects[0] if objects else None
            selected_ids = {
                str(item["obs_id"])
                for item in (
                    selected_object["observations"]
                    if selected_object is not None else []
                )
            }
            observation_rows: list[dict[str, Any]] = []
            for index, observation in enumerate(observations):
                context_score, masked_score, semantic_score = (
                    round(float(value), 12) for value in scores[index]
                )
                (
                    image_ref,
                    mask_ref,
                    image_sha256,
                    mask_sha256,
                ) = source_metadata[index]
                observation_rows.append({
                    "observation_id": str(observation["obs_id"]),
                    "frame_id": str(observation["frame_id"]),
                    "center_vggt": observation["center"],
                    "retrieval_score": float(
                        observation["retrieval_score"]
                    ),
                    "sam_score": float(observation["sam_score"]),
                    "valid_point_ratio": float(
                        observation["valid_point_ratio"]
                    ),
                    "quality": observation_quality(observation),
                    "context_crop_cosine": context_score,
                    "masked_crop_cosine": masked_score,
                    "semantic_score": semantic_score,
                    "crop_bounds_xyxy": list(crop_metadata[index]),
                    "image_ref": image_ref,
                    "mask_ref": mask_ref,
                    "image_sha256": image_sha256,
                    "mask_sha256": mask_sha256,
                    "in_selected_object": (
                        str(observation["obs_id"]) in selected_ids
                    ),
                })
            selected_rows = [
                item for item in observation_rows
                if item["in_selected_object"]
            ]
            representative = (
                select_semantic_representative(selected_rows)
                if selected_rows else None
            )
            task_rows.append({
                "task": task,
                "sam_query": sam_query,
                "status": (
                    "PE_SEMANTIC_REPRESENTATIVE"
                    if representative is not None
                    else "Q0_FALLBACK_REQUIRED"
                ),
                "selected_object_id": (
                    str(selected_object["object_id"])
                    if selected_object is not None else None
                ),
                "selected_object_confidence": (
                    float(selected_object["confidence"])
                    if selected_object is not None else None
                ),
                "selected_observation_id": (
                    representative["observation_id"]
                    if representative is not None else None
                ),
                "selected_center_vggt": (
                    representative["center_vggt"]
                    if representative is not None else None
                ),
                "observations": observation_rows,
            })
            source_rows.append({
                "task": task,
                "d6_observations": _relative(root, observation_path),
                "d6_observations_sha256": _sha256(observation_path),
                "a2_memory": (
                    _relative(root, memory_path)
                    if memory_path.is_file() else None
                ),
                "a2_memory_sha256": (
                    _sha256(memory_path)
                    if memory_path.is_file() else None
                ),
            })
    finally:
        backend.close()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "stage": PREDICTION_STAGE,
        "experiment_id": config["experiment_id"],
        "scene_id": scene_id,
        "split_role": role,
        "contract": {
            "ground_truth_used": False,
            "world_alignment_used": False,
            "learned_parameters": False,
            "object_selection": config["object_selection"],
            "center_rule": config["center_rule"],
            "fallback": config["fallback"],
        },
        "sources": {
            "query_manifest": _relative(root, query_manifest_path),
            "query_manifest_sha256": _sha256(query_manifest_path),
            "config": _relative(root, config_path),
            "config_sha256": _sha256(config_path),
            "run_root": _relative(root, run_root),
            "pe_source_commit": actual_source_commit,
            "pe_checkpoint_sha256": checkpoint_sha256,
            "artifacts": source_rows,
        },
        "counts": {
            "tasks": len(task_rows),
            "observations": sum(
                len(item["observations"]) for item in task_rows
            ),
            "selected_object_observations": sum(
                sum(
                    bool(observation["in_selected_object"])
                    for observation in item["observations"]
                )
                for item in task_rows
            ),
            "tasks_with_d6_observations": sum(
                bool(item["observations"]) for item in task_rows
            ),
            "tasks_with_a2_object": sum(
                item["selected_object_id"] is not None
                for item in task_rows
            ),
        },
        "tasks": task_rows,
    }
    return _rounded(payload)


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument(
        "--scene-id",
        required=True,
        choices=("apartment", "cubicle"),
    )
    parser.add_argument("--query-manifest", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--config",
        default="configs/clio_pe_semantic_fusion.json",
    )
    parser.add_argument(
        "--pe-source-root",
        default="third_party/VGGT-SLAM/third_party/perception_models",
    )
    parser.add_argument("--pe-checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.project_root).resolve()

    def resolve(value: str) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    payload = build_prediction(
        project_root=root,
        scene_id=args.scene_id,
        query_manifest_path=resolve(args.query_manifest),
        run_root=resolve(args.run_root),
        config_path=resolve(args.config),
        pe_source_root=resolve(args.pe_source_root),
        pe_checkpoint=resolve(args.pe_checkpoint),
        device=args.device,
        batch_size=args.batch_size,
    )
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"],
        "scene_id": payload["scene_id"],
        **payload["counts"],
        "output": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
