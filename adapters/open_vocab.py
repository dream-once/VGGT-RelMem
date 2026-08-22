"""Isolated Perception Encoder + SAM 3 adapter for the D4 top-1 baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import importlib
import json
import sys

import numpy as np


PE_SOURCE_COMMIT = "3e352cca660658d4b5c90f42a7808b11469e4c66"
SAM3_SOURCE_COMMIT = "8f0b7f4d4e7eda2ed606ebde6702c93359ad01da"


@dataclass(frozen=True)
class FrameSource:
    frame_id: str
    image_path: Path
    geometry_index: int
    submap_id: int
    submap_frame_index: int


@dataclass(frozen=True)
class Top1Match:
    frame_id: str
    index: int
    score: float
    cosine: float


@dataclass(frozen=True)
class SegmentationBatch:
    masks: np.ndarray
    boxes_xyxy: np.ndarray
    scores: np.ndarray


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    # NumPy cannot represent torch.bfloat16 directly. SAM 3 runs under BF16
    # autocast, so normalize any floating outputs before moving them to NumPy.
    if "bfloat16" in str(getattr(value, "dtype", "")) and hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _normalized_rows(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite non-empty matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError(f"{name} contains a zero vector")
    return array / norms


def load_frame_sources(
    manifest_path: str | Path,
    geometry_frame_ids: Sequence[str],
    *,
    project_root: str | Path,
) -> list[FrameSource]:
    """Resolve original images in exactly the exported geometry-frame order."""

    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    records = payload.get("frames", [])
    by_id = {str(record["frame_id"]): record for record in records}
    if len(by_id) != len(records):
        raise ValueError("geometry manifest contains duplicate frame_id values")
    root = Path(project_root).resolve()
    sources: list[FrameSource] = []
    for index, frame_id in enumerate(geometry_frame_ids):
        if frame_id not in by_id:
            raise ValueError(f"geometry manifest has no source for frame {frame_id}")
        record = by_id[frame_id]
        image_path = Path(record["image_path"]).expanduser()
        if not image_path.is_absolute():
            image_path = root / image_path
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"source image is missing: {image_path}")
        sources.append(
            FrameSource(
                frame_id=str(frame_id),
                image_path=image_path,
                geometry_index=index,
                submap_id=int(record["submap_id"]),
                submap_frame_index=int(record["submap_frame_index"]),
            )
        )
    return sources


def select_top1(
    frame_ids: Sequence[str],
    image_embeddings: np.ndarray,
    text_embedding: np.ndarray,
) -> Top1Match:
    """Reproduce upstream non-negative cosine top-1 selection deterministically."""

    if not frame_ids or len(set(frame_ids)) != len(frame_ids):
        raise ValueError("frame_ids must be non-empty and unique")
    images = _normalized_rows(image_embeddings, "image_embeddings")
    text = np.asarray(text_embedding, dtype=np.float64)
    if text.ndim == 1:
        text = text[None, :]
    text = _normalized_rows(text, "text_embedding")
    if text.shape[0] != 1 or images.shape != (len(frame_ids), text.shape[1]):
        raise ValueError("embedding shapes do not match frame_ids and one text query")
    cosine = images @ text[0]
    # GraphMap.retrieve_best_semantic_frame initializes its best score at zero.
    scores = np.clip(cosine, 0.0, 1.0)
    index = int(np.argmax(scores))
    return Top1Match(str(frame_ids[index]), index, float(scores[index]), float(cosine[index]))


def prepare_sam_outputs(
    masks: Any,
    boxes_xyxy: Any,
    scores: Any,
    *,
    image_shape: tuple[int, int] | None = None,
) -> SegmentationBatch:
    mask_array = _numpy(masks).astype(bool, copy=False)
    if mask_array.ndim == 2:
        mask_array = mask_array[None, ...]
    elif mask_array.ndim == 4 and mask_array.shape[1] == 1:
        mask_array = mask_array[:, 0]
    if mask_array.ndim != 3:
        raise ValueError("SAM masks must have shape (N,H,W) or (N,1,H,W)")
    if image_shape is not None and tuple(mask_array.shape[1:]) != tuple(image_shape):
        raise ValueError("SAM masks do not match the original image resolution")
    box_array = _numpy(boxes_xyxy).astype(np.float64, copy=False).reshape(-1, 4)
    score_array = _numpy(scores).astype(np.float64, copy=False).reshape(-1)
    count = len(mask_array)
    if len(box_array) != count or len(score_array) != count:
        raise ValueError("SAM masks, boxes and scores have different instance counts")
    if not np.all(np.isfinite(box_array)) or not np.all(np.isfinite(score_array)):
        raise ValueError("SAM boxes and scores must be finite")
    if np.any((score_array < 0.0) | (score_array > 1.0)):
        raise ValueError("SAM scores must be in [0, 1]")
    return SegmentationBatch(mask_array, box_array, score_array)


def resize_mask_nearest(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize a full-resolution SAM mask to the VGGT point-map grid."""

    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("mask must be a non-empty 2D array")
    target_h, target_w = (int(target_shape[0]), int(target_shape[1]))
    if target_h < 1 or target_w < 1:
        raise ValueError("target_shape must be positive")
    if array.shape == (target_h, target_w):
        return array.copy()
    source_h, source_w = array.shape
    rows = np.minimum(
        np.floor((np.arange(target_h) + 0.5) * source_h / target_h).astype(int),
        source_h - 1,
    )
    columns = np.minimum(
        np.floor((np.arange(target_w) + 0.5) * source_w / target_w).astype(int),
        source_w - 1,
    )
    return array[np.ix_(rows, columns)]


def validate_source_checkout(root: str | Path, required: Sequence[str]) -> Path:
    path = Path(root).resolve()
    missing = [value for value in required if not (path / value).exists()]
    if missing:
        raise FileNotFoundError(f"invalid upstream checkout {path}; missing {missing}")
    return path


def _prepend_import_path(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


class PerceptionEncoderBackend:
    """Lazy wrapper around Meta's official PE-Core CLIP implementation."""

    def __init__(
        self,
        source_root: str | Path,
        *,
        config: str = "PE-Core-L14-336",
        checkpoint_path: str | Path | None = None,
        device: str = "cuda",
    ) -> None:
        root = validate_source_checkout(
            source_root,
            ("core/vision_encoder/pe.py", "core/vision_encoder/transforms.py"),
        )
        _prepend_import_path(root)
        self.torch = importlib.import_module("torch")
        pe = importlib.import_module("core.vision_encoder.pe")
        transforms = importlib.import_module("core.vision_encoder.transforms")
        self.device = device
        self.model = pe.CLIP.from_config(
            config,
            pretrained=True,
            checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
        )
        self.model = self.model.eval().to(device)
        self.preprocess = transforms.get_image_transform(self.model.image_size)
        self.tokenizer = transforms.get_text_tokenizer(self.model.context_length)

    def encode_images(self, image_paths: Sequence[Path], batch_size: int = 8) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        image_module = importlib.import_module("PIL.Image")
        outputs: list[np.ndarray] = []
        for offset in range(0, len(image_paths), batch_size):
            tensors = []
            for path in image_paths[offset : offset + batch_size]:
                with image_module.open(path) as image:
                    tensors.append(self.preprocess(image.convert("RGB")))
            batch = self.torch.stack(tensors).to(self.device)
            with self.torch.inference_mode():
                features = self.model.encode_image(batch, normalize=True)
            outputs.append(_numpy(features).astype(np.float32, copy=False))
        if not outputs:
            raise ValueError("at least one image is required")
        return np.concatenate(outputs, axis=0)

    def encode_text(self, query: str) -> np.ndarray:
        if not query.strip():
            raise ValueError("query must not be empty")
        tokens = self.tokenizer([query.strip()]).to(self.device)
        with self.torch.inference_mode():
            features = self.model.encode_text(tokens, normalize=True)
        return _numpy(features).astype(np.float32, copy=False)

    def close(self) -> None:
        self.model = None
        if self.device.startswith("cuda") and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class Sam3Backend:
    """Lazy wrapper around Meta's official SAM 3 image processor."""

    def __init__(
        self,
        source_root: str | Path,
        *,
        checkpoint_path: str | Path | None = None,
        confidence_threshold: float = 0.5,
        device: str = "cuda",
    ) -> None:
        root = validate_source_checkout(
            source_root,
            ("sam3/model_builder.py", "sam3/model/sam3_image_processor.py"),
        )
        _prepend_import_path(root)
        self.torch = importlib.import_module("torch")
        builder = importlib.import_module("sam3.model_builder")
        processor_module = importlib.import_module("sam3.model.sam3_image_processor")
        self.device = device
        local_checkpoint = None if checkpoint_path is None else str(Path(checkpoint_path).resolve())
        model = builder.build_sam3_image_model(
            checkpoint_path=local_checkpoint,
            load_from_HF=local_checkpoint is None,
            device=device,
        )
        self.processor = processor_module.Sam3Processor(
            model,
            device=device,
            confidence_threshold=float(confidence_threshold),
        )

    def segment(self, image: Any, query: str) -> SegmentationBatch:
        prompt = query.strip()
        if not prompt:
            raise ValueError("query must not be empty")
        device_type = self.device.split(":", 1)[0]
        amp_enabled = device_type == "cuda"
        amp_dtype = None
        if amp_enabled:
            amp_dtype = (
                self.torch.bfloat16
                if self.torch.cuda.is_bf16_supported()
                else self.torch.float16
            )
        # Official SAM 3 examples run both image encoding and prompt inference
        # inside one CUDA autocast context. Without it the preprocessed BF16
        # activations meet FP32 linear weights and PyTorch raises a dtype error.
        with self.torch.inference_mode(), self.torch.autocast(
            device_type=device_type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            state = self.processor.set_image(image)
            output = self.processor.set_text_prompt(prompt=prompt, state=state)
        return prepare_sam_outputs(
            output["masks"],
            output["boxes"],
            output["scores"],
            image_shape=(image.height, image.width),
        )
