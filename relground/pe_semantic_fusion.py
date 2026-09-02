"""Pure utilities for the post-D21 PE mask-crop fusion experiment."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "0.1"
PREDICTION_STAGE = "clio-pe-mask-crop-fusion-prediction"
EVALUATION_STAGE = "clio-pe-mask-crop-fusion-evaluation"


def expanded_mask_bounds(
    mask: np.ndarray,
    *,
    padding_fraction: float = 0.15,
    min_padding_pixels: int = 2,
) -> tuple[int, int, int, int]:
    """Return exclusive XY bounds around a mask with deterministic context."""

    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2 or not np.any(array):
        raise ValueError("mask must be a non-empty 2D foreground mask")
    if not 0.0 <= padding_fraction <= 1.0:
        raise ValueError("padding_fraction must be in [0, 1]")
    if min_padding_pixels < 0:
        raise ValueError("min_padding_pixels must be non-negative")
    rows, columns = np.nonzero(array)
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    padding = max(
        int(min_padding_pixels),
        round(padding_fraction * max(x1 - x0, y1 - y0)),
    )
    height, width = array.shape
    return (
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(width, x1 + padding),
        min(height, y1 + padding),
    )


def build_crop_variants(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    padding_fraction: float = 0.15,
    min_padding_pixels: int = 2,
    background_value: int = 127,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Create an RGB context crop and a neutral-background masked crop."""

    pixels = np.asarray(image)
    foreground = np.asarray(mask, dtype=bool)
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")
    if pixels.shape[:2] != foreground.shape:
        raise ValueError("image and mask shapes do not match")
    if not 0 <= int(background_value) <= 255:
        raise ValueError("background_value must be in [0, 255]")
    bounds = expanded_mask_bounds(
        foreground,
        padding_fraction=padding_fraction,
        min_padding_pixels=min_padding_pixels,
    )
    x0, y0, x1, y1 = bounds
    context = pixels[y0:y1, x0:x1].astype(np.uint8, copy=True)
    crop_mask = foreground[y0:y1, x0:x1]
    masked = np.where(
        crop_mask[..., None],
        context,
        np.uint8(background_value),
    ).astype(np.uint8, copy=False)
    return context, masked, bounds


def cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("embedding shapes must match and be non-empty")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("embeddings must be finite")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator < 1e-12:
        raise ValueError("embeddings must be non-zero")
    return float(np.dot(left, right) / denominator)


def mean_crop_query_score(
    context_embedding: Sequence[float],
    masked_embedding: Sequence[float],
    text_embedding: Sequence[float],
) -> float:
    return 0.5 * (
        cosine_similarity(context_embedding, text_embedding)
        + cosine_similarity(masked_embedding, text_embedding)
    )


def observation_quality(record: Mapping[str, Any]) -> float:
    values = np.asarray(
        [
            max(float(record["retrieval_score"]), 0.0),
            max(float(record["sam_score"]), 0.0),
            max(float(record["valid_point_ratio"]), 0.0),
        ],
        dtype=np.float64,
    )
    return float(np.prod(values) ** (1.0 / 3.0))


def select_semantic_representative(
    records: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if not records:
        raise ValueError("at least one observation is required")
    return sorted(
        records,
        key=lambda item: (
            -float(item["semantic_score"]),
            str(item["observation_id"]),
        ),
    )[0]


def select_quality_representative(
    records: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if not records:
        raise ValueError("at least one observation is required")
    return sorted(
        records,
        key=lambda item: (
            -float(item["quality"]),
            str(item["observation_id"]),
        ),
    )[0]


def select_medoid(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not records:
        raise ValueError("at least one observation is required")
    ordered = sorted(records, key=lambda item: str(item["observation_id"]))
    centers = np.asarray([item["center_vggt"] for item in ordered], dtype=np.float64)
    if centers.shape != (len(ordered), 3) or not np.all(np.isfinite(centers)):
        raise ValueError("observation centers must be finite 3-vectors")
    distances = np.linalg.norm(
        centers[:, None, :] - centers[None, :, :],
        axis=2,
    )
    return ordered[int(np.argmin(distances.sum(axis=1)))]


def aggregate_task_results(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, Any]:
    total = len(rows)
    answered = sum(bool(row[key]["answered"]) for row in rows)
    strict = sum(bool(row[key]["correct"]) for row in rows)
    padded = sum(
        bool(row[key]["correct_with_alignment_rmse_margin"])
        for row in rows
    )
    return {
        "task_count": total,
        "answered_tasks": answered,
        "coverage": answered / total if total else 0.0,
        "grounding_acc_at_1": strict / total if total else 0.0,
        "grounding_acc_at_1_with_alignment_rmse_margin": (
            padded / total if total else 0.0
        ),
        "conditional_acc_at_1": strict / answered if answered else 0.0,
        "conditional_acc_at_1_with_alignment_rmse_margin": (
            padded / answered if answered else 0.0
        ),
    }


def paired_transitions(
    rows: Sequence[Mapping[str, Any]],
    baseline_key: str,
    variant_key: str,
) -> dict[str, int]:
    return {
        "both_wrong": sum(
            not row[baseline_key]["correct"] and not row[variant_key]["correct"]
            for row in rows
        ),
        "regressions": sum(
            row[baseline_key]["correct"] and not row[variant_key]["correct"]
            for row in rows
        ),
        "wins": sum(
            not row[baseline_key]["correct"] and row[variant_key]["correct"]
            for row in rows
        ),
        "both_correct": sum(
            row[baseline_key]["correct"] and row[variant_key]["correct"]
            for row in rows
        ),
    }
