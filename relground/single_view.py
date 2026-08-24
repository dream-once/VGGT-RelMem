"""Controlled single-view baseline contracts and geometry lifting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .observations import LiftedGeometry, LiftingError
from .schemas import ObjectObservation, OrientedBoundingBox


BASELINE_SCHEMA_VERSION = "1.0"
PREPROCESS_SCHEMA_VERSION = "1.0"
B0_OFFICIAL = "B0-official"
B1_ROBUST_SINGLE_VIEW = "B1-robust-single-view"
SUPPORTED_SINGLE_VIEW_BASELINES = (
    B0_OFFICIAL,
    B1_ROBUST_SINGLE_VIEW,
)
PREPROCESS_FIELDS = (
    "schema_version",
    "mode",
    "target_size",
    "patch_size",
    "source_size",
    "resized_size",
    "crop_xyxy",
    "padding_ltrb",
    "output_size",
    "output_shape",
)


@dataclass(frozen=True)
class VGGTImageTransform:
    """Exact resize/crop/batch-pad mapping used by upstream VGGT crop mode."""

    source_size: tuple[int, int]
    resized_size: tuple[int, int]
    crop_xyxy: tuple[int, int, int, int]
    padding_ltrb: tuple[int, int, int, int]
    output_size: tuple[int, int]
    target_size: int = 518
    patch_size: int = 14
    mode: str = "crop"
    schema_version: str = PREPROCESS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PREPROCESS_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported preprocess schema: {self.schema_version}"
            )
        if self.mode != "crop":
            raise ValueError("only upstream VGGT crop mode is supported")
        for name, pair in (
            ("source_size", self.source_size),
            ("resized_size", self.resized_size),
            ("output_size", self.output_size),
        ):
            if len(pair) != 2 or any(int(value) < 1 for value in pair):
                raise ValueError(f"{name} must contain two positive integers")
        if len(self.crop_xyxy) != 4 or len(self.padding_ltrb) != 4:
            raise ValueError("crop and padding must contain four integers")
        if any(int(value) < 0 for value in self.padding_ltrb):
            raise ValueError("padding must be non-negative")

    @property
    def output_shape(self) -> tuple[int, int]:
        return (self.output_size[1], self.output_size[0])

    def apply(self, image: Any) -> Any:
        """Apply the recorded mapping to a PIL image."""

        from PIL import Image, ImageOps

        if tuple(image.size) != self.source_size:
            raise ValueError(
                f"source image size changed: {image.size} != {self.source_size}"
            )
        if image.mode == "RGBA":
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
        resized = image.convert("RGB").resize(
            self.resized_size,
            resample=Image.Resampling.BICUBIC,
        )
        transformed = resized.crop(self.crop_xyxy)
        if any(self.padding_ltrb):
            transformed = ImageOps.expand(
                transformed,
                border=self.padding_ltrb,
                fill=(255, 255, 255),
            )
        if tuple(transformed.size) != self.output_size:
            raise ValueError(
                f"preprocessed image shape mismatch: "
                f"{transformed.size} != {self.output_size}"
            )
        return transformed

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "target_size": self.target_size,
            "patch_size": self.patch_size,
            "source_size": list(self.source_size),
            "resized_size": list(self.resized_size),
            "crop_xyxy": list(self.crop_xyxy),
            "padding_ltrb": list(self.padding_ltrb),
            "output_size": list(self.output_size),
            "output_shape": list(self.output_shape),
        }
        if tuple(payload) != PREPROCESS_FIELDS:
            raise AssertionError("VGGT preprocess field order changed")
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VGGTImageTransform":
        actual = set(payload)
        expected = set(PREPROCESS_FIELDS)
        if actual != expected:
            raise ValueError(
                "VGGT preprocess fields differ from frozen schema: "
                f"missing={sorted(expected - actual)} "
                f"unexpected={sorted(actual - expected)}"
            )
        return cls(
            source_size=tuple(int(value) for value in payload["source_size"]),
            resized_size=tuple(int(value) for value in payload["resized_size"]),
            crop_xyxy=tuple(int(value) for value in payload["crop_xyxy"]),
            padding_ltrb=tuple(
                int(value) for value in payload["padding_ltrb"]
            ),
            output_size=tuple(int(value) for value in payload["output_size"]),
            target_size=int(payload.get("target_size", 518)),
            patch_size=int(payload.get("patch_size", 14)),
            mode=str(payload.get("mode", "crop")),
            schema_version=str(
                payload.get("schema_version", PREPROCESS_SCHEMA_VERSION)
            ),
        )


def compute_vggt_crop_transform(
    source_size: tuple[int, int],
    output_shape: tuple[int, int],
    *,
    target_size: int = 518,
    patch_size: int = 14,
) -> VGGTImageTransform:
    """Reconstruct upstream load_and_preprocess_images(..., mode='crop').

    The final symmetric padding models upstream batch padding when images in
    one VGGT submap have different preprocessed heights.
    """

    source_width, source_height = (int(source_size[0]), int(source_size[1]))
    output_height, output_width = (
        int(output_shape[0]),
        int(output_shape[1]),
    )
    if min(
        source_width,
        source_height,
        output_width,
        output_height,
        target_size,
        patch_size,
    ) < 1:
        raise ValueError("image and preprocessing dimensions must be positive")

    resized_width = target_size
    resized_height = (
        round(
            source_height
            * (resized_width / source_width)
            / patch_size
        )
        * patch_size
    )
    if resized_height < 1:
        raise ValueError("upstream preprocessing rounded image height to zero")

    crop_top = max(0, (resized_height - target_size) // 2)
    cropped_height = min(resized_height, target_size)
    crop = (
        0,
        crop_top,
        resized_width,
        crop_top + cropped_height,
    )
    if output_width < resized_width or output_height < cropped_height:
        raise ValueError(
            "geometry grid is smaller than the upstream preprocessed image"
        )
    width_padding = output_width - resized_width
    height_padding = output_height - cropped_height
    padding = (
        width_padding // 2,
        height_padding // 2,
        width_padding - width_padding // 2,
        height_padding - height_padding // 2,
    )
    crop_left, crop_top, crop_right, crop_bottom = crop
    if (
        crop_left < 0
        or crop_top < 0
        or crop_right > resized_width
        or crop_bottom > resized_height
        or crop_right <= crop_left
        or crop_bottom <= crop_top
    ):
        raise ValueError("computed VGGT crop is outside the resized image")
    return VGGTImageTransform(
        source_size=(source_width, source_height),
        resized_size=(resized_width, resized_height),
        crop_xyxy=crop,
        padding_ltrb=padding,
        output_size=(output_width, output_height),
        target_size=target_size,
        patch_size=patch_size,
    )


def load_vggt_sam_image(
    path: str | Path,
    output_shape: tuple[int, int],
) -> tuple[Any, VGGTImageTransform]:
    """Load an original frame and reproduce the exact VGGT image grid."""

    from PIL import Image

    with Image.open(path) as source:
        transform = compute_vggt_crop_transform(source.size, output_shape)
        return transform.apply(source), transform


def _project_points(
    points: np.ndarray,
    global_from_submap: np.ndarray | None,
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != (3,):
        raise LiftingError("points must have shape (N,3)")
    if global_from_submap is None:
        return values
    transform = np.asarray(global_from_submap, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise LiftingError("point transform must be a finite 4x4 matrix")
    homogeneous = np.column_stack([values, np.ones(len(values))])
    transformed = homogeneous @ transform.T
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return transformed[:, :3] / transformed[:, 3, None]


def official_pca_lift(
    mask: np.ndarray,
    point_map: np.ndarray,
    global_from_submap: np.ndarray | None = None,
) -> LiftedGeometry:
    """Upstream-compatible direct mask indexing plus finite-only PCA OBB."""

    mask_array = np.asarray(mask, dtype=bool)
    points_array = np.asarray(point_map, dtype=np.float64)
    if points_array.shape != mask_array.shape + (3,):
        raise LiftingError("point_map must have shape mask.shape + (3,)")
    masked_count = int(mask_array.sum())
    if masked_count == 0:
        raise LiftingError("empty mask")

    points = _project_points(
        points_array.reshape(-1, 3)[mask_array.reshape(-1)],
        global_from_submap,
    )
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 2:
        raise LiftingError(
            "official PCA requires at least two finite masked points"
        )

    centroid = points.mean(axis=0)
    centered = points - centroid
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    rotation = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    local = centered @ rotation
    lower = local.min(axis=0)
    upper = local.max(axis=0)
    center_local = 0.5 * (lower + upper)
    center = centroid + center_local @ rotation.T
    obb = OrientedBoundingBox(
        center=center,
        extent=upper - lower,
        rotation=rotation,
    )
    return LiftedGeometry(
        points=points,
        center=center.copy(),
        obb=obb,
        valid_point_ratio=float(len(points) / masked_count),
    )


def make_official_observation(
    *,
    obs_id: str,
    class_text: str,
    frame_id: str,
    mask: np.ndarray,
    point_map: np.ndarray,
    global_from_submap: np.ndarray | None,
    retrieval_score: float,
    sam_score: float,
    mask_ref: str,
    points_ref: str,
) -> tuple[ObjectObservation, np.ndarray]:
    lifted = official_pca_lift(
        mask,
        point_map,
        global_from_submap,
    )
    observation = ObjectObservation(
        obs_id=obs_id,
        class_text=class_text,
        frame_id=frame_id,
        mask_ref=mask_ref,
        retrieval_score=retrieval_score,
        sam_score=sam_score,
        valid_point_ratio=lifted.valid_point_ratio,
        points_ref=points_ref,
        center=lifted.center,
        obb=lifted.obb,
        metadata={
            "baseline_id": B0_OFFICIAL,
            "lifting": {
                "mask_indexing": "direct",
                "point_filter": "finite-only",
                "obb": "upstream-pca",
            },
        },
    )
    return observation, lifted.points
