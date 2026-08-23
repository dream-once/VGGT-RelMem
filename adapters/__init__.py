"""File-based boundaries to the geometry and open-vocabulary environments."""

from .geometry import (
    GeometryBundle,
    GeometryFrame,
    load_anchor_poses,
    load_geometry_npz,
    save_geometry_npz,
)
from .masks import MaskRecord, load_mask, load_mask_manifest, save_mask_manifest
from .vggt_slam import VGGTExportSummary, export_solver_geometry, validate_upstream_layout

__all__ = [
    "GeometryBundle",
    "GeometryFrame",
    "MaskRecord",
    "VGGTExportSummary",
    "export_solver_geometry",
    "load_anchor_poses",
    "load_geometry_npz",
    "load_mask",
    "load_mask_manifest",
    "save_geometry_npz",
    "save_mask_manifest",
    "validate_upstream_layout",
]
