"""JSON + NPY contract exported by the isolated ``open_vocab`` environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import json

import numpy as np


MASK_SCHEMA_VERSION = "0.1"


@dataclass(frozen=True)
class MaskRecord:
    obs_id: str
    frame_id: str
    class_text: str
    mask_ref: str
    retrieval_score: float
    sam_score: float
    semantic_embedding: list[float] | None = None

    def __post_init__(self) -> None:
        if not self.obs_id or not self.frame_id or not self.class_text or not self.mask_ref:
            raise ValueError("mask record identifiers, class_text and mask_ref are required")
        if not 0.0 <= self.retrieval_score <= 1.0 or not 0.0 <= self.sam_score <= 1.0:
            raise ValueError("retrieval_score and sam_score must be in [0, 1]")


def save_mask_manifest(path: str | Path, records: Iterable[MaskRecord]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": MASK_SCHEMA_VERSION,
        "records": [asdict(record) for record in records],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def load_mask_manifest(path: str | Path) -> list[MaskRecord]:
    payload: dict[str, Any] = json.loads(Path(path).read_text())
    return [MaskRecord(**record) for record in payload.get("records", [])]


def load_mask(record: MaskRecord, manifest_path: str | Path) -> np.ndarray:
    mask_path = Path(record.mask_ref)
    if not mask_path.is_absolute():
        mask_path = Path(manifest_path).parent / mask_path
    if mask_path.suffix == ".npy":
        return np.asarray(np.load(mask_path, allow_pickle=False), dtype=bool)
    with np.load(mask_path, allow_pickle=False) as archive:
        if "mask" not in archive:
            raise ValueError(f"mask archive has no 'mask' array: {mask_path}")
        return np.asarray(archive["mask"], dtype=bool)
