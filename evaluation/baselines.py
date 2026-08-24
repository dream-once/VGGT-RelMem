"""Frozen names and capability boundaries for the B0-B5 experiment ladder."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    capabilities: tuple[str, ...]
    question: str


BASELINES = (
    BaselineSpec(
        "B0-official",
        ("top1", "sam3", "direct_lifting", "upstream_pca_obb"),
        "What does the strict upstream single-frame path produce?",
    ),
    BaselineSpec(
        "B1-robust-single-view",
        ("top1", "sam3", "confidence_filter", "mad_filter"),
        "What changes when only robust 3D lifting is added?",
    ),
    BaselineSpec(
        "B2-topk-multiframe",
        ("topk", "sam3", "robust_lifting"),
        "What does Top-K add before cross-frame association?",
    ),
    BaselineSpec(
        "B3-multiview-fusion",
        ("topk", "association", "memory", "consistency", "abstention"),
        "Does real multi-view fusion improve identity and stability?",
    ),
    BaselineSpec(
        "B4-relational",
        ("memory", "relations"),
        "Do relations disambiguate same-class instances?",
    ),
    BaselineSpec(
        "B5-calibrated",
        ("memory", "relations", "calibration"),
        "Does calibrated abstention reduce answered risk?",
    ),
)
