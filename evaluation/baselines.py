"""Frozen names and capability boundaries for the B0-B5 experiment ladder."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    capabilities: tuple[str, ...]
    question: str


BASELINES = (
    BaselineSpec("B0", ("top1", "sam3"), "What is the upstream single-frame baseline?"),
    BaselineSpec("B1", ("topk",), "How much recall and duplication does raw top-K add?"),
    BaselineSpec("B2", ("topk", "association"), "Does 3D association improve identity consistency?"),
    BaselineSpec("B3", ("topk", "association", "memory"), "Does evidence fusion improve stability?"),
    BaselineSpec("B4", ("memory", "relations"), "Do relations disambiguate same-class instances?"),
    BaselineSpec("B5", ("memory", "relations", "calibration"), "Does abstention reduce answered risk?"),
)
