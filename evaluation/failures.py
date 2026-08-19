"""Stable failure taxonomy used by reports and failure galleries."""

from enum import Enum


class FailureType(str, Enum):
    RETRIEVAL = "retrieval"
    SEGMENTATION = "segmentation"
    GEOMETRY = "geometry"
    ASSOCIATION = "association"
    RELATION = "relation"
    CALIBRATION = "calibration"
