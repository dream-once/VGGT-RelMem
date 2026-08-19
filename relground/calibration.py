"""Small-sample confidence calibration and abstention utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import json

import numpy as np

from .schemas import GroundingResult, MemoryObject


FEATURE_NAMES = (
    "retrieval_score",
    "sam_score",
    "valid_point_ratio",
    "support_count",
    "association_margin",
    "relation_margin",
)


def confidence_features(
    memory_object: MemoryObject,
    *,
    association_margin: float,
    relation_margin: float,
) -> np.ndarray:
    observations = memory_object.observations
    weights = np.ones(len(observations)) / max(len(observations), 1)
    return np.array(
        [
            float(np.average([item.retrieval_score for item in observations], weights=weights)),
            float(np.average([item.sam_score for item in observations], weights=weights)),
            float(np.average([item.valid_point_ratio for item in observations], weights=weights)),
            float(np.log1p(len(observations))),
            float(association_margin),
            float(relation_margin),
        ],
        dtype=np.float64,
    )


class LogisticCalibrator:
    """L2-regularized logistic regression implemented with NumPy.

    This intentionally avoids a hard scikit-learn dependency in the lightweight
    project-logic environment. It is for dev-split calibration, not model training.
    """

    def __init__(self, l2: float = 1e-3, learning_rate: float = 0.05, max_iter: int = 2000) -> None:
        self.l2 = float(l2)
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, features: np.ndarray, labels: Sequence[int]) -> "LogisticCalibrator":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        if x.ndim != 2 or y.shape != (len(x),) or len(x) < 2:
            raise ValueError("features must be 2D and labels must match at least two rows")
        if not np.all(np.isfinite(x)) or not np.all(np.isin(y, [0.0, 1.0])):
            raise ValueError("features must be finite and labels binary")
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        standardized = (x - self.mean_) / self.scale_
        prevalence = float(np.clip(y.mean(), 1e-4, 1.0 - 1e-4))
        weights = np.zeros(x.shape[1], dtype=np.float64)
        intercept = float(np.log(prevalence / (1.0 - prevalence)))
        for _ in range(self.max_iter):
            logits = np.clip(standardized @ weights + intercept, -30.0, 30.0)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            residual = probabilities - y
            weight_gradient = standardized.T @ residual / len(x) + self.l2 * weights
            intercept_gradient = float(residual.mean())
            weights -= self.learning_rate * weight_gradient
            intercept -= self.learning_rate * intercept_gradient
        self.coef_, self.intercept_ = weights, intercept
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if any(item is None for item in (self.mean_, self.scale_, self.coef_, self.intercept_)):
            raise RuntimeError("calibrator has not been fitted")
        x = np.asarray(features, dtype=np.float64)
        one_row = x.ndim == 1
        x = np.atleast_2d(x)
        if x.shape[1] != len(self.coef_):
            raise ValueError("feature dimension does not match fitted calibrator")
        logits = np.clip((x - self.mean_) / self.scale_ @ self.coef_ + self.intercept_, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        return probabilities[0] if one_row else probabilities

    def to_dict(self) -> dict[str, Any]:
        if any(item is None for item in (self.mean_, self.scale_, self.coef_, self.intercept_)):
            raise RuntimeError("calibrator has not been fitted")
        return {
            "feature_names": list(FEATURE_NAMES),
            "l2": self.l2,
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "coef": self.coef_.tolist(),
            "intercept": self.intercept_,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "LogisticCalibrator":
        data = json.loads(Path(path).read_text())
        calibrator = cls(l2=float(data.get("l2", 1e-3)))
        calibrator.mean_ = np.asarray(data["mean"], dtype=np.float64)
        calibrator.scale_ = np.asarray(data["scale"], dtype=np.float64)
        calibrator.coef_ = np.asarray(data["coef"], dtype=np.float64)
        calibrator.intercept_ = float(data["intercept"])
        return calibrator


@dataclass(frozen=True)
class AbstentionPolicy:
    answer_threshold: float = 0.60

    def __post_init__(self) -> None:
        if not 0.0 <= self.answer_threshold <= 1.0:
            raise ValueError("answer_threshold must be in [0, 1]")

    def apply(self, result: GroundingResult, calibrated_confidence: float) -> GroundingResult:
        confidence = float(np.clip(calibrated_confidence, 0.0, 1.0))
        result.confidence = confidence
        if not result.abstain and confidence < self.answer_threshold:
            result.abstain = True
            result.reason = "calibrated_low_confidence"
        return result
