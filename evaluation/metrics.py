"""Dependency-light metrics for retrieval, association and selective grounding."""

from __future__ import annotations

from typing import Hashable, Mapping, Sequence

import numpy as np


def frame_recall_at_k(
    relevant_frames: Mapping[str, set[str]],
    ranked_frames: Mapping[str, Sequence[str]],
    k: int,
) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    if not relevant_frames:
        return 0.0
    hits = [
        bool(relevant & set(ranked_frames.get(query_id, ())[:k]))
        for query_id, relevant in relevant_frames.items()
    ]
    return float(np.mean(hits))


def duplicate_rate(gt_to_predicted_ids: Mapping[Hashable, Sequence[Hashable]]) -> float:
    total_predictions = sum(len(set(items)) for items in gt_to_predicted_ids.values())
    duplicates = sum(max(0, len(set(items)) - 1) for items in gt_to_predicted_ids.values())
    return 0.0 if total_predictions == 0 else duplicates / total_predictions


def count_error(predicted_count: int, ground_truth_count: int) -> int:
    return abs(int(predicted_count) - int(ground_truth_count))


def pairwise_f1(
    ground_truth_labels: Sequence[Hashable],
    predicted_labels: Sequence[Hashable],
) -> dict[str, float]:
    if len(ground_truth_labels) != len(predicted_labels):
        raise ValueError("label sequences must have the same length")
    true_positive = false_positive = false_negative = 0
    for first in range(len(ground_truth_labels)):
        for second in range(first + 1, len(ground_truth_labels)):
            same_truth = ground_truth_labels[first] == ground_truth_labels[second]
            same_prediction = predicted_labels[first] == predicted_labels[second]
            true_positive += int(same_truth and same_prediction)
            false_positive += int(not same_truth and same_prediction)
            false_negative += int(same_truth and not same_prediction)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def grounding_metrics(
    answers: Sequence[str | None],
    rankings: Sequence[Sequence[str]],
    abstentions: Sequence[bool] | None = None,
) -> dict[str, float]:
    if len(answers) != len(rankings):
        raise ValueError("answers and rankings must have the same length")
    if abstentions is not None and len(abstentions) != len(answers):
        raise ValueError("abstentions must have the same length as answers")
    if not answers:
        return {
            "acc_at_1": 0.0,
            "mrr": 0.0,
            "negative_rejection_accuracy": 0.0,
            "task_accuracy": 0.0,
        }
    reciprocal_ranks: list[float] = []
    top_hits: list[float] = []
    negative_hits: list[float] = []
    task_hits: list[float] = []
    abstention_values = list(abstentions) if abstentions is not None else [not item for item in rankings]
    for answer, ranking, abstained in zip(answers, rankings, abstention_values):
        if answer is None:
            negative_hits.append(float(abstained))
            task_hits.append(float(abstained))
            continue
        is_top_hit = bool(ranking) and ranking[0] == answer
        top_hits.append(float(is_top_hit))
        task_hits.append(float(is_top_hit and not abstained))
        if answer not in ranking:
            reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(1.0 / (ranking.index(answer) + 1))
    return {
        "acc_at_1": float(np.mean(top_hits)) if top_hits else 0.0,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "negative_rejection_accuracy": float(np.mean(negative_hits)) if negative_hits else 0.0,
        "task_accuracy": float(np.mean(task_hits)),
    }


def brier_score(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    predictions = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    if predictions.shape != targets.shape or predictions.size == 0:
        raise ValueError("non-empty probabilities and labels must have the same shape")
    return float(np.mean((predictions - targets) ** 2))


def risk_coverage_curve(
    confidences: Sequence[float],
    correct: Sequence[bool],
) -> list[dict[str, float]]:
    scores = np.asarray(confidences, dtype=np.float64)
    outcomes = np.asarray(correct, dtype=bool)
    if scores.shape != outcomes.shape or scores.size == 0:
        raise ValueError("non-empty confidences and correctness must have the same shape")
    order = np.argsort(-scores, kind="stable")
    sorted_outcomes = outcomes[order]
    cumulative_accuracy = np.cumsum(sorted_outcomes) / np.arange(1, len(scores) + 1)
    return [
        {
            "coverage": float((index + 1) / len(scores)),
            "risk": float(1.0 - cumulative_accuracy[index]),
            "threshold": float(scores[order[index]]),
        }
        for index in range(len(scores))
    ]
