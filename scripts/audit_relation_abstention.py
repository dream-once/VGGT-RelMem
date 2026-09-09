"""Replay frozen Clio caches for post-hoc abstention diagnostics (CPU only).

No threshold is selected, no model is fitted, and labels only enter evaluation.
All comparisons retain the same queries, memory, geometry, and denominators.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import csv

import numpy as np

from relground.association import ObjectMemory
from relground.calibration import AbstentionPolicy
from relground.clio_relation_benchmark import evaluate_clio_relation_prediction
from relground.relations import RelationConfig, RelationGrounder
from relground.schemas import GroundingQuery

SCENES = {
    "apartment": "clio-apartment-dev-v2-lc",
    "cubicle": "clio-cubicle-heldout-v1",
}
THRESHOLDS = (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)
CONFIDENCE_REASONS = {"low_confidence", "calibrated_low_confidence", "uncalibrated_low_score"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def policy_results(raw, mode, threshold=0.6):
    results = deepcopy(raw)
    for result in results:
        if mode == "proposals_without_gates":
            # Missing target/reference remains unanswerable even in this ablation.
            if result.ranked_ids and result.explanation.get("reference_id"):
                result.abstain, result.reason = False, None
        else:
            if result.reason in CONFIDENCE_REASONS:
                result.abstain, result.reason = False, None
            if mode != "geometry_and_ambiguity_only" and not result.abstain and result.confidence < threshold:
                result.abstain, result.reason = True, "uncalibrated_low_score"
    return [r.to_dict() for r in results]


def summarize(evaluation):
    rows = evaluation["rows"]
    positive = [r for r in rows if r["answerable"]]
    negative = [r for r in rows if not r["answerable"]]
    answers = [r for r in rows if not r["abstain"]]
    strict_correct = sum(r["strict_correct"] for r in answers)
    padded_correct = sum(r["alignment_rmse_padded_correct"] for r in answers)
    result = {
        "queries": len(rows), "positives": len(positive), "negatives": len(negative),
        "answers": len(answers),
        "coverage": ratio(len(answers), len(rows)),
        "positive_answers": sum(not r["abstain"] for r in positive),
        "positive_answer_rate": ratio(sum(not r["abstain"] for r in positive), len(positive)),
        "positive_false_rejections": sum(r["abstain"] for r in positive),
        "positive_strict_correct": strict_correct,
        "positive_strict_acc": ratio(strict_correct, len(positive)),
        "positive_padded_correct": padded_correct,
        "positive_padded_acc": ratio(padded_correct, len(positive)),
        "answered_strict_precision": ratio(strict_correct, len(answers)),
        "answered_padded_precision": ratio(padded_correct, len(answers)),
        "answered_strict_risk": ratio(len(answers) - strict_correct, len(answers)),
        "answered_padded_risk": ratio(len(answers) - padded_correct, len(answers)),
        "negative_false_answers": sum(not r["abstain"] for r in negative),
        "negative_rejection_rate": ratio(sum(r["abstain"] for r in negative), len(negative)),
        "positive_reasons": dict(Counter(r["reason"] or "answered" for r in positive)),
        "negative_reasons": dict(Counter(r["reason"] or "answered" for r in negative)),
    }
    for mode in ("strict", "alignment_rmse_padded"):
        for polarity, subset in (("positive", positive), ("negative", negative)):
            paired = [r for r in subset if r[f"pair_{mode}_gt_match"]]
            result[f"{polarity}_selected_pair_{mode}_count"] = len(paired)
            if polarity == "negative":
                rejected = sum(r["abstain"] and r["reason"] == "relation_conflict_or_boundary" for r in paired)
                result[f"negative_{mode}_grounded_reason_rejections"] = rejected
                result[f"negative_{mode}_grounded_reason_rate_all_negatives"] = ratio(rejected, len(negative))
                result[f"negative_{mode}_reason_rate_conditioned_on_selected_pair"] = ratio(rejected, len(paired))
    return result


def availability(labels):
    result = {}
    for mode in ("strict", "alignment_rmse_padded"):
        tasks = {}
        positives = [r for r in labels["labels"] if r["answerable"]]
        for row in positives:
            for role in ("target", "reference"):
                tasks[row[f"{role}_task"]] = row[f"acceptable_{role}_object_ids_{mode}"]
        possible = sum(bool(r[f"acceptable_target_object_ids_{mode}"]) and bool(r[f"acceptable_reference_object_ids_{mode}"]) for r in positives)
        result[mode] = {
            "eligible_tasks": len(tasks),
            "tasks_with_localized_permanent_object": sum(bool(ids) for ids in tasks.values()),
            "positive_pairs_with_both_localized_objects_available": possible,
            "positive_count": len(positives),
            "candidate_localization_ceiling": ratio(possible, len(positives)),
        }
    return result


def compare_rows(baseline, variant):
    keys = ("strict_correct", "alignment_rmse_padded_correct", "negative_rejection_correct")
    changes = {key: {"wins": [], "losses": []} for key in keys}
    reference_changes = []
    for before, after in zip(baseline["rows"], variant["rows"], strict=True):
        assert before["query_id"] == after["query_id"]
        for key in keys:
            if before[key] != after[key]:
                changes[key]["wins" if after[key] else "losses"].append(before["query_id"])
        if before["predicted_reference_object_id"] != after["predicted_reference_object_id"]:
            reference_changes.append(before["query_id"])
    return {"decisions": changes, "reference_changed_query_ids": reference_changes}


def association_scope(path):
    source = read(path)
    groups = {}
    for scope in ("all_pairs", "cross_frame_only", "same_frame_only"):
        groups[scope] = {}
        for policy in ("A1", "A2"):
            tp = fp = tn = fn = 0
            for task in source["tasks"]:
                frames = {a["observation_id"]: a["frame_id"] for a in task["assignments"]}
                for pair in task["pairs"]:
                    if pair["expected_same"] is None:
                        continue
                    cross = frames[pair["observation_id_a"]] != frames[pair["observation_id_b"]]
                    if (scope == "cross_frame_only" and not cross) or (scope == "same_frame_only" and cross):
                        continue
                    expected, predicted = pair["expected_same"], pair["predictions"][policy]
                    tp += int(expected and predicted)
                    fp += int(not expected and predicted)
                    tn += int(not expected and not predicted)
                    fn += int(expected and not predicted)
            groups[scope][policy] = {
                "pairs": tp + fp + tn + fn, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
                "f1": ratio(2 * tp, 2 * tp + fp + fn),
            }
            if scope == "all_pairs":
                assert abs(groups[scope][policy]["f1"] - source["metrics"][policy]["f1"]) < 1e-10
    return {"source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "label_rule": source["contract"]["target_assignment"], "counts": source["counts"], "groups": groups}


def run(root, output):
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "post-hoc CPU cache diagnostics; historically exposed scenes",
        "calibration_fitted": False, "threshold_selected": False,
        "default_production_policy_changed": False,
        "limitations": [
            "Inverse-direction negatives share objects, anchors and pairs with positives; queries are not independent samples.",
            "Near-boundary pairs and multi-GT tasks are excluded by the existing query protocol.",
            "Padded OBB containment is an alignment sensitivity diagnostic, not strict accuracy or a confidence interval.",
            "The candidate-localization ceiling is evaluator-only and ignores ranking/gating errors.",
            "Reference semantic weighting is a lexical Jaccard score ablation, not a learned semantic model.",
        ],
        "code_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in (
            "scripts/audit_relation_abstention.py", "relground/relations.py",
            "relground/association.py", "relground/calibration.py", "relground/schemas.py",
            "relground/clio_relation_benchmark.py",
        )},
        "scenes": {},
    }
    csv_rows = []
    for scene, directory in SCENES.items():
        bundle = root / "runs" / directory / "relation-benchmark-v2"
        saved = read(bundle / "prediction.json")
        old_eval = read(bundle / "evaluation.json")
        queries = [GroundingQuery.from_dict(q) for q in saved["queries"]]
        memory = ObjectMemory.load(bundle / "scene_object_memory.json")
        anchors = {k: np.asarray(v) for k, v in read(bundle / "anchor_poses.json").items()}
        raw = [RelationGrounder(memory, anchors).ground(q) for q in queries]
        replay = [AbstentionPolicy(0.6).apply(deepcopy(r), r.confidence).to_dict() if not r.abstain else r.to_dict() for r in raw]
        if replay != saved["results"]:
            raise ValueError(f"{scene}: frozen prediction replay changed")
        weighted_config = replace(RelationConfig(), score_reference_semantics=True)
        weighted = [RelationGrounder(memory, anchors, weighted_config).ground(q) for q in queries]
        # Prediction methods are fully generated before opening evaluator labels.
        methods = {
            "frozen_default": replay,
            "proposals_without_gates": policy_results(raw, "proposals_without_gates"),
            "geometry_and_ambiguity_only": policy_results(raw, "geometry_and_ambiguity_only"),
            "reference_weighted_default": policy_results(weighted, "threshold", 0.6),
        }
        for name, values in (("frozen", raw), ("reference_weighted", weighted)):
            for threshold in THRESHOLDS:
                methods[f"{name}_threshold_{threshold:.2f}"] = policy_results(values, "threshold", threshold)
        labels = read(bundle / "labels.json")
        evaluations = {}
        destination = output / scene
        destination.mkdir(exist_ok=True)
        for method, results in methods.items():
            prediction = {"scene_id": saved["scene_id"], "split_role": saved["split_role"], "queries": saved["queries"], "results": results}
            evaluated = evaluate_clio_relation_prediction(prediction, labels, source={}, generation=old_eval["generation"], created_at="post-hoc-diagnostic")
            if method == "frozen_default" and (evaluated["rows"] != old_eval["rows"] or evaluated["metrics"] != old_eval["metrics"]):
                raise ValueError(f"{scene}: frozen evaluator replay changed")
            evaluations[method] = evaluated
            if method in ("frozen_default", "reference_weighted_default", "proposals_without_gates", "geometry_and_ambiguity_only"):
                write(destination / f"{method}.json", {"scope": report["scope"], "results": results, "rows": evaluated["rows"]})
        summaries = {method: summarize(e) for method, e in evaluations.items()}
        for method, metrics in summaries.items():
            csv_rows.append({"scene": scene, "method": method, **{k: v for k, v in metrics.items() if not isinstance(v, dict)}})
        report["scenes"][scene] = {
            "source_sha256": {str((bundle / name).relative_to(root)): hashlib.sha256((bundle / name).read_bytes()).hexdigest() for name in ("prediction.json", "evaluation.json", "labels.json", "scene_object_memory.json", "anchor_poses.json")},
            "frozen_prediction_and_evaluation_replay": "PASS",
            "generation": old_eval["generation"],
            "availability": availability(labels),
            "association_metric_audit": association_scope(root / "runs" / directory / "association_benchmark.json"),
            "methods": summaries,
            "reference_weighted_changes": compare_rows(evaluations["frozen_default"], evaluations["reference_weighted_default"]),
        }
    write(output / "summary.json", report)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({"status": "PASS", "output": str(output), "scenes": list(report["scenes"])}, indent=2))


def plot_summary(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report = read(path)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for column, (scene, data) in enumerate(report["scenes"].items()):
        for row, metric in enumerate(("strict", "padded")):
            ax = axes[row, column]
            for prefix, label, color in (("frozen", "Frozen score", "#3769ac"), ("reference_weighted", "+ reference match score", "#b44d28")):
                points = [data["methods"][f"{prefix}_threshold_{t:.2f}"] for t in THRESHOLDS]
                points = sorted((m for m in points if m["answers"]), key=lambda m: m["coverage"])
                ax.plot([m["coverage"] for m in points], [m[f"answered_{metric}_risk"] for m in points], "o-", color=color, label=label)
                current = data["methods"][f"{prefix}_threshold_0.60"]
                ax.scatter([current["coverage"]], [current[f"answered_{metric}_risk"]], marker="*", s=170, color=color, edgecolor="black", zorder=5)
            ungated = data["methods"]["proposals_without_gates"]
            ax.scatter([ungated["coverage"]], [ungated[f"answered_{metric}_risk"]], marker="x", color="gray", s=60, label="Proposals without gates")
            ax.set_title(f"{scene.title()} / {metric}")
            ax.set_xlim(0, 0.75)
            ax.set_ylim(-0.03, 1.06)
            ax.grid(alpha=0.2)
            if row == 1:
                ax.set_xlabel("Answers / all positive + negative queries")
            if column == 0:
                ax.set_ylabel("Incorrect answers / emitted answers")
    axes[0, 1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Relation abstention: fixed-cache threshold diagnostics", fontsize=14)
    fig.text(0.5, 0.012, "Stars: threshold 0.60. Padded = alignment-RMSE-expanded OBBs. No threshold selected; zero-answer risk undefined.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    for extension in ("png", "pdf"):
        fig.savefig(path.parent / f"risk_coverage.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/interview-relation-audit-20260909"))
    parser.add_argument("--plot-summary", type=Path, help="Plot a saved summary using an environment with matplotlib")
    args = parser.parse_args()
    if args.plot_summary:
        plot_summary(args.plot_summary.resolve())
    else:
        run(Path(__file__).resolve().parents[1], args.output.resolve())
