"""Build and validate the lightweight post-D21 PE fusion summary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "0.1"
STAGE = "post-D21-PE-mask-crop-representative-center-summary"
METHODS = (
    "current_q1f",
    "quality_representative",
    "medoid_representative",
    "pe_semantic_representative",
    "raw_observation_oracle",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("summary source escapes project root") from error


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _method(metric: Mapping[str, Any]) -> dict[str, Any]:
    task_count = int(metric["task_count"])
    return {
        "task_count": task_count,
        "answered_tasks": int(metric["answered_tasks"]),
        "strict_correct": round(
            task_count * float(metric["grounding_acc_at_1"])
        ),
        "strict_acc_at_1": float(metric["grounding_acc_at_1"]),
        "padded_correct": round(
            task_count
            * float(
                metric[
                    "grounding_acc_at_1_with_alignment_rmse_margin"
                ]
            )
        ),
        "padded_acc_at_1": float(
            metric["grounding_acc_at_1_with_alignment_rmse_margin"]
        ),
    }


def _scene(
    *,
    role: str,
    prediction: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    if prediction["scene_id"] != evaluation["scene_id"]:
        raise ValueError("prediction/evaluation scene mismatch")
    if prediction["experiment_id"] != evaluation["experiment_id"]:
        raise ValueError("prediction/evaluation experiment mismatch")
    methods = {
        key: _method(evaluation["metrics"][key])
        for key in METHODS
    }
    prediction_by_task = {
        str(item["task"]): item for item in prediction["tasks"]
    }
    rescued_tasks = []
    for row in evaluation["tasks"]:
        if (
            not row["current_q1f"]["correct"]
            and row["pe_semantic_representative"]["correct"]
        ):
            prediction_row = prediction_by_task[str(row["task"])]
            selected_id = prediction_row["selected_observation_id"]
            selected = next(
                item
                for item in prediction_row["observations"]
                if item["observation_id"] == selected_id
            )
            rescued_tasks.append({
                "task": str(row["task"]),
                "selected_observation_id": selected_id,
                "semantic_score": float(selected["semantic_score"]),
            })
    return {
        "role": role,
        "counts": dict(prediction["counts"]),
        "methods": methods,
        "comparisons": dict(evaluation["comparisons"]),
        "rescued_tasks": rescued_tasks,
    }


def build_summary(
    *,
    project_root: str | Path,
    config_path: str | Path,
    apartment_prediction_path: str | Path,
    apartment_evaluation_path: str | Path,
    cubicle_prediction_path: str | Path,
    cubicle_evaluation_path: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    paths = {
        "apartment_prediction": Path(apartment_prediction_path).resolve(),
        "apartment_evaluation": Path(apartment_evaluation_path).resolve(),
        "cubicle_prediction": Path(cubicle_prediction_path).resolve(),
        "cubicle_evaluation": Path(cubicle_evaluation_path).resolve(),
    }
    config = _load(config_path)
    payloads = {key: _load(path) for key, path in paths.items()}
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "stage": STAGE,
        "experiment_id": config["experiment_id"],
        "method": {
            "encoder": config["encoder"],
            "text_source": config["text_source"],
            "crop": {
                "context_padding_fraction": config[
                    "context_padding_fraction"
                ],
                "min_padding_pixels": config["min_padding_pixels"],
                "masked_background_value": config[
                    "masked_background_value"
                ],
                "semantic_score": config["semantic_score"],
            },
            "object_selection": config["object_selection"],
            "center_rule": config["center_rule"],
            "fallback": config["fallback"],
            "learned_parameters": config["learned_parameters"],
        },
        "scenes": {
            "apartment": _scene(
                role="development",
                prediction=payloads["apartment_prediction"],
                evaluation=payloads["apartment_evaluation"],
            ),
            "cubicle": _scene(
                role="fixed-confirmatory",
                prediction=payloads["cubicle_prediction"],
                evaluation=payloads["cubicle_evaluation"],
            ),
        },
        "claim_boundary": {
            "development_pe_specific_gain_over_quality_baseline": False,
            "cubicle_incremental_strict_gain_over_q1f": True,
            "cubicle_incremental_strict_gain_over_quality_baseline": True,
            "cubicle_padded_gain_over_q1f": False,
            "learned_reranker": False,
            "siglip2_used": False,
            "untouched_held_out_claim": False,
            "statistical_significance_claim": False,
            "d21_headline_replaced": False,
        },
        "sources": {
            "config": {
                "path": _relative(root, config_path),
                "sha256": sha256_file(config_path),
            },
            **{
                key: {
                    "path": _relative(root, path),
                    "sha256": sha256_file(path),
                }
                for key, path in paths.items()
            },
        },
    }
    return summary


def validate_summary(
    payload: Mapping[str, Any],
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    try:
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported summary schema")
        if payload.get("stage") != STAGE or payload.get("status") != "PASS":
            raise ValueError("summary stage/status mismatch")
        if set(payload["scenes"]) != {"apartment", "cubicle"}:
            raise ValueError("scene inventory changed")
        if payload["scenes"]["apartment"]["role"] != "development":
            raise ValueError("Apartment must remain development")
        if payload["scenes"]["cubicle"]["role"] != "fixed-confirmatory":
            raise ValueError("Cubicle must remain fixed-confirmatory")
        for scene in payload["scenes"].values():
            for method in METHODS:
                metric = scene["methods"][method]
                total = int(metric["task_count"])
                if total != 18:
                    raise ValueError("task denominator must remain 18")
                if abs(
                    float(metric["strict_acc_at_1"])
                    - int(metric["strict_correct"]) / total
                ) > 1e-11:
                    raise ValueError("strict accuracy is not reproducible")
                if abs(
                    float(metric["padded_acc_at_1"])
                    - int(metric["padded_correct"]) / total
                ) > 1e-11:
                    raise ValueError("padded accuracy is not reproducible")
            for comparison in scene["comparisons"].values():
                transitions = comparison.get("paired_strict_transitions")
                if transitions is None:
                    continue
                if sum(int(value) for value in transitions.values()) != 18:
                    raise ValueError("paired transition denominator changed")
        apartment = payload["scenes"]["apartment"]["methods"]
        cubicle = payload["scenes"]["cubicle"]["methods"]
        if (
            apartment["current_q1f"]["strict_correct"] != 2
            or apartment["quality_representative"]["strict_correct"] != 3
            or apartment["pe_semantic_representative"]["strict_correct"] != 3
        ):
            raise ValueError("Apartment development result changed")
        if (
            cubicle["current_q1f"]["strict_correct"] != 7
            or cubicle["quality_representative"]["strict_correct"] != 7
            or cubicle["medoid_representative"]["strict_correct"] != 8
            or cubicle["pe_semantic_representative"]["strict_correct"] != 8
            or cubicle["pe_semantic_representative"]["padded_correct"] != 13
        ):
            raise ValueError("Cubicle fixed-confirmatory result changed")
        boundary = payload["claim_boundary"]
        required_false = (
            "development_pe_specific_gain_over_quality_baseline",
            "cubicle_padded_gain_over_q1f",
            "learned_reranker",
            "siglip2_used",
            "untouched_held_out_claim",
            "statistical_significance_claim",
            "d21_headline_replaced",
        )
        if any(boundary.get(key) is not False for key in required_false):
            raise ValueError("claim boundary was overstated")
        if (
            boundary.get("cubicle_incremental_strict_gain_over_q1f")
            is not True
            or boundary.get(
                "cubicle_incremental_strict_gain_over_quality_baseline"
            )
            is not True
        ):
            raise ValueError("observed Cubicle comparison was hidden")
        if project_root is not None:
            root = Path(project_root).resolve()
            config = payload["sources"]["config"]
            config_path = root / str(config["path"])
            if (
                not config_path.is_file()
                or sha256_file(config_path) != config["sha256"]
            ):
                raise ValueError("tracked config hash mismatch")
    except (KeyError, TypeError, ValueError) as error:
        failures.append(str(error))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not failures else "FAIL",
        "stage": f"{STAGE}-validation",
        "checks": {
            "arithmetic_recomputed": not failures,
            "denominators_frozen": not failures,
            "development_baselines_visible": not failures,
            "cubicle_scope_honest": not failures,
            "claim_boundary_honest": not failures,
            "tracked_config_hash_matches": not failures,
        },
        "failures": failures,
    }
