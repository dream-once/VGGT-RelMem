"""D20 reproducibility package and JSON-derived result tables."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
import json
import re


D20_SCHEMA_VERSION = "0.1"
D20_STATUS = "CPU_COMPLETE"
D20_BINARY_STATUS = "OPTIONAL_BINARY_RELEASE_PENDING"
MANIFEST_FIELDS = (
    "schema_version",
    "stage",
    "status",
    "optional_binary_release_status",
    "repository_base_commit",
    "inputs",
    "validators",
    "outputs",
    "readme_checks",
    "evidence_policy",
    "frozen_at",
)
INPUT_FIELDS = ("input_id", "path", "sha256")
OUTPUT_FILES = (
    "result_tables.json",
    "qxa_table.md",
    "relation_table.md",
    "ablation_table.md",
)


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _relative(value: Any, name: str) -> str:
    path = Path(str(value))
    if not str(value) or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be repository-relative")
    return path.as_posix()


def validate_reproduction_manifest(
    payload: Mapping[str, Any],
    *,
    project_root: str | Path | None = None,
) -> None:
    if set(payload) != set(MANIFEST_FIELDS):
        raise ValueError("D20 manifest fields are not frozen")
    if payload["schema_version"] != D20_SCHEMA_VERSION:
        raise ValueError("unsupported D20 schema")
    if payload["stage"] != "D20-reproduction-package":
        raise ValueError("D20 stage changed")
    if payload["status"] != D20_STATUS:
        raise ValueError("D20 status must remain CPU_COMPLETE")
    if payload["optional_binary_release_status"] != D20_BINARY_STATUS:
        raise ValueError("D20 binary-release boundary changed")
    if re.fullmatch(
        r"[0-9a-f]{40}", str(payload["repository_base_commit"])
    ) is None:
        raise ValueError("D20 base commit must be a Git hash")
    inputs = payload["inputs"]
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("D20 inputs must be a non-empty list")
    identifiers: list[str] = []
    for item in inputs:
        if not isinstance(item, Mapping) or set(item) != set(INPUT_FIELDS):
            raise ValueError("D20 input fields are not frozen")
        identifiers.append(str(item["input_id"]))
        reference = _relative(item["path"], "D20 input path")
        digest = str(item["sha256"])
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("D20 input hash must be SHA-256")
        if project_root is not None:
            root = Path(project_root).resolve()
            if sha256_file(root / reference) != digest:
                raise ValueError(f"D20 input hash mismatch: {reference}")
    expected_ids = [
        "d15_5_manifest",
        "d15_5_viewpoint_audit",
        "d17_evaluation",
        "d18_prediction",
        "d18_evaluation",
        "d19_prediction",
        "d19_evaluation",
    ]
    if identifiers != expected_ids:
        raise ValueError("D20 input identity or order changed")
    if payload["validators"] != ["D16", "D17", "D18", "D19"]:
        raise ValueError("D20 validator set changed")
    if payload["outputs"] != list(OUTPUT_FILES):
        raise ValueError("D20 output inventory changed")
    if payload["readme_checks"] != [
        "d15_5_counts",
        "d17_query_counts",
        "d19_zero_delta_boundary",
        "project_scope",
    ]:
        raise ValueError("D20 README checks changed")
    if payload["evidence_policy"] != {
        "extensions": [".json", ".md"],
        "bundle_limit_bytes": 131072,
        "week_limit_bytes": 786432,
    }:
        raise ValueError("D20 evidence policy changed")


def input_paths(
    manifest: Mapping[str, Any], root: str | Path
) -> dict[str, Path]:
    project = Path(root).resolve()
    return {
        str(item["input_id"]): project / str(item["path"])
        for item in manifest["inputs"]
    }


def _format(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def build_result_tables(
    project_root: str | Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    validate_reproduction_manifest(manifest, project_root=project_root)
    paths = input_paths(manifest, project_root)
    visual = load_json(paths["d15_5_manifest"])
    viewpoint = load_json(paths["d15_5_viewpoint_audit"])
    relation = load_json(paths["d17_evaluation"])
    d18_prediction = load_json(paths["d18_prediction"])
    d18_evaluation = load_json(paths["d18_evaluation"])
    d19_prediction = load_json(paths["d19_prediction"])
    d19_evaluation = load_json(paths["d19_evaluation"])

    eval_by_combo = {
        item["combination_id"]: item
        for item in d18_evaluation["rows"]
    }
    qxa_rows = []
    for row in d18_prediction["matrix_rows"]:
        evaluated = eval_by_combo[row["combination_id"]]
        metrics = evaluated["metrics"]
        qxa_rows.append({
            "combination_id": row["combination_id"],
            "matrix_role": row["matrix_role"],
            "status": row["status"],
            "selected_count": row["selected_count"],
            "observation_count": row["observation_count"],
            "pair_count": (
                None if metrics is None else metrics["pair_count"]
            ),
            "f1": None if metrics is None else metrics["f1"],
            "scope": "synthetic_correctness_not_performance",
        })

    relation_metrics = relation["metrics"]
    relation_rows = [
        {"metric": key, "value": relation_metrics[key]}
        for key in (
            "query_count",
            "positive_count",
            "negative_count",
            "task_accuracy",
            "negative_rejection_accuracy",
            "brier",
            "ece_10",
            "aurc_discrete",
        )
    ]

    q2_base = d19_evaluation["q2_ablations"][0]["metrics"]
    q2_rows = []
    for row in d19_evaluation["q2_ablations"]:
        metrics = row["metrics"]
        q2_rows.append({
            "variant_id": row["variant_id"],
            "changed_factor": row["changed_factor"],
            "observed_instance_recall": metrics[
                "observed_instance_recall"
            ],
            "recall_delta": (
                metrics["observed_instance_recall"]
                - q2_base["observed_instance_recall"]
            ),
            "sam_calls": metrics["sam_calls"],
            "sam_call_delta": metrics["sam_calls"] - q2_base["sam_calls"],
        })
    a2_base = d19_evaluation["a2_ablations"][0]["metrics"]
    a2_rows = []
    for row in d19_evaluation["a2_ablations"]:
        metrics = row["metrics"]
        a2_rows.append({
            "variant_id": row["variant_id"],
            "changed_factor": row["changed_factor"],
            "f1": metrics["f1"],
            "f1_delta": metrics["f1"] - a2_base["f1"],
            "failure_count": row["failure_count"],
        })
    zero_delta = all(
        row["recall_delta"] == 0.0 and row["sam_call_delta"] == 0
        for row in q2_rows
    ) and all(row["f1_delta"] == 0.0 for row in a2_rows)

    counts = visual["counts"]
    status_counts = viewpoint["status_counts"]
    return {
        "schema_version": D20_SCHEMA_VERSION,
        "status": "PASS",
        "stage": "D20-derived-results",
        "sources": {
            key: {
                "path": paths[key].relative_to(
                    Path(project_root).resolve()
                ).as_posix(),
                "sha256": sha256_file(paths[key]),
            }
            for key in paths
        },
        "d15_5": {
            "anchor_cameras": counts["anchor_cameras"],
            "selected_cameras": counts["selected_cameras"],
            "observations": counts["observations"],
            "predicted_objects": counts["predicted_objects"],
            "ply_points": counts["ply_points"],
            "video_frames": counts["video_frames"],
            "strict_multiview_objects": status_counts[
                "STRICT_MULTIVIEW"
            ],
            "diagnostic_parallax_objects": status_counts[
                "DIAGNOSTIC_PARALLAX"
            ],
            "binary_artifacts_retained_in_git": False,
            "binary_release_status": D20_BINARY_STATUS,
        },
        "qxa": {
            "scope": "synthetic_correctness_not_performance",
            "rows": qxa_rows,
        },
        "relations": {
            "scope": "synthetic_correctness_not_real_calibration",
            "rows": relation_rows,
        },
        "ablations": {
            "scope": "synthetic_correctness_not_real_ablation",
            "q2_rows": q2_rows,
            "a2_rows": a2_rows,
            "all_reported_deltas_zero": zero_delta,
            "historical_success_ablation": d19_prediction[
                "historical_success_ablation"
            ],
        },
        "failure_audit": d19_evaluation["failure_audit"],
        "claim_boundary": {
            "clio_held_out": "PENDING",
            "real_calibration": "PENDING",
            "real_ablation": "PENDING",
            "performance_improvement": None,
        },
    }


def render_qxa_table(results: Mapping[str, Any]) -> str:
    lines = [
        "# Q×A synthetic correctness table",
        "",
        "> Correctness fixture only; not held-out performance.",
        "",
        "| combination | role | status | frames | observations | pairs | F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in results["qxa"]["rows"]:
        lines.append(
            "| {combination_id} | {matrix_role} | {status} | "
            "{selected_count} | {observation_count} | {pair_count} | "
            "{f1} |".format(
                **{key: _format(value) for key, value in row.items()}
            )
        )
    return "\n".join(lines) + "\n"


def render_relation_table(results: Mapping[str, Any]) -> str:
    lines = [
        "# Relation and abstention synthetic table",
        "",
        "> Synthetic correctness only; real calibration remains pending.",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {row['metric']} | {_format(row['value'])} |"
        for row in results["relations"]["rows"]
    )
    return "\n".join(lines) + "\n"


def render_ablation_table(results: Mapping[str, Any]) -> str:
    lines = [
        "# D19 synthetic one-factor ablations",
        "",
        "> Correctness ablation only; all current deltas are zero.",
        "",
        "## Q2",
        "",
        "| variant | factor | recall | Δ recall | SAM calls | Δ calls |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in results["ablations"]["q2_rows"]:
        lines.append(
            "| {variant_id} | {changed_factor} | "
            "{observed_instance_recall} | {recall_delta} | "
            "{sam_calls} | {sam_call_delta} |".format(
                **{key: _format(value) for key, value in row.items()}
            )
        )
    lines.extend([
        "",
        "## A2",
        "",
        "| variant | factor | F1 | Δ F1 | failures |",
        "|---|---|---:|---:|---:|",
    ])
    for row in results["ablations"]["a2_rows"]:
        lines.append(
            "| {variant_id} | {changed_factor} | {f1} | "
            "{f1_delta} | {failure_count} |".format(
                **{key: _format(value) for key, value in row.items()}
            )
        )
    return "\n".join(lines) + "\n"


def readme_numeric_checks(
    readme_text: str, results: Mapping[str, Any]
) -> dict[str, bool]:
    d15 = results["d15_5"]
    relation = {
        row["metric"]: row["value"]
        for row in results["relations"]["rows"]
    }
    return {
        "d15_5_counts": all(
            snippet in readme_text
            for snippet in (
                f"{d15['anchor_cameras']} 帧几何",
                f"{d15['observations']} 条有效 3D observation",
                f"{d15['predicted_objects']} 个满足跨帧支持",
                f"{d15['ply_points']:,} 个顶点",
                (
                    f"{d15['predicted_objects']} 个对象中 "
                    f"{d15['strict_multiview_objects']} 个"
                ),
            )
        ),
        "d17_query_counts": (
            f"{relation['positive_count']} 个正查询与 "
            f"{relation['negative_count']} 个负查询"
        ) in readme_text,
        "d19_zero_delta_boundary": (
            results["ablations"]["all_reported_deltas_zero"]
            and "synthetic 数值无变化" in readme_text
        ),
        "project_scope": (
            "不称“完整导航系统”" in readme_text
            or "不称完整导航系统" in readme_text
        ),
    }


def write_derived_outputs(
    output_dir: str | Path,
    results: Mapping[str, Any],
) -> dict[str, str]:
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    payloads = {
        "result_tables.json": (
            json.dumps(results, ensure_ascii=False, indent=2) + "\n"
        ),
        "qxa_table.md": render_qxa_table(results),
        "relation_table.md": render_relation_table(results),
        "ablation_table.md": render_ablation_table(results),
    }
    for name, text in payloads.items():
        (output / name).write_text(text, encoding="utf-8")
    return {
        name: sha256_file(output / name)
        for name in OUTPUT_FILES
    }
