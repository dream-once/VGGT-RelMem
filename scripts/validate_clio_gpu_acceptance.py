"""Validate the lightweight public Clio apartment GPU acceptance evidence."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from relground.candidate_cache import CandidateOutcomeCache


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contained(root: Path, value: str) -> Path:
    ref = Path(value)
    if ref.is_absolute() or ".." in ref.parts:
        raise ValueError("artifact paths must be repository-relative")
    path = (root / ref).resolve()
    if path != root and root not in path.parents:
        raise ValueError("artifact path escapes project root")
    return path


def validate(report_path: str | Path, *, project_root: str | Path, verify_local: bool = False) -> dict[str, object]:
    root = Path(project_root).resolve()
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    checks["contract"] = (
        report.get("schema_version") == "clio-gpu-acceptance/0.1"
        and report.get("status") == "PASS"
        and report.get("stage") == "clio-apartment-development-gpu-acceptance"
    )
    scope = report["scope"]
    checks["split_guard"] = (
        scope["scene_id"] == "apartment"
        and scope["role"] == "development"
        and scope["held_out_scene"] == "cubicle"
        and scope["held_out_downloaded"] is False
        and scope["held_out_evaluated"] is False
    )
    dataset = report["dataset"]
    checks["data_boundary"] = (
        dataset["materialization"] == "RGB_TASK_METADATA_DEVELOPMENT_SUBSET"
        and dataset["selected_rgb_frames"] == 24
        and dataset["full_scene_modalities_materialized"] is False
        and dataset["redistribution_allowed"] is False
    )
    geometry = report["geometry"]
    checks["genuine_multiview"] = (
        geometry["validator_status"] == "PASS"
        and geometry["frame_count"] == 24
        and geometry["max_translation_reconstruction_units"] > 0.15
        and geometry["max_rotation_deg"] > 3.0
    )
    perception = report["perception"]
    checks["query_specific_multiview"] = (
        perception["d5_status"] == "PASS"
        and perception["d6_status"] == "PASS"
        and perception["candidate_count"] == 24
        and perception["available_candidates"] == 24
        and perception["lifted_instances"] == 3
        and len(perception["frames_with_lifted_observations"]) == 2
        and perception["query_specific_translation_reconstruction_units"] > 0.15
        and perception["query_specific_rotation_deg"] > 3.0
    )
    downstream = report["downstream"]
    checks["downstream_honest_failure"] = (
        downstream["d7_status"] == "PASS"
        and downstream["d8_status"] == "PASS"
        and downstream["d12_status"] == "PASS"
        and downstream["d12_promoted_clusters"] == 0
        and downstream["d12_permanent_objects"] == 0
    )
    public = report["artifacts"]["public"]
    public_ok = True
    for artifact in public:
        path = _contained(root, artifact["path"])
        public_ok = public_ok and path.is_file() and path.stat().st_size == artifact["bytes"] and _hash(path) == artifact["sha256"]
    checks["public_artifact_hashes"] = public_ok
    cache_path = _contained(root, public[0]["path"])
    cache = CandidateOutcomeCache.load(cache_path).to_dict()
    checks["complete_cache"] = (
        cache["materialization_status"] == "complete"
        and cache["counts"]["candidate_count"] == 24
        and cache["counts"]["available_candidates"] == 24
    )
    rows = report["qxa_rows"]
    checks["qxa_replay"] = (
        len(rows) == 6
        and all(row["status"] == "PASS" for row in rows)
        and any(row["combination_id"].startswith("Q0") and row["observation_count"] == 0 for row in rows)
        and all(row["permanent_objects"] == 0 for row in rows)
    )
    findings = set(report["failure_findings"])
    checks["failure_boundaries"] = {
        "Q0_TOP1_RETURNED_ZERO_OBSERVATIONS",
        "Q2_STOPPED_AFTER_TWO_CONSECUTIVE_LOW_GAIN_WITH_ONE_OBSERVATION",
        "A2_PROMOTED_ZERO_PERMANENT_OBJECTS",
        "CUBICLE_HELD_OUT_UNTOUCHED",
    }.issubset(findings)
    if verify_local:
        local_ok = True
        for artifact in report["artifacts"]["local_not_in_git"]:
            path = _contained(root, artifact["path"])
            local_ok = local_ok and path.is_file() and path.stat().st_size == artifact["bytes"] and _hash(path) == artifact["sha256"]
        checks["local_artifact_hashes"] = local_ok
    status = "PASS" if all(checks.values()) else "FAIL"
    return {"schema_version":"0.1","status":status,"stage":"Clio-GPU-validation","checks":checks,"scope":"development_engineering_replay_not_performance","held_out_status":"CLIO_CUBICLE_UNTOUCHED"}

def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--verify-local", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = validate(args.report, project_root=args.project_root, verify_local=args.verify_local)
    if args.output:
        Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result["status"] == "PASS" else 2

if __name__ == "__main__":
    raise SystemExit(main())
