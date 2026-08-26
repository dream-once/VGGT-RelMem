"""Independently validate and replay a D11 candidate-cache bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from relground.candidate_cache import (
    CandidateOutcomeCache,
    VisualMemoryManifest,
    build_d11_payloads,
)
from relground.observation_cache import sha256_file


REQUIRED_ARTIFACTS = (
    "candidate_cache.json",
    "visual_memory_manifest.json",
    "source_d5_retrieval.json",
    "source_d6_result.json",
    "source_d7_observations.json",
    "run_manifest.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} root must be an object")
    return payload


def _resolve_contained(root: Path, reference: str, name: str) -> Path:
    relative = Path(reference)
    if relative.is_absolute():
        raise ValueError(f"{name} must be relative")
    boundary = root.resolve()
    candidate = (boundary / relative).resolve()
    if candidate != boundary and boundary not in candidate.parents:
        raise ValueError(f"{name} escapes its allowed root")
    return candidate


def validate_output(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(path).resolve()
    repository = Path(project_root or Path.cwd()).resolve()
    failures: list[str] = []
    for name in REQUIRED_ARTIFACTS:
        artifact = root / name
        if not artifact.is_file() or artifact.stat().st_size == 0:
            failures.append(f"missing or empty artifact: {name}")
    if failures:
        return {"status": "FAIL", "failures": failures}

    try:
        raw_cache = _read_json(root / "candidate_cache.json")
        raw_visual = _read_json(root / "visual_memory_manifest.json")
        cache = CandidateOutcomeCache.from_dict(raw_cache).to_dict()
        visual = VisualMemoryManifest.from_dict(raw_visual).to_dict()
        manifest = _read_json(root / "run_manifest.json")

        expected_artifacts = {
            "visual_memory_manifest": "visual_memory_manifest.json",
            "d5_retrieval": "source_d5_retrieval.json",
            "d6_result": "source_d6_result.json",
            "d7_observations": "source_d7_observations.json",
        }
        if cache["artifacts"] != expected_artifacts:
            raise ValueError("candidate-cache artifact references changed")
        resolved = {
            key: _resolve_contained(root, value, f"artifacts.{key}")
            for key, value in cache["artifacts"].items()
        }
        if resolved["visual_memory_manifest"] != (
            root / "visual_memory_manifest.json"
        ):
            raise ValueError("visual-memory artifact reference is inconsistent")

        retrieval = _read_json(resolved["d5_retrieval"])
        d6_result = _read_json(resolved["d6_result"])
        observations = _read_json(resolved["d7_observations"])
        source_hashes = {
            "d5_retrieval_sha256": sha256_file(resolved["d5_retrieval"]),
            "d6_result_sha256": sha256_file(resolved["d6_result"]),
            "d7_observations_sha256": sha256_file(
                resolved["d7_observations"]
            ),
        }
        if cache["sources"] != source_hashes:
            raise ValueError("retained D5/D6/D7 source hash changed")

        image_refs: dict[str, str] = {}
        image_hashes: dict[str, str] = {}
        for frame in visual["frames"]:
            frame_id = str(frame["frame_id"])
            image_ref = str(frame["image_ref"])
            image_path = _resolve_contained(
                repository, image_ref, f"frames.{frame_id}.image_ref"
            )
            if not image_path.is_file():
                raise ValueError(f"missing source image for {frame_id}")
            image_refs[frame_id] = image_ref
            image_hashes[frame_id] = sha256_file(image_path)
            if image_hashes[frame_id] != frame["image_sha256"]:
                raise ValueError(f"source image hash changed for {frame_id}")

        expected_visual, expected_cache = build_d11_payloads(
            retrieval=retrieval,
            d6_result=d6_result,
            observations_payload=observations,
            scene_id=str(cache["scene_id"]),
            query_id=str(cache["query_id"]),
            image_refs=image_refs,
            image_hashes=image_hashes,
            source_hashes=source_hashes,
            artifact_refs=expected_artifacts,
            created_at=str(cache["created_at"]),
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {
            "status": "FAIL",
            "failures": [f"invalid D11 artifact: {error}"],
        }

    if raw_cache != cache or raw_visual != visual:
        failures.append("D11 schema round-trip changed JSON")
    if cache != expected_cache:
        failures.append("candidate cache differs from deterministic replay")
    if visual != expected_visual:
        failures.append("visual-memory manifest differs from deterministic replay")
    if cache["candidate_universe"] != [
        str(item["frame_id"]) for item in retrieval["raw_ranking"]
    ]:
        failures.append("candidate universe is not conserved from D5 ranking")

    config = manifest.get("config", {})
    if not isinstance(config, Mapping):
        failures.append("run manifest config is not an object")
    else:
        expected_manifest = {
            "stage": "D11",
            "status": "CPU_COMPLETE",
            "gpu_acceptance": "PENDING",
            "candidate_cache": "candidate_cache.json",
            "materialization_status": cache["materialization_status"],
            "source_hashes": cache["sources"],
        }
        for key, value in expected_manifest.items():
            if config.get(key) != value:
                failures.append(f"run manifest {key} is inconsistent")
    if manifest.get("peak_vram_mb") is not None:
        failures.append("CPU-only D11 unexpectedly records GPU memory")

    accepted_status = (
        "PASS"
        if cache["materialization_status"] == "complete"
        else "PASS_WITH_UNMATERIALIZED_OUTCOMES"
    )
    return {
        "status": "FAIL" if failures else accepted_status,
        "failures": failures,
        "stage": "D11",
        "cpu_completion": "COMPLETE",
        "gpu_acceptance": "PENDING",
        "scene_id": cache["scene_id"],
        "query_id": cache["query_id"],
        "embedding_status": visual["embedding_status"],
        **cache["counts"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--report")
    args = parser.parse_args()
    report = validate_output(
        args.output_dir,
        project_root=args.project_root,
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    if report["status"] not in {
        "PASS",
        "PASS_WITH_UNMATERIALIZED_OUTCOMES",
    }:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
