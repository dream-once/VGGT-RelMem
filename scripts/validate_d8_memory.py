"""Validate a D8 frozen ObjectMemory schema and round-trip artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from relground.association import ObjectMemory
from relground.observation_cache import sha256_file
from relground.schemas import (
    MEMORY_OBJECT_SCHEMA_VERSION,
    OBJECT_MEMORY_SCHEMA_VERSION,
    OBJECT_OBSERVATION_SCHEMA_VERSION,
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} root must be an object")
    return payload


def validate_output(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    failures: list[str] = []
    result_path = root / "d8_result.json"
    memory_path = root / "object_memory.json"
    run_manifest_path = root / "run_manifest.json"
    for artifact in (result_path, memory_path, run_manifest_path):
        if not artifact.is_file() or artifact.stat().st_size == 0:
            failures.append(
                f"missing or empty artifact: {artifact.name}"
            )
    if failures:
        return {"status": "FAIL", "failures": failures}

    try:
        result = read_json(result_path)
        raw_memory = read_json(memory_path)
        run_manifest = read_json(run_manifest_path)
        memory = ObjectMemory.load(memory_path)
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        return {
            "status": "FAIL",
            "failures": [f"invalid D8 artifact: {error}"],
        }

    if result.get("status") != "PASS" or result.get("stage") != "D8":
        failures.append("result is not a passing D8 artifact")
    if result.get("schema_version") != OBJECT_MEMORY_SCHEMA_VERSION:
        failures.append("result ObjectMemory schema version is unsupported")
    versions = result.get("version_fields", {})
    expected_versions = {
        "object_memory": OBJECT_MEMORY_SCHEMA_VERSION,
        "memory_object": MEMORY_OBJECT_SCHEMA_VERSION,
        "object_observation": OBJECT_OBSERVATION_SCHEMA_VERSION,
    }
    if versions != expected_versions:
        failures.append("result version fields are inconsistent")
    if raw_memory != memory.to_dict():
        failures.append("ObjectMemory canonical round-trip changed JSON")
    if memory.objects:
        failures.append("D8 unexpectedly contains permanent objects")
    if memory.decisions:
        failures.append("D8 unexpectedly contains association decisions")
    pending = list(memory.pending_observations.values())
    if not pending:
        failures.append("D8 has no pending observations")
    frame_ids = memory.to_dict()["evidence"]["frame_ids"]
    if len(frame_ids) < 2:
        failures.append("D8 evidence covers fewer than two frames")
    if result.get("pending_observation_count") != len(pending):
        failures.append("pending observation count is inconsistent")
    if result.get("permanent_object_count") != len(memory.objects):
        failures.append("permanent object count is inconsistent")
    if result.get("association_decision_count") != len(memory.decisions):
        failures.append("association decision count is inconsistent")
    if result.get("frame_ids") != frame_ids:
        failures.append("result frame_ids differ from memory evidence")
    if result.get("round_trip_equal") is not True:
        failures.append("result does not certify equal round-trip")

    source = result.get("source", {})
    try:
        source_path = Path(str(source["cache_path"]))
        source_hash = sha256_file(source_path)
        if source_hash != source.get("cache_sha256"):
            failures.append("source D7 cache hash changed")
        if memory.metadata.get("source_cache") != str(source_path):
            failures.append("memory source cache path is inconsistent")
        if memory.metadata.get("source_cache_sha256") != source_hash:
            failures.append("memory source cache hash is inconsistent")
    except (KeyError, OSError, TypeError, ValueError) as error:
        failures.append(f"invalid source D7 cache: {error}")

    config = run_manifest.get("config", {})
    if (
        not isinstance(config, dict)
        or config.get("association_executed") is not False
        or config.get("object_memory_schema")
        != OBJECT_MEMORY_SCHEMA_VERSION
    ):
        failures.append("run manifest does not describe schema-only D8")

    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "scene_id": result.get("scene_id"),
        "query": result.get("query"),
        "pending_observations": len(pending),
        "permanent_objects": len(memory.objects),
        "association_decisions": len(memory.decisions),
        "frame_ids": frame_ids,
        "round_trip_equal": raw_memory == memory.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    args = parser.parse_args()
    report = validate_output(args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
