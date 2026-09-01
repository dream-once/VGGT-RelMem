"""Validate tracked Clio query manifests and optional official task sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_tracked_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Check invariants that a tracked-files-only public clone can prove."""

    failures: list[str] = []
    try:
        source = payload["source"]
        split = payload["split"]
        records = payload["queries"]
        if not isinstance(records, list) or not records:
            raise ValueError("query manifest must contain records")
        tasks = [str(item["task"]) for item in records]
        if len(tasks) != len(set(tasks)):
            raise ValueError("query manifest contains duplicate tasks")
        if not all(str(item["sam_query"]).strip() for item in records):
            raise ValueError("every query must contain a non-empty SAM prompt")
        if "calibration_count" in split or "development_count" in split:
            calibration_count = int(split["calibration_count"])
            development_count = int(split["development_count"])
            actual_calibration = {
                str(item["task"])
                for item in records
                if item["split"] == "calibration"
            }
            actual_development = sum(
                item["split"] == "development" for item in records
            )
            if len(actual_calibration) != calibration_count:
                raise ValueError("calibration split count changed")
            if actual_development != development_count:
                raise ValueError("development split count changed")
            if len(records) != calibration_count + development_count:
                raise ValueError("split counts do not cover the tracked query records")
            scene_id = str(payload.get("scene_id", "apartment"))
            expected_calibration = set(sorted(
                tasks,
                key=lambda query: hashlib.sha256(
                    f"clio-{scene_id}-v1|{query}".encode()
                ).hexdigest(),
            )[:calibration_count])
            if actual_calibration != expected_calibration:
                raise ValueError("calibration hash split changed")
        elif "held_out_count" in split:
            held_out_count = int(split["held_out_count"])
            actual_held_out = sum(item["split"] == "held-out" for item in records)
            if actual_held_out != held_out_count:
                raise ValueError("held-out split count changed")
            if len(records) != held_out_count:
                raise ValueError("held-out count does not cover the tracked query records")
        else:
            raise ValueError("query manifest uses an unsupported split contract")
        task_yaml = Path(str(source["task_yaml"]))
        if task_yaml.is_absolute() or ".." in task_yaml.parts:
            raise ValueError("task YAML reference must be a safe relative path")
        digest = str(source["task_yaml_sha256"])
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("task YAML digest must be lowercase SHA-256")
    except (KeyError, TypeError, ValueError) as error:
        failures.append(str(error))
    return {
        "status": "PASS" if not failures else "FAIL",
        "scope": "tracked_manifest_invariants",
        "failures": failures,
    }


def validate_official_task_source(
    payload: Mapping[str, Any], task_yaml_path: Path
) -> dict[str, Any]:
    """Check the tracked records against an explicitly supplied Clio YAML."""

    failures = list(validate_tracked_manifest(payload)["failures"])
    try:
        if _sha256(task_yaml_path) != payload["source"]["task_yaml_sha256"]:
            raise ValueError("official task YAML SHA-256 differs from the manifest")
        official_payload = yaml.safe_load(task_yaml_path.read_text(encoding="utf-8"))
        if not isinstance(official_payload, Mapping):
            raise ValueError("official task YAML must contain a task mapping")
        official = {str(task) for task in official_payload}
        tracked = {str(item["task"]) for item in payload["queries"]}
        if tracked != official:
            raise ValueError("tracked query records do not exactly cover official tasks")
    except (KeyError, TypeError, ValueError, OSError, yaml.YAMLError) as error:
        failures.append(str(error))
    return {
        "status": "PASS" if not failures else "FAIL",
        "scope": "official_task_source_integration",
        "task_yaml": str(task_yaml_path),
        "failures": failures,
    }


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
