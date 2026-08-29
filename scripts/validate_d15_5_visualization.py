"""Independently validate D15.5 scene-memory visualization artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


MANIFEST_SCHEMA = "d15.5-scene-memory-visualization/0.1"
AUDIT_SCHEMA = "0.1"
STRICT_THRESHOLDS = {
    "strict_min_angle_deg": 15.0,
    "strict_min_baseline_depth_ratio": 0.2,
    "strict_min_distinct_frames": 3,
    "strict_min_qualifying_pairs": 2,
    "strict_min_covered_frames": 3,
}
DIAGNOSTIC_ANGLE_DEG = 8.0
DIAGNOSTIC_BASELINE_DEPTH_RATIO = 0.1
MIN_VIDEO_SECONDS = 8.0
MIN_VIDEO_MOTION_RATIO = 0.20
MOTION_DIFFERENCE_THRESHOLD = 0.65
REQUIRED_BASENAMES = {
    "viewpoint_audit.json",
    "overview.png",
    "object_parallax.mp4",
    "scene_memory.ply",
}


def _safe_path(root: Path, reference: str) -> Path:
    relative = Path(reference)
    if not reference or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"artifact escapes output directory: {reference!r}")
    root = root.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"artifact escapes output directory: {reference!r}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonfinite(value: Any, prefix: str = "$") -> list[str]:
    result: list[str] = []
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return result
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            result.append(prefix)
    elif isinstance(value, Mapping):
        for key, child in value.items():
            result.extend(_nonfinite(child, f"{prefix}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            result.extend(_nonfinite(child, f"{prefix}[{index}]"))
    return result


def _artifact_records(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = manifest.get("artifacts", {})
    records: list[dict[str, Any]] = []
    if isinstance(raw, Mapping):
        for name, value in raw.items():
            if isinstance(value, Mapping):
                record = dict(value)
                record.setdefault("name", str(name))
                records.append(record)
    elif isinstance(raw, list):
        records.extend(dict(value) for value in raw if isinstance(value, Mapping))
    return records


def _is_available(record: Mapping[str, Any]) -> bool:
    if "available" in record:
        return record.get("available") is True
    return str(record.get("status", "available")).lower() in {
        "available",
        "pass",
        "ready",
    }


def _record_size(record: Mapping[str, Any]) -> Any:
    return record.get("bytes", record.get("size_bytes"))


def _probe_overview(path: Path) -> dict[str, int]:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("OpenCV is required to decode overview.png") from error
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0 or image.ndim not in (2, 3):
        raise ValueError("overview.png is not decodable")
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("overview.png has invalid dimensions")
    return {"width": int(width), "height": int(height)}


def _probe_video(path: Path) -> dict[str, float | int]:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("OpenCV is required to probe object_parallax.mp4") from error
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("object_parallax.mp4 cannot be opened")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError("object_parallax.mp4 has invalid FPS")
        stride = max(1, int(round(fps / 2.0)))
        decoded = 0
        previous: np.ndarray | None = None
        differences: list[float] = []
        while True:
            readable, frame = capture.read()
            if not readable:
                break
            if decoded % stride == 0:
                gray = cv2.cvtColor(
                    cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2GRAY,
                ).astype(np.float32)
                if previous is not None:
                    differences.append(float(np.mean(np.abs(gray - previous))))
                previous = gray
            decoded += 1
        if decoded < 2 or not differences:
            raise ValueError("object_parallax.mp4 has too few decodable frames")
        motion_ratio = float(
            np.mean(np.asarray(differences) >= MOTION_DIFFERENCE_THRESHOLD)
        )
        return {
            "fps": fps,
            "frame_count": decoded,
            "duration_seconds": decoded / fps,
            "motion_ratio": motion_ratio,
        }
    finally:
        capture.release()


def _probe_ply(path: Path) -> dict[str, int]:
    raw = path.read_bytes()
    if not raw.startswith(b"ply\n") and not raw.startswith(b"ply\r\n"):
        raise ValueError("scene_memory.ply has no PLY magic header")
    marker = b"end_header\n"
    index = raw.find(marker)
    if index < 0:
        marker = b"end_header\r\n"
        index = raw.find(marker)
    if index < 0 or index + len(marker) >= len(raw):
        raise ValueError("scene_memory.ply has an empty or incomplete header/body")
    header = raw[: index + len(marker)].decode("ascii", errors="strict")
    vertex_lines = [
        line for line in header.splitlines() if line.startswith("element vertex ")
    ]
    if len(vertex_lines) != 1:
        raise ValueError("scene_memory.ply must declare one vertex element")
    vertices = int(vertex_lines[0].split()[-1])
    if vertices <= 0:
        raise ValueError("scene_memory.ply declares no vertices")
    return {"vertices": vertices}


def _thresholds_are_frozen(audit: Mapping[str, Any]) -> bool:
    raw = audit.get("thresholds", {})
    strict = raw.get("strict", raw) if isinstance(raw, Mapping) else {}
    if not isinstance(strict, Mapping):
        return False
    for key, expected in STRICT_THRESHOLDS.items():
        actual = strict.get(key)
        if isinstance(expected, int):
            if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
                return False
        else:
            try:
                if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
                    return False
            except (TypeError, ValueError):
                return False
    return True


def _frames(record: Mapping[str, Any]) -> list[str]:
    evidence = record.get("frame_evidence")
    if isinstance(evidence, list):
        values = [
            str(item.get("frame_id", ""))
            for item in evidence
            if isinstance(item, Mapping)
        ]
        return sorted({value for value in values if value})
    raw = record.get(
        "observation_frame_ids",
        record.get("frame_ids", record.get("distinct_frame_ids", [])),
    )
    if not isinstance(raw, list):
        return []
    return sorted({str(value) for value in raw if str(value)})


def _metric(pair: Mapping[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        if name in pair:
            try:
                value = float(pair[name])
            except (TypeError, ValueError):
                return None
            return value if math.isfinite(value) else None
    return None


def _recompute_evidence(audit: Mapping[str, Any]) -> dict[str, Any]:
    objects = audit.get("objects", [])
    if not isinstance(objects, list):
        objects = []
    decisions: list[dict[str, Any]] = []
    for record in sorted(
        (item for item in objects if isinstance(item, Mapping)),
        key=lambda item: str(item.get("object_id", "")),
    ):
        frames = _frames(record)
        strict_pairs: list[list[str]] = []
        diagnostic_pairs: list[list[str]] = []
        seen: set[tuple[str, str]] = set()
        pairs = record.get("pair_metrics", record.get("pairs", record.get("view_pairs", [])))
        if not isinstance(pairs, list):
            pairs = []
        for pair in pairs:
            if not isinstance(pair, Mapping):
                continue
            frame_a = str(pair.get("frame_a", pair.get("frame_id_a", "")))
            frame_b = str(pair.get("frame_b", pair.get("frame_id_b", "")))
            identity = tuple(sorted((frame_a, frame_b)))
            if not all(identity) or identity[0] == identity[1] or identity in seen:
                continue
            seen.add(identity)
            angle = _metric(pair, ("angle_deg", "ray_angle_deg", "object_ray_angle_deg"))
            ratio = _metric(
                pair,
                ("baseline_depth_ratio", "baseline_to_depth_ratio"),
            )
            if angle is None or ratio is None:
                continue
            # The two gates must be satisfied by this same unordered pair.
            if angle >= 15.0 and ratio >= 0.2:
                strict_pairs.append(list(identity))
            if angle >= DIAGNOSTIC_ANGLE_DEG and ratio >= DIAGNOSTIC_BASELINE_DEPTH_RATIO:
                diagnostic_pairs.append(list(identity))
        strict_pairs.sort()
        diagnostic_pairs.sort()
        covered = sorted({frame for pair in strict_pairs for frame in pair})
        strong = len(frames) >= 3 and len(strict_pairs) >= 2 and len(covered) >= 3
        status = (
            "STRICT_MULTIVIEW"
            if strong
            else "DIAGNOSTIC_PARALLAX"
            if diagnostic_pairs
            else "WEAK_OR_SINGLE_VIEW"
        )
        decisions.append(
            {
                "object_id": str(record.get("object_id", "")),
                "status": status,
                "distinct_frame_count": len(frames),
                "strict_qualifying_pair_count": len(strict_pairs),
                "strict_covered_frames": covered,
                "strict_pairs": strict_pairs,
            }
        )
    counter = Counter(item["status"] for item in decisions)
    counts = {
        status: int(counter.get(status, 0))
        for status in (
            "STRICT_MULTIVIEW",
            "DIAGNOSTIC_PARALLAX",
            "WEAK_OR_SINGLE_VIEW",
        )
    }
    if counts.get("STRICT_MULTIVIEW", 0):
        overall = "STRONG_OBJECT_CENTRIC_MULTIVIEW"
    elif counts.get("DIAGNOSTIC_PARALLAX", 0):
        overall = "DIAGNOSTIC_OBJECT_CENTRIC_PARALLAX"
    else:
        overall = "WEAK_OBJECT_CENTRIC_EVIDENCE"
    return {"evidence_status": overall, "status_counts": counts, "objects": decisions}


def _counts(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def validate_output(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    errors: list[str] = []
    manifest_path = root / "visualization_manifest.json"
    if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
        return {
            "status": "FAIL",
            "artifact_status": "FAIL",
            "viewpoint_evidence_status": "NOT_EVALUATED",
            "errors": ["missing or empty visualization_manifest.json"],
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "status": "FAIL",
            "artifact_status": "FAIL",
            "viewpoint_evidence_status": "NOT_EVALUATED",
            "errors": [f"invalid visualization manifest: {error}"],
        }
    if not isinstance(manifest, Mapping):
        errors.append("visualization manifest must be an object")
        manifest = {}
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append("visualization manifest schema is not frozen D15.5")
    if manifest.get("artifact_status") != "PASS":
        errors.append("visualization manifest artifact_status is not PASS")
    invalid = _nonfinite(manifest)
    if invalid:
        errors.append("manifest contains non-finite values: " + ", ".join(invalid))

    records = _artifact_records(manifest)
    available = [record for record in records if _is_available(record)]
    available_paths: dict[str, Path] = {}
    for record in available:
        reference = str(record.get("path", ""))
        try:
            path = _safe_path(root, reference)
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"missing or empty available artifact: {reference}")
                continue
            expected_bytes = _record_size(record)
            if isinstance(expected_bytes, bool) or int(expected_bytes) != path.stat().st_size:
                errors.append(f"byte-size mismatch: {reference}")
            if str(record.get("sha256", "")) != _sha256(path):
                errors.append(f"SHA-256 mismatch: {reference}")
            available_paths[Path(reference).name] = path
        except (OSError, TypeError, ValueError) as error:
            errors.append(f"invalid available artifact {reference!r}: {error}")
    missing = REQUIRED_BASENAMES - set(available_paths)
    if missing:
        errors.append("missing required available artifacts: " + ", ".join(sorted(missing)))

    audit: Mapping[str, Any] = {}
    audit_path = available_paths.get("viewpoint_audit.json")
    if audit_path:
        try:
            loaded = json.loads(audit_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, Mapping):
                raise ValueError("audit must be a JSON object")
            audit = loaded
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"invalid viewpoint audit: {error}")
    if audit:
        if audit.get("schema_version") != AUDIT_SCHEMA:
            errors.append("viewpoint audit schema is not 0.1")
        if audit.get("artifact_status") != "PASS":
            errors.append("viewpoint audit artifact_status is not PASS")
        invalid = _nonfinite(audit)
        if invalid:
            errors.append("audit contains non-finite values: " + ", ".join(invalid))
        if not _thresholds_are_frozen(audit):
            errors.append("strict viewpoint thresholds differ from frozen D15.5")
        for field in ("scene_id", "query"):
            if audit.get(field) != manifest.get(field):
                errors.append(f"{field} differs between manifest and audit")
        manifest_counts = _counts(manifest.get("counts"))
        audit_counts = _counts(audit.get("counts"))
        for key in set(manifest_counts) & set(audit_counts):
            if manifest_counts[key] != audit_counts[key]:
                errors.append(f"count mismatch for {key}")
        raw_objects = audit.get("objects", [])
        if isinstance(raw_objects, list):
            object_count = audit_counts.get("objects", audit_counts.get("object_count"))
            if object_count is not None and object_count != len(raw_objects):
                errors.append("audit object count differs from objects list")

    recomputed = _recompute_evidence(audit)
    if audit:
        declared = audit.get("evidence_status")
        if declared != recomputed["evidence_status"]:
            errors.append("audit evidence_status differs from recomputed evidence")
        declared_counts = audit.get("status_counts")
        if declared_counts is not None and declared_counts != recomputed["status_counts"]:
            errors.append("audit status_counts differ from recomputed status_counts")

    overview_probe = video_probe = ply_probe = None
    try:
        overview_probe = _probe_overview(available_paths["overview.png"])
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        errors.append(f"invalid overview.png: {error}")
    try:
        video_probe = _probe_video(available_paths["object_parallax.mp4"])
        if float(video_probe["duration_seconds"]) < MIN_VIDEO_SECONDS:
            errors.append("object_parallax.mp4 is shorter than 8 seconds")
        if float(video_probe["motion_ratio"]) < MIN_VIDEO_MOTION_RATIO:
            errors.append("object_parallax.mp4 does not contain enough motion")
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        errors.append(f"invalid object_parallax.mp4: {error}")
    try:
        ply_probe = _probe_ply(available_paths["scene_memory.ply"])
    except (KeyError, OSError, TypeError, UnicodeDecodeError, ValueError) as error:
        errors.append(f"invalid scene_memory.ply: {error}")

    artifact_status = "PASS" if not errors else "FAIL"
    evidence_status = (
        recomputed["evidence_status"] if artifact_status == "PASS" else "NOT_EVALUATED"
    )
    if artifact_status == "FAIL":
        status = "FAIL"
    elif evidence_status == "STRONG_OBJECT_CENTRIC_MULTIVIEW":
        status = "PASS"
    else:
        status = "PASS_WITH_WEAK_VIEWPOINT_EVIDENCE"
    return {
        "status": status,
        "artifact_status": artifact_status,
        "viewpoint_evidence_status": evidence_status,
        "errors": errors,
        "scene_id": manifest.get("scene_id"),
        "query": manifest.get("query"),
        "status_counts": recomputed["status_counts"],
        "objects": recomputed["objects"],
        "overview_probe": overview_probe,
        "video_probe": video_probe,
        "ply_probe": ply_probe,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    args = parser.parse_args()
    result = validate_output(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["status"] != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
