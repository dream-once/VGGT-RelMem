"""Run D17 relation prediction without reading labels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json

from relground.association import ObjectMemory
from relground.relation_protocol import run_relation_prediction


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _bundle_input(path: str, output_parent: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved.parent != output_parent.resolve():
        raise ValueError("D17 formal bundle inputs must share the output directory")
    return resolved


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    memory_path = _bundle_input(args.memory, output.parent)
    anchor_path = _bundle_input(args.anchor_poses, output.parent)
    query_path = _bundle_input(args.queries, output.parent)
    calibration_path = _bundle_input(args.calibration, output.parent)
    query_payload = json.loads(query_path.read_text(encoding="utf-8"))
    calibration_payload = json.loads(
        calibration_path.read_text(encoding="utf-8")
    )
    anchors = json.loads(anchor_path.read_text(encoding="utf-8"))
    source = {
        "object_memory": memory_path.name,
        "object_memory_sha256": _sha(memory_path),
        "anchor_poses": anchor_path.name,
        "anchor_poses_sha256": _sha(anchor_path),
        "queries": query_path.name,
        "queries_sha256": _sha(query_path),
        "calibration": calibration_path.name,
        "calibration_sha256": _sha(calibration_path),
    }
    result = run_relation_prediction(
        ObjectMemory.load(memory_path),
        anchors,
        query_payload,
        calibration_payload,
        source=source,
        created_at=args.created_at or datetime.now(timezone.utc).isoformat(),
    )
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", required=True)
    parser.add_argument("--anchor-poses", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--created-at")
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
