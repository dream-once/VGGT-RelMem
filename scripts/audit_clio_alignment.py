"""Build and validate Clio apartment RGB/pose alignment readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relground.clio_alignment_audit import (
    build_alignment_readiness,
    validate_alignment_readiness,
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--database", default="data/clio/apartment/colmap/database.db")
    parser.add_argument("--rgb-root", default="data/clio/apartment/images")
    parser.add_argument("--sparse-root", default="data/clio/apartment/sparse/0")
    parser.add_argument("--rosbag", default="data/clio/apartment/apartment.bag")
    parser.add_argument("--output", default="runs/clio-apartment-gpu/d21_1-pillow-audit/alignment_readiness.json")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    payload = build_alignment_readiness(
        project_root=project_root,
        database_path=project_root / args.database,
        rgb_root=project_root / args.rgb_root,
        sparse_root=project_root / args.sparse_root,
        rosbag_path=project_root / args.rosbag,
    )
    report = validate_alignment_readiness(payload, project_root=project_root)
    if report["status"] != "PASS":
        raise ValueError("alignment readiness validation failed")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit": payload, "validation": report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
