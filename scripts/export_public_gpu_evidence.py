"""Export portable lightweight evidence from local GPU/D15.5 products."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relground.public_evidence import export_public_bundle


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument(
        "--gpu-bundle",
        default="runs/gpu-acceptance-20260829",
    )
    parser.add_argument(
        "--visualization-dir",
        default=(
            "runs/office-loop-d15_5-s5/"
            "d15_5-trash-can-k24"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="evidence/week3/d15-gpu-public",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.project_root).resolve()
    manifest = export_public_bundle(
        project_root=root,
        gpu_bundle=root / args.gpu_bundle,
        visualization_dir=root / args.visualization_dir,
        output_dir=root / args.output_dir,
    )
    print(json.dumps({
        "status": manifest["status"],
        "stage": manifest["stage"],
        "artifact_count": len(manifest["artifacts"]),
        "coverage_aware": manifest["q2_semantics"]["coverage_aware"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
