"""Build the tracked lightweight post-D21 PE fusion summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relground.pe_semantic_fusion_summary import (
    build_summary,
    validate_summary,
)


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument(
        "--config",
        default="configs/clio_pe_semantic_fusion.json",
    )
    parser.add_argument(
        "--apartment-prediction",
        default=(
            "runs/clio-pe-semantic-fusion-v1/apartment/"
            "prediction.json"
        ),
    )
    parser.add_argument(
        "--apartment-evaluation",
        default=(
            "runs/clio-pe-semantic-fusion-v1/apartment/"
            "evaluation.json"
        ),
    )
    parser.add_argument(
        "--cubicle-prediction",
        default=(
            "runs/clio-pe-semantic-fusion-v1/cubicle/"
            "prediction.json"
        ),
    )
    parser.add_argument(
        "--cubicle-evaluation",
        default=(
            "runs/clio-pe-semantic-fusion-v1/cubicle/"
            "evaluation.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "evidence/post-d21-pe-fusion/"
            "benchmark_summary.json"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.project_root).resolve()

    def resolve(value: str) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    payload = build_summary(
        project_root=root,
        config_path=resolve(args.config),
        apartment_prediction_path=resolve(args.apartment_prediction),
        apartment_evaluation_path=resolve(args.apartment_evaluation),
        cubicle_prediction_path=resolve(args.cubicle_prediction),
        cubicle_evaluation_path=resolve(args.cubicle_evaluation),
    )
    report = validate_summary(payload, project_root=root)
    if report["status"] != "PASS":
        raise SystemExit(json.dumps(report, ensure_ascii=False, indent=2))
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"],
        "stage": payload["stage"],
        "output": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
