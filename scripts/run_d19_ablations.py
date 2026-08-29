"""Run D19 one-factor ablations from retained candidate caches."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from relground.ablation_protocol import (
    evaluate_synthetic_ablations,
    run_ablation_prediction,
    validate_ablation_manifest,
)
from relground.d9_association import ManualInstanceLabels
from relground.experiment_protocol import (
    load_json,
    sha256_file,
    source_by_id,
    validate_experiment_manifest,
)


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.project_root).resolve()
    manifest_path = root / args.manifest
    manifest = load_json(manifest_path)
    validate_ablation_manifest(manifest, project_root=root)
    d18 = load_json(root / manifest["d18_manifest"])
    validate_experiment_manifest(d18, project_root=root)
    source = source_by_id(d18, args.source_id)
    cache = load_json(root / source["cache_ref"])
    created_at = args.created_at or datetime.now(timezone.utc).isoformat()
    output = Path(args.output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    prediction = run_ablation_prediction(
        manifest,
        cache,
        source_id=args.source_id,
        cache_ref=source["cache_ref"],
        cache_sha256=source["cache_sha256"],
        manifest_ref=Path(args.manifest).as_posix(),
        manifest_sha256=sha256_file(manifest_path),
        created_at=created_at,
    )
    prediction_path = output / "prediction.json"
    write_json(prediction_path, prediction)

    evaluation = None
    if source["labels_ref"] is not None:
        labels = ManualInstanceLabels.load(root / source["labels_ref"])
        evaluation = evaluate_synthetic_ablations(
            prediction,
            cache,
            labels,
            prediction_ref="prediction.json",
            prediction_sha256=sha256_file(prediction_path),
            labels_ref=source["labels_ref"],
            labels_sha256=source["labels_sha256"],
            created_at=created_at,
        )
        write_json(output / "evaluation.json", evaluation)

    report = {
        "status": "PASS",
        "source_id": args.source_id,
        "prediction_status": prediction["status"],
        "evaluation_status": (
            None if evaluation is None else evaluation["status"]
        ),
        "real_ablation_status": "REAL_ABLATION_PENDING",
    }
    write_json(output / "run_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument(
        "--manifest", default="configs/d19_ablation_manifest.json"
    )
    parser.add_argument(
        "--source-id",
        choices=("office-loop-development", "synthetic-correctness"),
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--created-at")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
