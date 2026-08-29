"""Run the frozen D18 Q x A replay without model inference."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from relground.d9_association import ManualInstanceLabels
from relground.experiment_protocol import (
    evaluate_synthetic_prediction,
    load_json,
    run_experiment_prediction,
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
    manifest_path = (root / args.manifest).resolve()
    manifest = load_json(manifest_path)
    validate_experiment_manifest(manifest, project_root=root)
    source = source_by_id(manifest, args.source_id)
    cache = load_json(root / source["cache_ref"])
    created_at = args.created_at or datetime.now(timezone.utc).isoformat()
    output = Path(args.output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    prediction = run_experiment_prediction(
        manifest,
        cache,
        manifest_ref=Path(args.manifest).as_posix(),
        manifest_sha256=sha256_file(manifest_path),
        source_id=args.source_id,
        created_at=created_at,
    )
    prediction_path = output / "prediction.json"
    write_json(prediction_path, prediction)

    evaluation = None
    if source["labels_ref"] is not None:
        labels = ManualInstanceLabels.load(root / source["labels_ref"])
        evaluation = evaluate_synthetic_prediction(
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
        "held_out_status": prediction["held_out"]["status"],
    }
    write_json(output / "run_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument(
        "--manifest", default="configs/d18_experiment_manifest.json"
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
