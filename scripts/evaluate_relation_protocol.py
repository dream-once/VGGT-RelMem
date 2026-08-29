"""Evaluate a frozen D17 prediction with a separate label file."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json

from relground.relation_protocol import evaluate_relation_prediction


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = Path(args.prediction).resolve()
    labels_path = Path(args.labels).resolve()
    if (
        prediction_path.parent != output.parent.resolve()
        or labels_path.parent != output.parent.resolve()
    ):
        raise ValueError("D17 evaluation inputs must share the output directory")
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    source = {
        "prediction": prediction_path.name,
        "prediction_sha256": _sha(prediction_path),
        "labels": labels_path.name,
        "labels_sha256": _sha(labels_path),
    }
    result = evaluate_relation_prediction(
        prediction,
        labels,
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
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--created-at")
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
