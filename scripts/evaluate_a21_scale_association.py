"""Evaluate A2.1 predictions with development labels kept outside prediction."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from relground.a21_scale_association import (
    A21_ASSOCIATION_ID,
    A21_STATUS,
    evaluate_a21_predictions,
)
from relground.association import ObjectMemory
from relground.d9_association import ManualInstanceLabels
from relground.observation_cache import sha256_file


def project_reference(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(
            f"A2.1 evaluation source is outside project root: {path}"
        ) from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--project-root", default=str(Path(__file__).resolve().parents[1])
    )
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    prediction_dir = Path(args.prediction_dir).resolve()
    result_path = prediction_dir / "a21_result.json"
    result = json.loads(result_path.read_text())
    if result.get("association_id") != A21_ASSOCIATION_ID or result.get("status") != "PASS":
        raise ValueError("A2.1 prediction is not a passing compatible result")
    source = ObjectMemory.load(prediction_dir / result["artifacts"]["source_memory"])
    labels_path = Path(args.labels).resolve()
    labels = ManualInstanceLabels.load(labels_path)
    if (
        labels.scene_id != result.get("scene_id")
        or labels.query != result.get("query")
    ):
        raise ValueError("A2.1 labels differ from prediction scene/query")
    evaluation = evaluate_a21_predictions(
        list(source.pending_observations.values()), result["pairs"], labels
    )
    payload = {
        "schema_version": "0.1",
        "status": "PASS",
        "stage": "D21.1-A2.1-development-evaluation",
        "association_id": A21_ASSOCIATION_ID,
        "development_status": A21_STATUS,
        "scene_id": labels.scene_id,
        "query": labels.query,
        "source": {
            "prediction_result": project_reference(project_root, result_path),
            "prediction_result_sha256": sha256_file(result_path),
            "labels": project_reference(project_root, labels_path),
            "labels_sha256": sha256_file(labels_path),
        },
        "metrics": evaluation["metrics"],
        "pairs": evaluation["pairs"],
        "failure_cases": evaluation["failure_cases"],
        "claim_boundary": {
            "development_only": True,
            "cubicle_held_out_untouched": True,
            "formal_method_upgrade_allowed": False,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
