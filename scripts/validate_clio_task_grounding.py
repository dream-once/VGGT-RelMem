"""Validate deterministic replay of a Clio task-grounding evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relground.clio_task_evaluation import validate_clio_task_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.evaluation.read_text(encoding="utf-8"))
    report = validate_clio_task_evaluation(payload, project_root=Path.cwd())
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
