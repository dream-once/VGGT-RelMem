"""Validate a D21.1 label-free segmentation inventory and optional labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relground.segmentation_audit import (
    evaluate_visibility,
    read_json_object,
    validate_segmentation_inventory,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--visibility-labels")
    args = parser.parse_args()
    inventory = read_json_object(Path(args.inventory))
    report = validate_segmentation_inventory(inventory, project_root=Path(args.project_root))
    if args.visibility_labels:
        report["visibility_evaluation"] = evaluate_visibility(
            inventory,
            read_json_object(Path(args.visibility_labels)),
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
