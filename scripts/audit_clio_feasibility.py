"""Create the no-download D16 Clio feasibility report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json

from relground.clio_protocol import audit_clio_filesystem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/clio_dataset_manifest.json")
    parser.add_argument("--splits", default="configs/clio_splits.json")
    parser.add_argument(
        "--filesystem", default="/root/autodl-tmp",
        help="Filesystem whose free bytes are audited; no files are written there.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--checked-at")
    return parser


def run(args: argparse.Namespace) -> dict:
    dataset = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    checked_at = args.checked_at or datetime.now(timezone.utc).isoformat()
    report = audit_clio_filesystem(
        dataset, splits, filesystem_path=args.filesystem, checked_at=checked_at,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    report = run(build_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
