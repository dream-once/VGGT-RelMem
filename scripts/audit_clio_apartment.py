"""Build a lightweight receipt for an extracted local Clio apartment scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relground.clio_acquisition import build_receipt


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--archive", required=True)
    parser.add_argument("--extracted", required=True)
    parser.add_argument("--apartment-folder-url", required=True)
    parser.add_argument("--checked-at", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt = build_receipt(
        project_root=args.project_root,
        archive_path=args.archive,
        extraction_path=args.extracted,
        apartment_folder_url=args.apartment_folder_url,
        checked_at=args.checked_at,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
