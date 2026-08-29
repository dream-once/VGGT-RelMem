"""Build the D21 final result card and README claim audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from relground.result_card import (
    audit_readme_claims,
    build_result_card,
    load_json,
    render_result_card,
    validate_result_card_manifest,
)


def execute(
    *,
    project_root: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    manifest_file = Path(manifest_path).resolve()
    output = Path(output_dir).resolve()
    manifest = load_json(manifest_file)
    validate_result_card_manifest(manifest, project_root=root)
    readme = root / next(
        item["path"]
        for item in manifest["inputs"]
        if item["input_id"] == "readme"
    )
    card = build_result_card(root, manifest)
    audit = audit_readme_claims(readme.read_text(encoding="utf-8"))
    if audit["status"] != "PASS":
        raise ValueError("README claim audit requires review")
    output.mkdir(parents=True, exist_ok=True)
    (output / "result_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "result_card.md").write_text(
        render_result_card(card),
        encoding="utf-8",
    )
    (output / "claim_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"card": card, "audit": audit}


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument(
        "--manifest",
        default=str(root / "configs/d21_result_card_manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(root / "evidence/week3/d21-final"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = execute(
        project_root=args.project_root,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "status": "PASS",
        "stage": "D21-final-result-card",
        "result_count": len(result["card"]["results"]),
        "claim_occurrences": result["audit"]["occurrence_count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
