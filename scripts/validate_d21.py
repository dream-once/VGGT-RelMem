"""Validate the D21 result card, claim audit, and source references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from relground.result_card import (
    D21_SCHEMA_VERSION,
    D21_STATUS,
    FINAL_CONCLUSION,
    LIMITED_PERFORMANCE_CLAIM,
    PROJECT_POSITIONING,
    REQUIRED_GAPS,
    RESULT_IDS,
    load_json,
    render_result_card,
    sha256_file,
)
from scripts.build_d21_result_card import execute
from scripts.validate_d20 import run as validate_d20


def _references_match(root: Path, card: dict[str, object]) -> bool:
    for row in card["results"]:
        for key in ("evidence", "config"):
            reference = row[key]
            path = Path(reference["path"])
            if path.is_absolute() or ".." in path.parts:
                return False
            source = root / path
            if not source.is_file():
                return False
            if sha256_file(source) != reference["sha256"]:
                return False
    return True


def run(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.project_root).resolve()
    retained = (root / args.retained_root).resolve()
    with tempfile.TemporaryDirectory(prefix="vggt-relmem-d21-") as directory:
        rebuilt = Path(directory) / "moved" / "final"
        products = execute(
            project_root=root,
            manifest_path=(root / args.manifest).resolve(),
            output_dir=rebuilt,
        )
        card = products["card"]
        audit = products["audit"]
        retained_card = load_json(retained / "result_card.json")
        retained_audit = load_json(retained / "claim_audit.json")
        d20 = validate_d20(
            argparse.Namespace(
                project_root=str(root),
                manifest="configs/d20_reproduction_manifest.json",
                retained_root="evidence/week3/d20-reproduction",
                output=None,
            )
        )
        checks = {
            "result_card_matches": card == retained_card,
            "result_card_markdown_matches": (
                render_result_card(card)
                == (retained / "result_card.md").read_text(
                    encoding="utf-8"
                )
            ),
            "claim_audit_matches": audit == retained_audit,
            "claim_audit_pass": (
                audit["status"] == "PASS"
                and audit["review_required_count"] == 0
            ),
            "positioning_frozen": (
                card["project_positioning"] == PROJECT_POSITIONING
                and audit["required_positioning_present"]
            ),
            "conclusion_frozen": (
                card["final_conclusion"] == FINAL_CONCLUSION
                and audit["required_conclusion_present"]
            ),
            "result_inventory_complete": (
                len(card["results"]) == len(RESULT_IDS)
                and all(
                    row["sample_size"]
                    and row["budget"]
                    and row["validation_status"] == "PASS"
                    for row in card["results"]
                )
            ),
            "source_references_match": _references_match(root, card),
            "external_gaps_explicit": card["gaps"] == REQUIRED_GAPS,
            "claim_boundary_honest": (
                not card["claim_boundary"]["official_found_it_reproduction"]
                and not card["claim_boundary"]["closed_loop_navigation"]
                and not card["claim_boundary"]["held_out_performance"]
                and not card["claim_boundary"]["sota_or_superiority_claim"]
                and card["claim_boundary"]["performance_improvement"]
                == LIMITED_PERFORMANCE_CLAIM
            ),
            "d20_reproduction_still_passes": d20["status"] == "PASS",
        }
    status = "PASS" if all(checks.values()) else "FAIL"
    validation = {
        "schema_version": D21_SCHEMA_VERSION,
        "status": status,
        "stage": "D21-validation",
        "checks": checks,
        "source_status": D21_STATUS,
        "external_gaps": REQUIRED_GAPS,
    }
    if args.output:
        Path(args.output).write_text(
            json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return validation


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument(
        "--manifest", default="configs/d21_result_card_manifest.json"
    )
    parser.add_argument(
        "--retained-root", default="evidence/week3/d21-final"
    )
    parser.add_argument("--output")
    return parser


if __name__ == "__main__":
    raise SystemExit(
        0 if run(build_parser().parse_args())["status"] == "PASS" else 2
    )
