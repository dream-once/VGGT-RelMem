"""Build the lightweight public Apartment/Cubicle benchmark summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relground.clio_final_summary import build_summary, validate_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--apartment-run", default="runs/clio-apartment-dev-v2-lc")
    parser.add_argument("--cubicle-run", default="runs/clio-cubicle-heldout-v1")
    parser.add_argument("--protocol", default="configs/clio_relation_confirmatory_protocol.json")
    parser.add_argument("--output", default="evidence/final-clio/benchmark_summary.json")
    parser.add_argument("--created-at")
    args = parser.parse_args()
    apartment = Path(args.apartment_run)
    cubicle = Path(args.cubicle_run)
    payload = build_summary(
        project_root=Path(args.project_root),
        apartment_grounding_path=apartment / "grounding_benchmark.json",
        apartment_association_path=apartment / "association_benchmark.json",
        apartment_relation_path=apartment / "relation-benchmark-v2/evaluation.json",
        cubicle_grounding_path=cubicle / "grounding_benchmark.json",
        cubicle_association_path=cubicle / "association_benchmark.json",
        cubicle_relation_path=cubicle / "relation-benchmark-v2/evaluation.json",
        protocol_path=Path(args.protocol),
        created_at=args.created_at,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validation = validate_summary(payload)
    (output.parent / "validation_report.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    raise SystemExit(0 if validation["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
