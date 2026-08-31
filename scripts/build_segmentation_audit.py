"""Build a label-free D21.1 inventory and a separate visibility template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from relground.segmentation_audit import (
    build_segmentation_inventory,
    build_visibility_template,
    validate_segmentation_inventory,
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_contact_sheet(
    inventory: dict[str, Any],
    *,
    project_root: Path,
    output_path: Path,
) -> None:
    from PIL import Image, ImageDraw

    frames = inventory["frames"]
    columns = 3
    panel_width = 500
    image_width = 238
    image_height = 180
    panel_height = 218
    rows = (len(frames) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * panel_width, rows * panel_height), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(frames):
        left = (index % columns) * panel_width
        top = (index // columns) * panel_height
        with Image.open(project_root / row["sam_input_ref"]) as source:
            source_image = source.convert("RGB").resize((image_width, image_height))
        with Image.open(project_root / row["preview_ref"]) as source:
            preview_image = source.convert("RGB").resize((image_width, image_height))
        sheet.paste(source_image, (left + 6, top + 30))
        sheet.paste(preview_image, (left + 256, top + 30))
        draw.text(
            (left + 8, top + 8),
            f"#{row['rank']:02d} {row['frame_id']}  PE={row['retrieval_score']:.3f}  masks={row['sam_instances']}",
            fill=(245, 245, 245),
        )
        draw.text((left + 8, top + 202), "SAM input", fill=(180, 180, 180))
        draw.text((left + 258, top + 202), "mask overlay", fill=(180, 180, 180))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    inventory = build_segmentation_inventory(
        project_root=project_root,
        d6_result_path=(project_root / args.d6_result),
        observations_path=(project_root / args.observations),
        scene_id=args.scene_id,
    )
    report = validate_segmentation_inventory(inventory, project_root=project_root)
    if report["status"] != "PASS":
        raise ValueError("segmentation inventory failed validation: " + "; ".join(report["failures"]))
    write_json(output_dir / "inventory.json", inventory)
    write_json(output_dir / "visibility_labels.template.json", build_visibility_template(inventory))
    write_json(output_dir / "validation.json", report)
    save_contact_sheet(
        inventory,
        project_root=project_root,
        output_path=output_dir / "contact_sheet.png",
    )
    return {
        "status": "PASS_WITH_FRAME_VISIBILITY_PENDING",
        "inventory": str(output_dir / "inventory.json"),
        "visibility_template": str(output_dir / "visibility_labels.template.json"),
        "contact_sheet": str(output_dir / "contact_sheet.png"),
        "counts": inventory["counts"],
    }


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--scene-id", default="clio-apartment-pillow-k8")
    parser.add_argument("--d6-result", default="runs/clio-apartment-gpu/d6-pillow-all/d6_result.json")
    parser.add_argument("--observations", default="runs/clio-apartment-gpu/d6-pillow-all/observations.json")
    parser.add_argument("--output-dir", default="runs/clio-apartment-gpu/d21_1-pillow-audit")
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
