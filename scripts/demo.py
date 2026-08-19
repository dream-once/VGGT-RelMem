"""Run a synthetic, model-free relation-grounding smoke demo."""

from __future__ import annotations

import argparse
import json

import numpy as np

from relground.association import ObjectMemory
from relground.relations import RelationGrounder
from relground.schemas import GroundingQuery, ObjectObservation, OrientedBoundingBox


def _observation(obs_id: str, class_text: str, center: list[float]) -> ObjectObservation:
    center_array = np.asarray(center, dtype=np.float64)
    return ObjectObservation(
        obs_id=obs_id,
        class_text=class_text,
        frame_id=f"frame_{obs_id}",
        mask_ref=None,
        retrieval_score=0.92,
        sam_score=0.90,
        valid_point_ratio=0.95,
        points_ref=None,
        center=center_array,
        obb=OrientedBoundingBox(center_array, np.array([0.5, 0.8, 0.5])),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-memory", help="Optional output JSON path")
    args = parser.parse_args()

    memory = ObjectMemory()
    memory.add_many(
        [
            _observation("left_chair", "chair", [-1.0, 0.0, 0.0]),
            _observation("right_chair", "chair", [1.0, 0.0, 0.0]),
            _observation("desk", "desk", [0.0, 0.0, 0.0]),
        ]
    )
    if args.save_memory:
        memory.save(args.save_memory)

    query = GroundingQuery(
        query_id="demo_left_chair",
        target="chair",
        relation="left_of",
        reference="desk",
        anchor_frame="frame_0001",
    )
    grounder = RelationGrounder(memory, {"frame_0001": np.eye(4)})
    print(json.dumps(grounder.ground(query).to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
