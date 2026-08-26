"""Deterministic complete D14 fixture for CPU replay acceptance."""

from __future__ import annotations

from typing import Any

import numpy as np

from .candidate_cache import build_d11_payloads
from .d9_association import ManualInstanceLabels
from .schemas import ObjectObservation, OrientedBoundingBox


def build_synthetic_complete_fixture(
    *,
    created_at: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    frame_ids = [
        "frame_0001",
        "frame_0003",
        "frame_0005",
        "frame_0007",
        "frame_0009",
        "frame_0011",
    ]
    retrieval_rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    processed: list[dict[str, Any]] = []
    for index, frame_id in enumerate(frame_ids):
        center = np.asarray([index * 0.2, 0.0, 1.0], dtype=float)
        retrieval_rows.append({
            "rank": index + 1,
            "frame_id": frame_id,
            "geometry_index": index * 2,
            "image_path": f"images/{frame_id}.jpg",
            "retrieval_score": 0.95 - index * 0.08,
            "retrieval_cosine": 0.90 - index * 0.08,
            "camera_center": [index * 0.2, 0.0, 0.0],
            "view_direction": [0.0, 0.0, 1.0],
        })
        observations.append(ObjectObservation(
            obs_id=f"synthetic-{frame_id}",
            class_text="trash can",
            frame_id=frame_id,
            mask_ref=None,
            retrieval_score=0.95 - index * 0.08,
            sam_score=0.9,
            valid_point_ratio=0.9,
            points_ref=None,
            center=center,
            obb=OrientedBoundingBox(center=center, extent=np.full(3, 0.2)),
        ).to_dict())
        processed.append({
            "frame_id": frame_id,
            "sam_instances": 1,
            "lifted_instances": 1,
            "rejected_instances": 0,
        })
    retrieval = {
        "query": "trash can",
        "backend": "synthetic-pe",
        "retrieval_config": {"strategy": "hybrid"},
        "raw_ranking": retrieval_rows,
        "source_commits": {"perception_models": "synthetic-pe-revision"},
    }
    d6_result = {
        "query": "trash can",
        "sam_threshold": 0.5,
        "lifter_config": {"min_points": 30},
        "mask_resizing_after_sam": False,
        "processed_frames": processed,
        "rejected_instances": [],
        "source_commits": {"sam3": "synthetic-sam3-revision"},
    }
    _, cache = build_d11_payloads(
        retrieval=retrieval,
        d6_result=d6_result,
        observations_payload={
            "query": "trash can",
            "observations": observations,
        },
        scene_id="synthetic-d14-scene",
        query_id="synthetic-d14-trash-can",
        image_refs={frame_id: f"images/{frame_id}.jpg" for frame_id in frame_ids},
        image_hashes={frame_id: "a" * 64 for frame_id in frame_ids},
        source_hashes={
            "d5_retrieval_sha256": "b" * 64,
            "d6_result_sha256": "c" * 64,
            "d7_observations_sha256": "d" * 64,
        },
        artifact_refs={
            "visual_memory_manifest": "visual_memory_manifest.json",
            "d5_retrieval": "source_d5_retrieval.json",
            "d6_result": "source_d6_result.json",
            "d7_observations": "source_d7_observations.json",
        },
        created_at=created_at,
    )
    labels = {
        "schema_version": "0.1",
        "scene_id": "synthetic-d14-scene",
        "query": "trash can",
        "annotation_method": "deterministic synthetic fixture",
        "notes": ["Labels are evaluator-only and never enter policy selection."],
        "instance_groups": [
            {
                "instance_id": "synthetic-instance-a",
                "observation_ids": [
                    f"synthetic-{frame_id}" for frame_id in frame_ids[:3]
                ],
            },
            {
                "instance_id": "synthetic-instance-b",
                "observation_ids": [
                    f"synthetic-{frame_id}" for frame_id in frame_ids[3:]
                ],
            },
        ],
    }
    ManualInstanceLabels.from_dict(labels)
    return cache, labels
