import unittest

import numpy as np

from relground.association import ObjectMemory
from relground.d9_association import (
    D9_PREDICTION_SCHEMA_VERSION,
    ManualInstanceGroup,
    ManualInstanceLabels,
    SpatialGateConfig,
    associate_pending,
    evaluate_predictions,
    predict_all_pairs,
    predict_pair,
)
from relground.schemas import ObjectObservation, OrientedBoundingBox


def make_observation(
    obs_id: str,
    frame_id: str,
    center: list[float],
    *,
    class_text: str = "trash can",
    extent: float = 0.2,
) -> ObjectObservation:
    center_array = np.asarray(center, dtype=float)
    return ObjectObservation(
        obs_id=obs_id,
        class_text=class_text,
        frame_id=frame_id,
        mask_ref=f"masks/{obs_id}.npy",
        retrieval_score=0.8,
        sam_score=0.9,
        valid_point_ratio=0.95,
        points_ref=f"points/{obs_id}.npz",
        center=center_array,
        obb=OrientedBoundingBox(
            center=center_array,
            extent=np.full(3, extent),
        ),
    )


def make_labels(
    groups: list[tuple[str, tuple[str, ...]]],
) -> ManualInstanceLabels:
    return ManualInstanceLabels(
        scene_id="scene",
        query="trash can",
        annotation_method="unit-test manual labels",
        notes=("synthetic",),
        instance_groups=tuple(
            ManualInstanceGroup(instance_id, observation_ids)
            for instance_id, observation_ids in groups
        ),
    )


class D9AssociationTests(unittest.TestCase):
    def test_only_cross_frame_component_becomes_permanent(self) -> None:
        observations = [
            make_observation("a", "frame_0001", [0.0, 0.0, 0.0]),
            make_observation("b", "frame_0002", [0.05, 0.0, 0.0]),
            make_observation("c", "frame_0001", [1.0, 0.0, 0.0]),
            make_observation("d", "frame_0001", [1.04, 0.0, 0.0]),
        ]
        labels = make_labels([
            ("instance_a", ("a", "b")),
            ("instance_b", ("c", "d")),
        ])
        memory = ObjectMemory(
            metadata={"scene_id": "scene", "query": "trash can"}
        )
        memory.stage_many(observations)

        config = SpatialGateConfig(center_distance_threshold=0.15)
        outcome = associate_pending(
            memory,
            config,
        )
        evaluation = evaluate_predictions(
            observations, outcome["pairs"], labels
        )

        self.assertEqual(evaluation["metrics"]["f1"], 1.0)
        self.assertEqual(len(outcome["pairs"]), 6)
        self.assertEqual(len(memory.objects), 1)
        self.assertEqual(list(memory.pending_observations), ["c", "d"])
        permanent = next(iter(memory.objects.values()))
        self.assertEqual(
            [item.obs_id for item in permanent.observations],
            ["a", "b"],
        )
        self.assertEqual(permanent.evidence_frames, [
            "frame_0001",
            "frame_0002",
        ])
        self.assertEqual(len(memory.decisions), 2)
        self.assertEqual(
            [item["promoted"] for item in outcome["components"]],
            [True, False],
        )

    def test_exact_class_gate_rejects_nearby_different_text(self) -> None:
        first = make_observation(
            "a", "frame_1", [0, 0, 0], class_text="trash_can"
        )
        second = make_observation(
            "b", "frame_2", [0, 0, 0], class_text="waste bin"
        )
        pair = predict_pair(
            first,
            second,
            SpatialGateConfig(),
        )
        self.assertFalse(pair.same_class)
        self.assertFalse(pair.predicted_same)
        self.assertEqual(pair.gate_reasons, ("class_mismatch",))

    def test_overlap_can_pass_when_center_distance_fails(self) -> None:
        first = make_observation(
            "a", "frame_1", [0, 0, 0], extent=1.0
        )
        second = make_observation(
            "b", "frame_2", [0.3, 0, 0], extent=1.0
        )
        pair = predict_pair(
            first,
            second,
            SpatialGateConfig(
                center_distance_threshold=0.1,
                min_overlap_iou=0.1,
            ),
        )
        self.assertFalse(pair.center_distance_pass)
        self.assertTrue(pair.overlap_pass)
        self.assertTrue(pair.predicted_same)

    def test_false_negative_is_saved_as_failure_case(self) -> None:
        observations = [
            make_observation(
                "a", "frame_1", [0, 0, 0], extent=0.05
            ),
            make_observation(
                "b", "frame_2", [0.3, 0, 0], extent=0.05
            ),
        ]
        labels = make_labels([("instance_a", ("a", "b"))])
        memory = ObjectMemory(
            metadata={"scene_id": "scene", "query": "trash can"}
        )
        memory.stage_many(observations)

        config = SpatialGateConfig(center_distance_threshold=0.1)
        outcome = associate_pending(memory, config)
        evaluation = evaluate_predictions(
            observations, outcome["pairs"], labels
        )

        self.assertEqual(evaluation["metrics"]["false_negative"], 1)
        self.assertEqual(evaluation["metrics"]["f1"], 0.0)
        self.assertEqual(
            evaluation["failure_cases"][0]["error_type"],
            "false_negative",
        )
        self.assertEqual(len(memory.objects), 0)

    def test_label_changes_do_not_change_predictions(self) -> None:
        observations = [
            make_observation("a", "frame_1", [0.0, 0.0, 0.0], extent=0.05),
            make_observation("b", "frame_2", [0.05, 0.0, 0.0], extent=0.05),
            make_observation("c", "frame_3", [1.0, 0.0, 0.0], extent=0.05),
        ]
        predictions = predict_all_pairs(
            observations,
            SpatialGateConfig(center_distance_threshold=0.15),
        )
        frozen_predictions = [pair.to_dict() for pair in predictions]
        all_same = make_labels([
            ("one_instance", ("a", "b", "c")),
        ])
        all_different = make_labels([
            ("instance_a", ("a",)),
            ("instance_b", ("b",)),
            ("instance_c", ("c",)),
        ])

        first_evaluation = evaluate_predictions(
            observations, predictions, all_same
        )
        second_evaluation = evaluate_predictions(
            observations, predictions, all_different
        )

        self.assertEqual(D9_PREDICTION_SCHEMA_VERSION, "0.2")
        self.assertEqual(
            [pair.to_dict() for pair in predictions],
            frozen_predictions,
        )
        self.assertNotEqual(
            first_evaluation["metrics"],
            second_evaluation["metrics"],
        )
        self.assertTrue(all(
            "expected_same" not in pair and "error_type" not in pair
            for pair in frozen_predictions
        ))

    def test_association_is_invariant_to_observation_order(self) -> None:
        observations = [
            make_observation("a", "frame_0001", [0.0, 0.0, 0.0]),
            make_observation("b", "frame_0002", [0.05, 0.0, 0.0]),
            make_observation("c", "frame_0001", [1.0, 0.0, 0.0]),
            make_observation("d", "frame_0001", [1.04, 0.0, 0.0]),
        ]
        config = SpatialGateConfig(center_distance_threshold=0.15)

        def run(order: list[ObjectObservation]) -> tuple[dict, dict]:
            memory = ObjectMemory(
                metadata={"scene_id": "scene", "query": "trash can"}
            )
            memory.stage_many(order)
            return associate_pending(memory, config), memory.to_dict()

        forward = run(observations)
        reverse = run(list(reversed(observations)))

        self.assertEqual(forward, reverse)

    def test_bridge_case_exposes_a1_transitive_false_merge(self) -> None:
        observations = [
            make_observation("a", "frame_1", [0.0, 0.0, 0.0], extent=0.01),
            make_observation("b", "frame_2", [0.1, 0.0, 0.0], extent=0.01),
            make_observation("c", "frame_3", [0.2, 0.0, 0.0], extent=0.01),
        ]
        memory = ObjectMemory(
            metadata={"scene_id": "scene", "query": "trash can"}
        )
        memory.stage_many(observations)
        outcome = associate_pending(
            memory,
            SpatialGateConfig(center_distance_threshold=0.11),
        )
        by_pair = {
            (pair["obs_id_a"], pair["obs_id_b"]): pair
            for pair in outcome["pairs"]
        }

        self.assertTrue(by_pair[("a", "b")]["predicted_same"])
        self.assertFalse(by_pair[("a", "c")]["predicted_same"])
        self.assertTrue(by_pair[("b", "c")]["predicted_same"])
        self.assertEqual(len(outcome["components"]), 1)
        self.assertEqual(
            outcome["components"][0]["observation_ids"],
            ["a", "b", "c"],
        )

        labels = make_labels([
            ("instance_ab", ("a", "b")),
            ("instance_c", ("c",)),
        ])
        evaluation = evaluate_predictions(
            observations, outcome["pairs"], labels
        )
        self.assertEqual(evaluation["metrics"]["false_positive"], 1)
        self.assertEqual(
            evaluation["failure_cases"][0]["error_type"],
            "false_positive",
        )

    def test_zero_match_prediction_is_valid(self) -> None:
        observations = [
            make_observation("a", "frame_1", [0.0, 0.0, 0.0], extent=0.01),
            make_observation("b", "frame_2", [1.0, 0.0, 0.0], extent=0.01),
        ]
        memory = ObjectMemory(
            metadata={"scene_id": "scene", "query": "trash can"}
        )
        memory.stage_many(observations)

        outcome = associate_pending(
            memory,
            SpatialGateConfig(center_distance_threshold=0.1),
        )

        self.assertFalse(outcome["pairs"][0]["predicted_same"])
        self.assertEqual(len(outcome["components"]), 2)
        self.assertEqual(len(memory.objects), 0)
        self.assertEqual(list(memory.pending_observations), ["a", "b"])
        labels = make_labels([
            ("instance_a", ("a",)),
            ("instance_b", ("b",)),
        ])
        evaluation = evaluate_predictions(
            observations, outcome["pairs"], labels
        )
        self.assertEqual(evaluation["metrics"]["true_negative"], 1)
        self.assertEqual(evaluation["metrics"]["accuracy"], 1.0)

    def test_labels_must_cover_every_observation_exactly(self) -> None:
        observations = [
            make_observation("a", "frame_1", [0, 0, 0]),
            make_observation("b", "frame_2", [0, 0, 0]),
        ]
        predictions = predict_all_pairs(
            observations, SpatialGateConfig()
        )
        labels = make_labels([("instance_a", ("a",))])
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            evaluate_predictions(observations, predictions, labels)

    def test_single_observation_cannot_be_promoted_directly(self) -> None:
        observation = make_observation("a", "frame_1", [0, 0, 0])
        memory = ObjectMemory()
        memory.stage_many([observation])
        with self.assertRaisesRegex(ValueError, "at least two"):
            memory.promote_group([observation])


if __name__ == "__main__":
    unittest.main()
