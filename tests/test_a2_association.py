import hashlib
import unittest
from pathlib import Path

import numpy as np

from relground.a2_association import (
    EvidenceAssociationConfig,
    associate_pending_a2,
    complete_link_clusters,
    compute_pair,
    predict_all_pairs,
)
from relground.association import ObjectMemory
from relground.schemas import ObjectObservation, OrientedBoundingBox
from scripts.validate_d9_association import validate_output as validate_a1
from scripts.validate_d9_evaluation import validate_output as validate_a1_eval


def make_observation(
    obs_id: str,
    frame_id: str,
    center: list[float],
    *,
    class_text: str = "trash can",
    retrieval: float = 0.9,
    sam: float = 0.9,
    valid: float = 0.9,
    embedding: list[float] | None = None,
    extent: float = 0.04,
) -> ObjectObservation:
    center_array = np.asarray(center, dtype=float)
    return ObjectObservation(
        obs_id=obs_id,
        class_text=class_text,
        frame_id=frame_id,
        mask_ref=None,
        retrieval_score=retrieval,
        sam_score=sam,
        valid_point_ratio=valid,
        points_ref=None,
        center=center_array,
        obb=OrientedBoundingBox(
            center=center_array,
            extent=np.full(3, extent),
        ),
        semantic_embedding=embedding,
    )


class A2AssociationTests(unittest.TestCase):
    def test_default_config_freezes_thresholds_and_weights(self) -> None:
        config = EvidenceAssociationConfig()

        self.assertEqual(config.semantic_threshold, 0.70)
        self.assertEqual(config.min_observation_quality, 0.25)
        self.assertEqual(config.center_distance_threshold, 0.15)
        self.assertEqual(
            config.to_dict()["weights"],
            {
                "semantic": 0.25,
                "center": 0.25,
                "overlap": 0.20,
                "obb_shape": 0.15,
                "quality": 0.15,
            },
        )
        self.assertEqual(
            EvidenceAssociationConfig.from_dict(config.to_dict()), config
        )

    def test_missing_embedding_falls_back_to_normalized_exact_class(self) -> None:
        first = make_observation(
            "a", "frame_0001", [0.0, 0.0, 0.0], embedding=[1.0, 0.0]
        )
        second = make_observation(
            "b", "frame_0002", [0.05, 0.0, 0.0], class_text="Trash_Can"
        )

        pair = compute_pair(first, second, EvidenceAssociationConfig())

        self.assertEqual(pair.semantic_mode, "exact_class")
        self.assertEqual(pair.semantic_similarity, 1.0)
        self.assertTrue(pair.semantic_compatible)
        self.assertTrue(pair.gate_pass)

    def test_embedding_cosine_can_reject_matching_class_text(self) -> None:
        first = make_observation(
            "a", "f1", [0.0, 0.0, 0.0], embedding=[1.0, 0.0]
        )
        second = make_observation(
            "b", "f2", [0.05, 0.0, 0.0], embedding=[-1.0, 0.0]
        )

        pair = compute_pair(first, second, EvidenceAssociationConfig())

        self.assertEqual(pair.semantic_mode, "embedding_cosine")
        self.assertFalse(pair.semantic_compatible)
        self.assertFalse(pair.gate_pass)

    def test_class_conflict_and_low_quality_are_explicit_rejections(self) -> None:
        good = make_observation("a", "f1", [0.0, 0.0, 0.0])
        different = make_observation(
            "b", "f2", [0.01, 0.0, 0.0], class_text="chair"
        )
        weak = make_observation(
            "c",
            "f3",
            [0.02, 0.0, 0.0],
            retrieval=0.01,
            sam=0.1,
            valid=0.1,
        )

        class_pair = compute_pair(good, different, EvidenceAssociationConfig())
        weak_pair = compute_pair(good, weak, EvidenceAssociationConfig())

        self.assertFalse(class_pair.gate_pass)
        self.assertIn("semantic_exact_class_reject", class_pair.reasons)
        self.assertFalse(weak_pair.quality_pass)
        self.assertIn("low_quality_b", weak_pair.reasons)

    def test_complete_link_splits_a_b_c_bridge(self) -> None:
        observations = [
            make_observation("a", "f1", [0.0, 0.0, 0.0]),
            make_observation("b", "f2", [0.1, 0.0, 0.0]),
            make_observation("c", "f3", [0.2, 0.0, 0.0]),
        ]
        raw_pairs = predict_all_pairs(
            observations, EvidenceAssociationConfig()
        )

        clusters, pairs, decisions = complete_link_clusters(
            observations, raw_pairs
        )
        membership = [[item.obs_id for item in cluster] for cluster in clusters]
        by_key = {(item.obs_id_a, item.obs_id_b): item for item in pairs}

        self.assertEqual(membership, [["a", "b"], ["c"]])
        self.assertTrue(by_key[("a", "b")].predicted_same)
        self.assertTrue(by_key[("b", "c")].gate_pass)
        self.assertFalse(by_key[("b", "c")].predicted_same)
        self.assertIn("complete_link_conflict", by_key[("b", "c")].reasons)
        self.assertEqual(len(decisions), 1)

    def test_association_is_invariant_to_input_order(self) -> None:
        observations = [
            make_observation("a", "f1", [0.0, 0.0, 0.0]),
            make_observation("b", "f2", [0.05, 0.0, 0.0]),
            make_observation("c", "f3", [1.0, 0.0, 0.0]),
        ]
        outputs = []
        for ordering in (observations, list(reversed(observations))):
            memory = ObjectMemory(metadata={"scene_id": "s", "query": "q"})
            memory.stage_many(ordering)
            outcome = associate_pending_a2(
                memory, EvidenceAssociationConfig()
            )
            outputs.append((outcome, memory.to_dict()))

        self.assertEqual(outputs[0], outputs[1])

    def test_same_frame_duplicates_cluster_but_remain_pending(self) -> None:
        memory = ObjectMemory(metadata={"scene_id": "s", "query": "q"})
        memory.stage_many([
            make_observation("a", "f1", [0.0, 0.0, 0.0]),
            make_observation("b", "f1", [0.01, 0.0, 0.0]),
        ])

        outcome = associate_pending_a2(memory, EvidenceAssociationConfig())

        self.assertEqual(outcome["clusters"][0]["observation_ids"], ["a", "b"])
        self.assertFalse(outcome["clusters"][0]["promoted"])
        self.assertEqual(set(memory.pending_observations), {"a", "b"})
        self.assertEqual(memory.objects, {})

    def test_cross_frame_promotion_conserves_observations(self) -> None:
        memory = ObjectMemory(metadata={"scene_id": "s", "query": "q"})
        memory.stage_many([
            make_observation("a", "f1", [0.0, 0.0, 0.0]),
            make_observation("b", "f2", [0.02, 0.0, 0.0]),
            make_observation("c", "f3", [1.0, 0.0, 0.0]),
        ])
        source_ids = set(memory.pending_observations)

        associate_pending_a2(memory, EvidenceAssociationConfig())
        associated = {
            item.obs_id
            for obj in memory.objects.values()
            for item in obj.observations
        }

        self.assertFalse(associated & set(memory.pending_observations))
        self.assertEqual(associated | set(memory.pending_observations), source_ids)
        self.assertEqual(len(memory.objects), 1)

    def test_published_a1_hash_and_validators_remain_frozen(self) -> None:
        root = Path(__file__).resolve().parents[1]
        bundle = root / "evidence/week2/d9-office-loop-trash-can"
        result = bundle / "prediction/d9_result.json"
        memory = bundle / "prediction/object_memory.json"

        self.assertEqual(
            hashlib.sha256(result.read_bytes()).hexdigest(),
            "6aabd27144ab57e77f00e3b3dab499b22fd0a386feb21df9a3adb4c86a8b0362",
        )
        self.assertEqual(
            hashlib.sha256(memory.read_bytes()).hexdigest(),
            "1d8029e819db70d15877797de7618635730731fca510056e452264d895cce9f3",
        )
        self.assertEqual(
            validate_a1(bundle / "prediction")["status"], "PASS"
        )
        self.assertEqual(
            validate_a1_eval(bundle / "evaluation")["status"], "PASS"
        )


if __name__ == "__main__":
    unittest.main()
