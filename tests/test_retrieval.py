import unittest

import numpy as np

from relground.retrieval import (
    FrameCandidate,
    RetrievalConfig,
    TopKFrameRetriever,
    viewpoint_from_world_pose,
)


class RetrievalTests(unittest.TestCase):
    def test_temporal_redundancy_keeps_best_frame(self) -> None:
        candidates = [
            FrameCandidate("f0", 0.95, index=0),
            FrameCandidate("f1", 0.90, index=1),
            FrameCandidate("f8", 0.80, index=8),
        ]
        retriever = TopKFrameRetriever(
            RetrievalConfig(top_k=2, redundancy="temporal", min_frame_gap=3)
        )
        self.assertEqual([item.frame_id for item in retriever.retrieve(candidates)], ["f0", "f8"])

    def test_viewpoint_redundancy_uses_position_and_angle(self) -> None:
        candidates = [
            FrameCandidate("a", 0.9, camera_center=np.zeros(3), view_direction=np.array([0, 0, 1])),
            FrameCandidate(
                "b", 0.8, camera_center=np.array([0.01, 0, 0]), view_direction=np.array([0, 0, 1])
            ),
            FrameCandidate("c", 0.7, camera_center=np.ones(3), view_direction=np.array([0, 0, 1])),
        ]
        retriever = TopKFrameRetriever(RetrievalConfig(top_k=3, redundancy="viewpoint"))
        self.assertEqual([item.frame_id for item in retriever.retrieve(candidates)], ["a", "c"])

    def test_score_ties_preserve_geometry_order(self) -> None:
        candidates = [
            FrameCandidate("z_first", 0.5, index=0),
            FrameCandidate("a_second", 0.5, index=1),
        ]
        selected = TopKFrameRetriever(
            RetrievalConfig(top_k=1, redundancy="none")
        ).retrieve(candidates)
        self.assertEqual(selected[0].frame_id, "z_first")

    def test_k_outputs_are_prefix_consistent(self) -> None:
        candidates = [
            FrameCandidate(f"f{index}", 1.0 - index / 10.0, index=index)
            for index in range(7)
        ]
        outputs = {
            k: TopKFrameRetriever(
                RetrievalConfig(top_k=k, redundancy="temporal", min_frame_gap=2)
            ).retrieve(candidates)
            for k in (1, 3, 5)
        }
        self.assertEqual(outputs[1], outputs[3][:1])
        self.assertEqual(outputs[3], outputs[5][:3])

    def test_world_pose_uses_translation_and_positive_z_axis(self) -> None:
        angle = np.pi / 2.0
        pose = np.eye(4)
        pose[:3, :3] = [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
        pose[:3, 3] = [1.0, 2.0, 3.0]
        center, direction = viewpoint_from_world_pose(pose)
        np.testing.assert_allclose(center, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(direction, [1.0, 0.0, 0.0], atol=1e-8)


if __name__ == "__main__":
    unittest.main()
