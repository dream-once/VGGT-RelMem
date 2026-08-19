import unittest

import numpy as np

from relground.retrieval import FrameCandidate, RetrievalConfig, TopKFrameRetriever


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


if __name__ == "__main__":
    unittest.main()
