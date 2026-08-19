import tempfile
import unittest
from pathlib import Path

import numpy as np

from relground.calibration import LogisticCalibrator


class CalibrationTests(unittest.TestCase):
    def test_fit_predict_and_round_trip(self) -> None:
        features = np.array(
            [
                [0.1, 0.1],
                [0.2, 0.3],
                [0.8, 0.7],
                [0.9, 0.9],
            ]
        )
        labels = [0, 0, 1, 1]
        calibrator = LogisticCalibrator(max_iter=800).fit(features, labels)
        probabilities = calibrator.predict_proba(features)
        self.assertLess(probabilities[0], probabilities[-1])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibrator.json"
            calibrator.save(path)
            restored = LogisticCalibrator.load(path)
            np.testing.assert_allclose(probabilities, restored.predict_proba(features))


if __name__ == "__main__":
    unittest.main()
