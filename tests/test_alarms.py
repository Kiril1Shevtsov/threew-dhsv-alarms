import unittest
import numpy as np
import pandas as pd
from src.alarms import evaluate
from src.features import features, SENSORS


class AlarmTests(unittest.TestCase):
    def meta(self):
        return pd.DataFrame(
            {
                "file": ["a"] * 6,
                "well": ["w"] * 6,
                "second": [3600, 3615, 3630, 3645, 3660, 3675],
                "onset": [3645] * 6,
                "label": [0, 0, 0, 2, 2, 2],
            }
        )

    def test_three_consecutive(self):
        r = evaluate(self.meta(), np.ones(6), 0.5)[0]
        self.assertEqual(r["false_episodes"], 1)
        self.assertEqual(r["false_windows"], 1)
        self.assertTrue(r["detected_300"])
        self.assertFalse(r["clean_300"])

    def test_clean_detection(self):
        r = evaluate(self.meta(), np.array([0, 0, 0, 1, 1, 1]), 0.5)[0]
        self.assertEqual(r["delay_seconds"], 30)
        self.assertTrue(r["clean_300"])

    def test_disabled(self):
        self.assertFalse(evaluate(self.meta(), np.ones(6), 1.0001)[0]["detected_300"])

    def test_gap_does_not_confirm(self):
        m = self.meta()
        m["second"] = [3600, 3615, 3660, 3675, 3690, 3705]
        r = evaluate(m, np.array([1, 1, 1, 0, 0, 0]), 0.5)[0]
        self.assertEqual(r["false_windows"], 0)

    def test_prefix(self):
        rng = np.random.default_rng(7)
        d = pd.DataFrame(rng.normal(size=(4200, 6)), columns=SENSORS)
        for normalized in [False, True]:
            a = features(d.iloc[:4000], normalized)
            b = features(d, normalized).loc[a.index]
            np.testing.assert_allclose(a, b, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
