import unittest
import numpy as np

from src.stereo_matching import (
    evaluate_matches,
    match_points_ncc,
    normalized_cross_correlation,
)


class TestNCC(unittest.TestCase):
    def test_identical_patch_has_unit_ncc(self):
        a = np.arange(25, dtype=float).reshape(5,5)
        self.assertAlmostEqual(normalized_cross_correlation(a, a), 1.0, places=12)

    def test_synthetic_match(self):
        left = np.zeros((80, 100), dtype=np.uint8)
        right = np.zeros((80, 100), dtype=np.uint8)

        # Distinct 7x7 pattern centered at (50, 40) in left
        pattern = np.array([
            [0,0,0,1,0,0,0],
            [0,1,1,2,1,1,0],
            [0,1,3,4,3,1,0],
            [1,2,4,9,4,2,1],
            [0,1,3,4,3,1,0],
            [0,1,1,2,1,1,0],
            [0,0,0,1,0,0,0],
        ], dtype=np.uint8) * 20

        left[37:44,47:54] = pattern

        # Shift same pattern to center (31, 42) in right
        right[39:46,28:35] = pattern

        pts = np.array([[50,40]], dtype=float)
        search = np.array([[20,45]], dtype=float)

        matched, scores = match_points_ncc(
            right, left, pts, search,
            kernel_size=7,
            threshold=0.4,
            y_half_window=10,
        )

        np.testing.assert_allclose(matched, [[31,42]])
        self.assertGreater(scores[0], 0.99)

    def test_rmse(self):
        matched = np.array([[10,10],[20,20],[0,0]], dtype=float)
        gt = np.array([[13,14],[20,20],[100,100]], dtype=float)
        result = evaluate_matches(matched, gt)
        self.assertEqual(result["n_valid"], 2)
        self.assertAlmostEqual(result["rmse_px"], (25/2)**0.5)


if __name__ == "__main__":
    unittest.main()
