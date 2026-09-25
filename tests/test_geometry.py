import unittest
import numpy as np

from src.stereo_matching import (
    FlightGeometry,
    compute_epipolar_search_space,
    ground_truth_coverage,
)


class TestEpipolarGeometry(unittest.TestCase):
    def test_reproduces_coursework_search_space(self):
        points = np.array([
            [9362,10999],[5614,6693],[7407,3563],
            [7729,6205],[8518,1279],[4766,9695],
        ], dtype=float)

        expected = np.array([
            [4911,5201],[1163,1453],[2956,3246],
            [3278,3568],[4067,4357],[315,605],
        ], dtype=float)

        actual = compute_epipolar_search_space(points, FlightGeometry())
        np.testing.assert_allclose(actual, expected)

    def test_ground_truth_coverage_is_two_of_six(self):
        search = np.array([
            [4911,5201],[1163,1453],[2956,3246],
            [3278,3568],[4067,4357],[315,605],
        ], dtype=float)
        gt = np.array([
            [5071,11009],[1539,6671],[3278,3563],
            [3552,6203],[4516,1289],[610,9667],
        ], dtype=float)
        coverage = ground_truth_coverage(search, gt)
        self.assertEqual(int(coverage.sum()), 2)
        self.assertListEqual(coverage.tolist(), [True, False, False, True, False, False])


if __name__ == "__main__":
    unittest.main()
