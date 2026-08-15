'''
Created on 14 August 2026

check synthetic motion truth and benchmark errors

@author: Dinghao Luo
'''

#%% imports
from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from registration_benchmark import _warp, registration_errors


#%% tests
class RegistrationBenchmarkTests(unittest.TestCase):
    def test_positive_displacement_moves_image_down_and_right(self):
        image = np.zeros((21, 21), dtype=np.float32)
        image[10, 10] = 1
        shifted = _warp(image, 3, 4)

        self.assertEqual(np.unravel_index(np.argmax(shifted), shifted.shape), (13, 14))

    def test_reference_offset_is_fitted_on_calibration_frames_only(self):
        truth_y = np.array([0, 1, 2, 3], dtype=float)
        truth_x = np.array([0, -1, -2, -3], dtype=float)
        estimate_y = truth_y + np.array([5, 5, 7, 7])
        estimate_x = truth_x + np.array([-2, -2, -1, -1])
        calibration = np.array([True, True, False, False])
        estimable = np.ones(4, dtype=bool)

        error, offset_y, offset_x, valid = registration_errors(
            estimate_y, estimate_x, truth_y, truth_x, calibration, estimable)

        self.assertEqual((offset_y, offset_x), (5, -2))
        np.testing.assert_allclose(error, [0, 0, np.sqrt(5), np.sqrt(5)])
        np.testing.assert_array_equal(valid, estimable)


if __name__ == '__main__':
    unittest.main()
