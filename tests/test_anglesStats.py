from unittest import TestCase

import numpy as np

from cesped.utils.anglesStats import computeAngularError


class Test(TestCase):
    def test_basic(self):
        # Add a basic test case
        predEulers = np.array([[0, 0, 0], [90, 45, 30]], dtype=np.float32)
        trueEulers = np.array([[0, 0, 0], [90, 45, 30]], dtype=np.float32)
        error, w_error, totalConf = computeAngularError(predEulers, trueEulers)

        self.assertAlmostEqual(error.item(), 0.0, places=4)
        self.assertEqual(np.isnan(w_error).any(), True)

    def test_symmetry_c2(self):
        # Add a test case for different symmetry groups
        predEulers = np.array([[0, 0, 0]], dtype=np.float32)
        trueEulers = np.array([[180, 0, 0]], dtype=np.float32)
        error, w_error, totalConf = computeAngularError(predEulers, trueEulers, symmetry="c2")

        self.assertAlmostEqual(error.item(), 0.0, places=4)

    def test_with_confidence(self):
        # Test with confidence values
        predEulers = np.array([[0, 0, 0], [90, 45, 30]], dtype=np.float32)
        trueEulers = np.array([[0, 0, 0], [90, 45, 30]], dtype=np.float32)
        confidence = np.array([1.0, 0.5], dtype=np.float32)
        error, w_error, totalConf = computeAngularError(predEulers, trueEulers, confidence)

        self.assertAlmostEqual(error, 0.0, places=4)
        self.assertAlmostEqual(w_error, 0.0, places=4)

    def test_c3_symmetry(self):
        # Test for c3 symmetry
        predEulers = np.array([[0, 0, 0]], dtype=np.float32)
        trueEulers = np.array([[120, 0, 0]], dtype=np.float32)
        error, w_error, totalConf = computeAngularError(predEulers, trueEulers, symmetry="c3")

        self.assertAlmostEqual(error.item(), 0.0, places=4)

    def test_non_zero_error(self):
        # Test for non-zero error
        predEulers = np.array([[10, 0, 0]], dtype=np.float32)
        trueEulers = np.array([[0, 0, 0]], dtype=np.float32)
        error, w_error, totalConf = computeAngularError(predEulers, trueEulers)

        self.assertAlmostEqual(error.item(), 10.0, places=4)

    def test_multiple_batches(self):
        # Test for multiple batches
        predEulers = np.array([[0, 0, 0], [90, 0, 0]], dtype=np.float32)
        trueEulers = np.array([[0, 0, 0], [270, 0, 0]], dtype=np.float32)
        error, w_error, totalConf = computeAngularError(predEulers, trueEulers, symmetry="c4")
        self.assertAlmostEqual(error.item(), 0.0, places=4)

    def test_exact_non_zero_error_with_c2_symmetry(self):
        # Test for a non-zero exact error with c2 symmetry
        predEulers = np.array([[90, 0, 0]], dtype=np.float32)
        trueEulers = np.array([[275, 0, 0]], dtype=np.float32)
        error, w_error, totalConf = computeAngularError(predEulers, trueEulers, symmetry="c2")

        self.assertAlmostEqual(error.item(), 5., places=4)