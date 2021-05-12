import unittest

import numpy as np
from auxiliaries import check_different_settings, rnd_points
from test_settings import DESIRED_PRECISION

from minterpy import (
    CanonicalPolynomial,
    Grid,
    LagrangePolynomial,
    MultiIndex,
    TransformationCanonicalToLagrange,
    TransformationLagrangeToCanonical,
)
from minterpy.utils import report_error


class CanonicalPolyArithmeticTest(unittest.TestCase):
    def setUp(self):
        exps1 = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        args1 = np.lexsort(exps1.T, axis=-1)
        mi1 = MultiIndex(exps1[args1])
        coeffs1 = np.array([1, 2, 3, 4])

        exps2 = np.array([[0, 0, 0], [2, 0, 0], [1, 1, 0], [0, 2, 0], [0, 0, 2]])
        args2 = np.lexsort(exps2.T, axis=-1)
        mi2 = MultiIndex(exps2[args2])
        coeffs2 = np.array([1, 2, 3, 4, 5])

        self.p1 = CanonicalPolynomial(coeffs1[args1], mi1)
        self.p2 = CanonicalPolynomial(coeffs2[args2], mi2)

    def test_neg(self):
        res = -self.p1
        groundtruth_coeffs = (-1) * self.p1.coeffs
        np.testing.assert_almost_equal(
            res.multi_index.exponents,
            self.p1.multi_index.exponents,
            decimal=DESIRED_PRECISION,
        )
        np.testing.assert_almost_equal(
            res.coeffs, groundtruth_coeffs, decimal=DESIRED_PRECISION
        )

    def test_add_same_poly(self):
        res = self.p1 + self.p1
        groundtruth_coeffs = self.p1.coeffs * 2
        np.testing.assert_almost_equal(
            res.multi_index.exponents,
            self.p1.multi_index.exponents,
            decimal=DESIRED_PRECISION,
        )
        np.testing.assert_almost_equal(
            res.coeffs, groundtruth_coeffs, decimal=DESIRED_PRECISION
        )

    def test_add_different_poly(self):
        res = self.p1 + self.p2
        groundtruth_coeffs = np.array([2, 2, 2, 3, 7, 4, 5])
        groundtruth_multi_index_exponents = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 2, 0],
                [0, 0, 2],
            ]
        )
        np.testing.assert_almost_equal(
            res.multi_index.exponents,
            groundtruth_multi_index_exponents,
            decimal=DESIRED_PRECISION,
        )
        np.testing.assert_almost_equal(
            res.coeffs, groundtruth_coeffs, decimal=DESIRED_PRECISION
        )

    def test_sub_same_poly(self):
        res = self.p1 - self.p1
        groundtruth_coeffs = np.zeros(self.p1.coeffs.shape)
        np.testing.assert_almost_equal(
            res.multi_index.exponents,
            self.p1.multi_index.exponents,
            decimal=DESIRED_PRECISION,
        )
        np.testing.assert_almost_equal(
            res.coeffs, groundtruth_coeffs, decimal=DESIRED_PRECISION
        )

    def test_sub_different_poly(self):
        res = self.p1 - self.p2
        groundtruth_coeffs = np.array([0, 2, -2, 3, 1, -4, -5])
        groundtruth_multi_index_exponents = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 2, 0],
                [0, 0, 2],
            ]
        )
        np.testing.assert_almost_equal(
            res.multi_index.exponents,
            groundtruth_multi_index_exponents,
            decimal=DESIRED_PRECISION,
        )
        np.testing.assert_almost_equal(
            res.coeffs, groundtruth_coeffs, decimal=DESIRED_PRECISION
        )


if __name__ == "__main__":
    unittest.main()
