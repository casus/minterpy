# -*- coding:utf-8 -*-
import unittest

import numpy as np

from minterpy import MultiIndex, Grid, NewtonPolynomial, TransformationNewtonToCanonical
from minterpy.utils import report_error
from tests.test_settings import DESIRED_PRECISION
from tests.auxiliaries import check_different_settings, rnd_points


# TODO unit test the single evaluation functions!


def eval_equality_test(spatial_dimension, poly_degree, lp_degree):
    multi_index = MultiIndex.from_degree(spatial_dimension, poly_degree, lp_degree)
    grid = Grid(multi_index)
    interpolation_nodes = grid.unisolvent_nodes

    nr_coefficients = len(multi_index)
    print(f"  - no. coefficients: {nr_coefficients}")
    coeffs_newton_true = rnd_points(nr_coefficients)  # \in [-1; 1]
    newton_poly = NewtonPolynomial(coeffs_newton_true, multi_index, grid=grid)

    # NOTE: if this fails, there could be an error in the transformation as well!
    n2c_transformation = TransformationNewtonToCanonical(newton_poly)
    # NOTE: includes building the transformation matrix!
    canonical_poly = n2c_transformation()

    # evaluating the newton ground truth polynomial on all interpolation points gives the lagrange coefficients:
    coeffs_lagrange1 = newton_poly(interpolation_nodes)

    # evaluating the polynomial in canonical form should yield the same coefficients:
    coeffs_lagrange2 = canonical_poly(interpolation_nodes)
    np.testing.assert_almost_equal(coeffs_lagrange1, coeffs_lagrange2, decimal=DESIRED_PRECISION)
    err = coeffs_lagrange1 - coeffs_lagrange2
    report_error(err, f'error of the Lagrange coefficients (= polynomial evaluated on the interpolation nodes):')


class PolyEvalTest(unittest.TestCase):

    def test_equality(self):
        print('\ntesting the equality of the different evaluation implementations:')
        # includes speed benchmarks
        check_different_settings(eval_equality_test)


# TODO test numerical error: generate precise ground truths with an increased precision (float 128) and compare
#   look at dev_jannik newt_eval_test.py
# TODO compare speed?

if __name__ == '__main__':
    # suite = unittest.TestLoader().loadTestsFromTestCase(HelperTest)
    # unittest.TextTestRunner(verbosity=2).run(suite)
    unittest.main()
