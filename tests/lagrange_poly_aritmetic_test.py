# -*- coding:utf-8 -*-
import unittest

import numpy as np

from minterpy import MultiIndex, Grid, LagrangePolynomial, TransformationLagrangeToCanonical
from minterpy import CanonicalPolynomial, TransformationCanonicalToLagrange
from minterpy.utils import report_error
from test_settings import DESIRED_PRECISION
from auxiliaries import check_different_settings, rnd_points


# TODO: Test addition with different multi indices
def add_lagrange_test(spatial_dimension, poly_degree, lp_degree):
    multi_index = MultiIndex.from_degree(spatial_dimension, poly_degree, lp_degree)
    grid = Grid(multi_index)
    interpolation_nodes = grid.unisolvent_nodes

    nr_coefficients = len(multi_index)
    print(f"  - no. coefficients: {nr_coefficients}")
    coeffs_lagrange1 = rnd_points(nr_coefficients)
    coeffs_lagrange2 = rnd_points(nr_coefficients)
    lagrange_poly1 = LagrangePolynomial(coeffs_lagrange1, multi_index, grid=grid)
    lagrange_poly2 = LagrangePolynomial(coeffs_lagrange2, multi_index, grid=grid)

    result_lagrange_poly = lagrange_poly1 + lagrange_poly2

    l2c_transformation = TransformationLagrangeToCanonical(lagrange_poly1)

    # NOTE: includes building the transformation matrix!
    canonical_poly1 = l2c_transformation(lagrange_poly1)
    canonical_poly2 = l2c_transformation(lagrange_poly2)

    result_canonical_poly = l2c_transformation(result_lagrange_poly)

    coeffs_canonical1 = canonical_poly1.coeffs
    coeffs_canonical2 = canonical_poly2.coeffs

    result_coeffs_canonical = coeffs_canonical1 + coeffs_canonical2

    np.testing.assert_almost_equal(result_canonical_poly.coeffs, result_coeffs_canonical, decimal=DESIRED_PRECISION)
    err = result_canonical_poly.coeffs - result_coeffs_canonical
    report_error(err, f'error in canonical coefficients : ')


def sub_lagrange_test(spatial_dimension, poly_degree, lp_degree):
    multi_index = MultiIndex.from_degree(spatial_dimension, poly_degree, lp_degree)
    grid = Grid(multi_index)
    interpolation_nodes = grid.unisolvent_nodes

    nr_coefficients = len(multi_index)
    print(f"  - no. coefficients: {nr_coefficients}")
    coeffs_lagrange1 = rnd_points(nr_coefficients)
    coeffs_lagrange2 = rnd_points(nr_coefficients)
    lagrange_poly1 = LagrangePolynomial(coeffs_lagrange1, multi_index, grid=grid)
    lagrange_poly2 = LagrangePolynomial(coeffs_lagrange2, multi_index, grid=grid)

    result_lagrange_poly = lagrange_poly1 - lagrange_poly2

    l2c_transformation = TransformationLagrangeToCanonical(lagrange_poly1)

    # NOTE: includes building the transformation matrix!
    canonical_poly1 = l2c_transformation(lagrange_poly1)
    canonical_poly2 = l2c_transformation(lagrange_poly2)

    result_canonical_poly = l2c_transformation(result_lagrange_poly)

    coeffs_canonical1 = canonical_poly1.coeffs
    coeffs_canonical2 = canonical_poly2.coeffs

    result_coeffs_canonical = coeffs_canonical1 - coeffs_canonical2

    np.testing.assert_almost_equal(result_canonical_poly.coeffs, result_coeffs_canonical, decimal=DESIRED_PRECISION)
    err = result_canonical_poly.coeffs - result_coeffs_canonical
    report_error(err, f'error in canonical coefficients : ')


class LagrangePolyArithmeticTest(unittest.TestCase):

    def test_add_lagrange(self):
        print('\ntesting the addition in lagrange basis:')
        check_different_settings(add_lagrange_test)

    def test_sub_lagrange(self):
        print('\ntesting the subtraction in lagrange basis:')
        check_different_settings(sub_lagrange_test)

    # TODO: Add more tests between polynomials having different multiindices
    def test_mul_lagrange(self):
        print('\ntesting the multiplication in lagrange basis:')
        mi = MultiIndex.from_degree(2, 1, 2.0)
        canonical_coeffs1 = np.array([0, 1, 1])
        canonical_coeffs2 = np.array([0, 1, -1])

        canonical_poly = CanonicalPolynomial(canonical_coeffs1, mi)

        c2l_transformation = TransformationCanonicalToLagrange(canonical_poly)

        lagrange_coeffs1 = c2l_transformation.transformation_operator @ canonical_coeffs1
        lagrange_coeffs2 = c2l_transformation.transformation_operator @ canonical_coeffs2

        lagrange_poly1 = LagrangePolynomial(lagrange_coeffs1, mi)
        lagrange_poly2 = LagrangePolynomial(lagrange_coeffs2, mi)

        res_lagrange_poly = lagrange_poly1 * lagrange_poly2
        print(res_lagrange_poly.coeffs)

        l2c_transformation = TransformationLagrangeToCanonical(res_lagrange_poly)
        res_canonical_poly = l2c_transformation(res_lagrange_poly)

        expected_product_coeffs = np.array([0, 0, 1, 0, 0, -1])

        np.testing.assert_almost_equal(expected_product_coeffs, res_canonical_poly.coeffs, decimal=DESIRED_PRECISION)


if __name__ == '__main__':
    unittest.main()
