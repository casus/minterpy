"""
Testing module for lagrange_polynomial.py

"""

import numpy as np
from conftest import (MultiIndices, NrPoints, NrSimilarPolynomials,
                      assert_polynomial_almost_equal, build_rnd_coeffs,
                      build_rnd_points)

from minterpy import CanonicalPolynomial, LagrangePolynomial, MultiIndexSet
from minterpy.transformations import CanonicalToLagrange, LagrangeToCanonical


def test_neg(MultiIndices, NrSimilarPolynomials):
    coeffs = build_rnd_coeffs(MultiIndices, NrSimilarPolynomials)
    poly = LagrangePolynomial(MultiIndices, coeffs)
    res = -poly
    groundtruth_coeffs = (-1) * coeffs
    groundtruth = poly.__class__(MultiIndices, groundtruth_coeffs)
    assert_polynomial_almost_equal(res, groundtruth)


Mi1 = MultiIndices
Mi2 = MultiIndices


def test_add_poly(Mi1, Mi2):
    coeffs1 = build_rnd_coeffs(Mi1)
    coeffs2 = build_rnd_coeffs(Mi2)

    lag_poly_1 = LagrangePolynomial(Mi1, coeffs1)
    lag_poly_2 = LagrangePolynomial(Mi2, coeffs2)

    res = lag_poly_1 + lag_poly_2

    transform_l2c_poly_1 = LagrangeToCanonical(lag_poly_1)
    transform_l2c_poly_2 = LagrangeToCanonical(lag_poly_2)
    transform_l2c_res = LagrangeToCanonical(res)

    can_poly_1 = transform_l2c_poly_1()
    can_poly_2 = transform_l2c_poly_2()
    can_poly_res = transform_l2c_res()

    groundtruth_res_poly = can_poly_1 + can_poly_2

    assert_polynomial_almost_equal(groundtruth_res_poly, can_poly_res)


def test_sub_poly(Mi1, Mi2):
    coeffs1 = build_rnd_coeffs(Mi1)
    coeffs2 = build_rnd_coeffs(Mi2)

    lag_poly_1 = LagrangePolynomial(Mi1, coeffs1)
    lag_poly_2 = LagrangePolynomial(Mi2, coeffs2)

    res = lag_poly_1 - lag_poly_2

    transform_l2c_poly_1 = LagrangeToCanonical(lag_poly_1)
    transform_l2c_poly_2 = LagrangeToCanonical(lag_poly_2)
    transform_l2c_res = LagrangeToCanonical(res)

    can_poly_1 = transform_l2c_poly_1()
    can_poly_2 = transform_l2c_poly_2()
    can_poly_res = transform_l2c_res()

    groundtruth_res_poly = can_poly_1 - can_poly_2
    assert_polynomial_almost_equal(can_poly_res, groundtruth_res_poly)


def test_mul():
    mi = MultiIndexSet.from_degree(2, 1, 2.0)
    canonical_coeffs1 = np.array([0, 1, 1])
    canonical_coeffs2 = np.array([0, 1, -1])

    can_poly_1 = CanonicalPolynomial(mi, canonical_coeffs1)
    can_poly_2 = CanonicalPolynomial(mi, canonical_coeffs2)

    c2l_transformation = CanonicalToLagrange(can_poly_1)

    lagrange_poly1 = c2l_transformation(can_poly_1)
    lagrange_poly2 = c2l_transformation(can_poly_2)

    res_lagrange_poly = lagrange_poly1 * lagrange_poly2

    l2c_transformation = LagrangeToCanonical(res_lagrange_poly)
    res_canonical_poly = l2c_transformation()

    groundtruth_product_coeffs = np.array([0, 0, 1, 0, 0, -1])
    groundtruth_mi = MultiIndexSet.from_degree(2, 2, 4.0)
    groundtruth = CanonicalPolynomial(groundtruth_mi, groundtruth_product_coeffs)

    assert_polynomial_almost_equal(groundtruth, res_canonical_poly)


def test_add_different_poly():
    mi1 = MultiIndexSet.from_degree(3, 2, 2.0)
    coeffs1 = np.array([1, 2, 0, 4, 3, 0, 5, 0, 0, 0, 0])

    mi2 = MultiIndexSet.from_degree(2, 2, 2.0)
    coeffs2 = np.array([1, 2, 0, 3, 4, 0])

    can_poly_1 = CanonicalPolynomial(mi1, coeffs1)
    can_poly_2 = CanonicalPolynomial(mi2, coeffs2)

    transform_c2l_1 = CanonicalToLagrange(can_poly_1)
    transform_c2l_2 = CanonicalToLagrange(can_poly_2)

    lag_poly_1 = transform_c2l_1()
    lag_poly_2 = transform_c2l_2()

    res = lag_poly_1 + lag_poly_2

    groundtruth_coeffs = np.array([2, 4, 0, 7, 7, 0, 5, 0, 0, 0, 0])

    groundtruth_multi_index = mi1

    groundtruth_canonical = CanonicalPolynomial(
        groundtruth_multi_index, groundtruth_coeffs
    )

    transform_c2l_res = CanonicalToLagrange(groundtruth_canonical)
    groundtruth = transform_c2l_res()

    assert_polynomial_almost_equal(res, groundtruth)
