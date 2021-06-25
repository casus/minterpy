"""
Testing module for lagrange_polynomial.py

"""

import numpy as np

from conftest import (MultiIndices, NrPoints, NrSimilarPolynomials,
                      build_rnd_coeffs, build_rnd_points, assert_polynomial_almost_equal)

from minterpy import (MultiIndex, CanonicalPolynomial, LagrangePolynomial,
                      TransformationLagrangeToCanonical, TransformationCanonicalToLagrange)


def test_neg(MultiIndices,NrSimilarPolynomials):
    coeffs = build_rnd_coeffs(MultiIndices,NrSimilarPolynomials)
    poly = LagrangePolynomial(coeffs,MultiIndices)
    res = -poly
    groundtruth_coeffs = (-1) * coeffs
    groundtruth = poly.__class__(groundtruth_coeffs,MultiIndices)
    assert_polynomial_almost_equal(res,groundtruth)


Mi1 = MultiIndices
Mi2 = MultiIndices

def test_add_different_poly(Mi1,Mi2):
    coeffs1 = build_rnd_coeffs(Mi1)
    coeffs2 = build_rnd_coeffs(Mi2)

    lag_poly_1 = LagrangePolynomial(coeffs1, Mi1)
    lag_poly_2 = LagrangePolynomial(coeffs2, Mi2)

    res = lag_poly_1 + lag_poly_2

    transform_l2c_poly_1 = TransformationLagrangeToCanonical(lag_poly_1)
    transform_l2c_poly_2 = TransformationLagrangeToCanonical(lag_poly_2)
    transform_l2c_res = TransformationLagrangeToCanonical(res)

    can_poly_1 = transform_l2c_poly_1()
    can_poly_2 = transform_l2c_poly_2()
    can_poly_res = transform_l2c_res()

    groundtruth_res_poly = can_poly_1 + can_poly_2
    assert_polynomial_almost_equal(groundtruth_res_poly, can_poly_res)

def test_sub_different_poly(Mi1,Mi2):
    coeffs1 = build_rnd_coeffs(Mi1)
    coeffs2 = build_rnd_coeffs(Mi2)

    lag_poly_1 = LagrangePolynomial(coeffs1, Mi1)
    lag_poly_2 = LagrangePolynomial(coeffs2, Mi2)

    res = lag_poly_1 - lag_poly_2

    transform_l2c_poly_1 = TransformationLagrangeToCanonical(lag_poly_1)
    transform_l2c_poly_2 = TransformationLagrangeToCanonical(lag_poly_2)
    transform_l2c_res = TransformationLagrangeToCanonical(res)

    can_poly_1 = transform_l2c_poly_1()
    can_poly_2 = transform_l2c_poly_2()
    can_poly_res = transform_l2c_res()

    groundtruth_res_poly = can_poly_1 - can_poly_2
    assert_polynomial_almost_equal(groundtruth_res_poly, can_poly_res)

def test_mul():
    mi = MultiIndex.from_degree(2, 1, 2.0)
    canonical_coeffs1 = np.array([0, 1, 1])
    canonical_coeffs2 = np.array([0, 1, -1])

    can_poly_1 = CanonicalPolynomial(canonical_coeffs1, mi)
    can_poly_2 = CanonicalPolynomial(canonical_coeffs2, mi)

    c2l_transformation = TransformationCanonicalToLagrange(can_poly_1)

    lagrange_poly1 = c2l_transformation(can_poly_1)
    lagrange_poly2 = c2l_transformation(can_poly_2)

    res_lagrange_poly = lagrange_poly1 * lagrange_poly2

    l2c_transformation = TransformationLagrangeToCanonical(res_lagrange_poly)
    res_canonical_poly = l2c_transformation()

    groundtruth_product_coeffs = np.array([0, 0, 1, 0, 0, -1])
    groundtruth_mi = MultiIndex.from_degree(2, 2, 4.0)
    groundtruth = CanonicalPolynomial(groundtruth_product_coeffs, groundtruth_mi)

    assert_polynomial_almost_equal(groundtruth, res_canonical_poly)
