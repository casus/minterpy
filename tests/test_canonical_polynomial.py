"""
testing module for canonical_polynomial.py

The subclassing is not tested here, see tesing module `test_polynomial.py`
"""
import operator as op

import numpy as np
import pytest
from conftest import (
    SEED,
    LpDegree,
    MultiIndices,
    NrPoints,
    NrSimilarPolynomials,
    PolyDegree,
    SpatialDimension,
    assert_polynomial_almost_equal,
    build_rnd_coeffs,
    build_rnd_points,
)
from numpy.testing import assert_, assert_almost_equal

from minterpy import CanonicalPolynomial, MultiIndexSet

# tests with a single polynomial


def test_neg(MultiIndices, NrSimilarPolynomials):
    coeffs = build_rnd_coeffs(MultiIndices, NrSimilarPolynomials)
    poly = CanonicalPolynomial(MultiIndices, coeffs)
    res = -poly
    groundtruth_coeffs = (-1) * coeffs
    groundtruth = poly.__class__(MultiIndices, groundtruth_coeffs)
    assert_polynomial_almost_equal(res, groundtruth)


def test_eval(MultiIndices, NrPoints):
    coeffs = build_rnd_coeffs(MultiIndices)
    poly = CanonicalPolynomial(MultiIndices, coeffs)
    pts = build_rnd_points(NrPoints, MultiIndices.spatial_dimension)
    res = poly(pts)
    groundtruth = np.dot(
        np.prod(np.power(pts[:, None, :], MultiIndices.exponents[None, :, :]), axis=-1),
        coeffs,
    )
    assert_almost_equal(res, groundtruth)


# tests with two polynomials
# todo:: find out if there are some more sophisticated tests for that
exps1 = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
args1 = np.lexsort(exps1.T, axis=-1)
mi1 = MultiIndexSet(exps1[args1])
coeffs1 = np.array([1, 2, 3, 4])

exps2 = np.array([[0, 0, 0], [2, 0, 0], [1, 1, 0], [0, 2, 0], [0, 0, 2]])
args2 = np.lexsort(exps2.T, axis=-1)
mi2 = MultiIndexSet(exps2[args2])
coeffs2 = np.array([1, 2, 3, 4, 5])

polys = [
    CanonicalPolynomial(mi1, coeffs1[args1]),
    CanonicalPolynomial(mi2, coeffs2[args2]),
]


@pytest.fixture(params=polys)
def Poly(request):
    return request.param


def test_sub_same_poly(Poly):
    res = Poly - Poly
    groundtruth_coeffs = np.zeros(Poly.coeffs.shape)
    groundtruth = Poly.__class__(Poly.multi_index, groundtruth_coeffs)
    assert_polynomial_almost_equal(res, groundtruth)


P1 = Poly
P2 = Poly


def test_add_different_poly(P1, P2):
    res = P1 + P2
    if P1 is P2:
        groundtruth_coeffs = P1.coeffs * 2
        groundtruth_multi_index_exponents = P1.multi_index.exponents
    else:
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
    groundtruth_multi_index = MultiIndexSet(groundtruth_multi_index_exponents)
    groundtruth = P1.__class__(groundtruth_multi_index, groundtruth_coeffs)
    assert_polynomial_almost_equal(res, groundtruth)


def test_sub_different_poly():
    """
    .. todo::
        - make this a bit better
    """
    res = polys[0] - polys[1]
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
    groundtruth_multi_index = MultiIndexSet(groundtruth_multi_index_exponents)
    groundtruth = polys[0].__class__(groundtruth_multi_index, groundtruth_coeffs)
    assert_polynomial_almost_equal(res, groundtruth)


def test_partial_diff():

    # ATTENTION: the exponent vectors of all derivatives have to be included already!
    exponents = np.array([[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1],
                          [0, 1, 1],
                          [0, 0, 2]])
    coeffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert exponents.shape == (5, 3)
    assert coeffs.shape == (5,)

    mi = MultiIndexSet(exponents)
    can_poly = CanonicalPolynomial(mi, coeffs)

    groundtruth_coeffs_dx = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    groundtruth_coeffs_dy = np.array([2.0, 0.0, 4.0, 0.0, 0.0])
    groundtruth_coeffs_dz = np.array([3.0, 4.0, 10.0, 0.0, 0.0])

    can_poly_dx = can_poly.partial_diff(0)
    coeffs_dx = can_poly_dx.coeffs
    assert np.allclose(coeffs_dx, groundtruth_coeffs_dx)

    can_poly_dy = can_poly.partial_diff(1)
    coeffs_dy = can_poly_dy.coeffs
    assert np.allclose(coeffs_dy, groundtruth_coeffs_dy)

    can_poly_dz = can_poly.partial_diff(2)
    coeffs_dz = can_poly_dz.coeffs
    assert np.allclose(coeffs_dz, groundtruth_coeffs_dz)



def test_diff():

    # ATTENTION: the exponent vectors of all derivatives have to be included already!
    exponents = np.array([[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1],
                          [0, 1, 1],
                          [0, 0, 2]])
    coeffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert exponents.shape == (5, 3)
    assert coeffs.shape == (5,)

    mi = MultiIndexSet(exponents)
    can_poly = CanonicalPolynomial(mi, coeffs)

    # Testing zeroth order derivatives
    can_poly_zero_deriv = can_poly.diff([0,0,0])
    coeffs_zero_deriv = can_poly_zero_deriv.coeffs
    assert np.allclose(coeffs_zero_deriv, coeffs)

    groundtruth_coeffs_dyz = np.array([4.0, 0.0, 0.0, 0.0, 0.0])
    groundtruth_coeffs_dz2 = np.array([10.0, 0.0, 0.0, 0.0, 0.0])
    groundtruth_coeffs_dyz2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    can_poly_dyz = can_poly.diff([0,1,1])
    coeffs_dyz = can_poly_dyz.coeffs
    assert np.allclose(coeffs_dyz, groundtruth_coeffs_dyz)

    can_poly_dz2 = can_poly.diff([0,0,2])
    coeffs_dz2 = can_poly_dz2.coeffs
    assert np.allclose(coeffs_dz2, groundtruth_coeffs_dz2)

    can_poly_dyz2 = can_poly.diff([0,1,2])
    coeffs_dyz2 = can_poly_dyz2.coeffs
    assert np.allclose(coeffs_dyz2, groundtruth_coeffs_dyz2)
