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

from minterpy import (
    CanonicalPolynomial,
    CanonicalToLagrange,
    CanonicalToNewton,
    Grid,
    MultiIndexSet,
)

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

    # navie impementation of canonical eval
    # related to issue #32
    groundtruth = np.zeros(NrPoints)
    for k,pt in enumerate(pts):
        single_groundtruth = 0.0
        for i,exponents in enumerate(MultiIndices.exponents):
            term = 1.0
            for j, expo in enumerate(exponents):
                term *= pt[j]**expo
            single_groundtruth+=coeffs[i]*term
        groundtruth[k] = single_groundtruth

    assert_almost_equal(res, groundtruth)


# tests with two polynomials
# todo:: find out if there are some more sophisticated tests for that
exps1 = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
args1 = np.lexsort(exps1.T, axis=-1)
mi1 = MultiIndexSet(exps1[args1], lp_degree=1.0)
coeffs1 = np.array([1, 2, 3, 4])

exps2 = np.array([[0, 0, 0], [2, 0, 0], [1, 1, 0], [0, 2, 0], [0, 0, 2]])
args2 = np.lexsort(exps2.T, axis=-1)
mi2 = MultiIndexSet(exps2[args2], lp_degree=1.0)
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
    groundtruth_multi_index = MultiIndexSet(groundtruth_multi_index_exponents, lp_degree=1.0)
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
    groundtruth_multi_index = MultiIndexSet(groundtruth_multi_index_exponents, lp_degree=1.0)
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

    mi = MultiIndexSet(exponents, lp_degree=1.0)
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

    mi = MultiIndexSet(exponents, lp_degree=1.0)
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


def test_partial_diff_multiple_poly():

    # ATTENTION: the exponent vectors of all derivatives have to be included already!
    exponents = np.array([[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1],
                          [0, 1, 1],
                          [0, 0, 2]])
    coeffs = np.array([[3.0, 3.0, 3.0, 3.0, 3.0],
                       [5.0, 4.0, 3.0, 2.0, 1.0]]).T

    assert exponents.shape == (5, 3)
    assert coeffs.shape == (5,2)

    mi = MultiIndexSet(exponents, lp_degree=1.0)
    can_poly = CanonicalPolynomial(mi, coeffs)

    groundtruth_coeffs_dx = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0]]).T
    groundtruth_coeffs_dy = np.array([[3.0, 0.0, 3.0, 0.0, 0.0],
                                      [4.0, 0.0, 2.0, 0.0, 0.0]]).T
    groundtruth_coeffs_dz = np.array([[3.0, 3.0, 6.0, 0.0, 0.0],
                                      [3.0, 2.0, 2.0, 0.0, 0.0]]).T

    can_poly_dx = can_poly.partial_diff(0)
    coeffs_dx = can_poly_dx.coeffs
    assert np.allclose(coeffs_dx, groundtruth_coeffs_dx)

    can_poly_dy = can_poly.partial_diff(1)
    coeffs_dy = can_poly_dy.coeffs
    assert np.allclose(coeffs_dy, groundtruth_coeffs_dy)

    can_poly_dz = can_poly.partial_diff(2)
    coeffs_dz = can_poly_dz.coeffs
    assert np.allclose(coeffs_dz, groundtruth_coeffs_dz)


def test_integrate_over_bounds_invalid_shape(
    SpatialDimension, PolyDegree, LpDegree
):
    """Test polynomial integration with bounds of invalid shape."""
    # Create a Canonical polynomial
    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
    can_coeffs = np.random.rand(len(mi))
    can_poly = CanonicalPolynomial(mi, can_coeffs)

    # Create bounds (outside the canonical domain of [-1, 1]^M)
    bounds = np.random.rand(SpatialDimension + 3, 2)
    bounds[:, 0] *= -1

    with pytest.raises(ValueError):
        can_poly.integrate_over(bounds)


def test_integrate_over_bounds_invalid_domain(
    SpatialDimension, PolyDegree, LpDegree
):
    """Test polynomial integration with bounds of invalid domain."""
    # Create a Canonical polynomial
    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
    can_coeffs = np.random.rand(len(mi))
    can_poly = CanonicalPolynomial(mi, can_coeffs)

    # Create bounds (outside the canonical domain of [-1, 1]^M)
    bounds = 2 * np.ones((SpatialDimension, 2))
    bounds[:, 0] *= -1

    with pytest.raises(ValueError):
        can_poly.integrate_over(bounds)


def test_integrate_over_bounds_equal(
    SpatialDimension, PolyDegree, LpDegree
):
    """Test polynomial integration with equal bounds (should be zero)."""
    # Create a Canonical polynomial
    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
    can_coeffs = np.random.rand(len(mi))
    can_poly = CanonicalPolynomial(mi, can_coeffs)

    # Create bounds (one of them has lb == ub)
    bounds = np.random.rand(SpatialDimension, 2)
    bounds[:, 0] *= -1
    idx = np.random.choice(SpatialDimension)
    bounds[idx, 0] = bounds[idx, 1]

    # Compute the integral
    ref = 0.0
    value = can_poly.integrate_over(bounds)

    # Assertion
    assert np.isclose(ref, value)


def test_integrate_over_bounds_flipped(
    SpatialDimension, PolyDegree, LpDegree
):
    """Test polynomial integration with flipped bounds."""
    # Create a Canonical polynomial
    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
    can_coeffs = np.random.rand(len(mi))
    can_poly = CanonicalPolynomial(mi, can_coeffs)

    # Compute the integral
    value_1 = can_poly.integrate_over()

    # Flip bounds
    bounds = np.ones((SpatialDimension, 2))
    bounds[:, 0] *= -1
    bounds[:, [0, 1]] = bounds[:, [1, 0]]

    # Compute the integral with flipped bounds
    value_2 = can_poly.integrate_over(bounds)

    if np.mod(SpatialDimension, 2) == 1:
        # Odd spatial dimension flips the sign
        assert np.isclose(value_1, -1 * value_2)
    else:
        assert np.isclose(value_1, value_2)


def test_integrate_over(
    SpatialDimension, PolyDegree, LpDegree
):
    """Test polynomial integration in different basis (sanity check)."""
    # Create a Canonical polynomial
    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
    can_coeffs = np.random.rand(len(mi))
    can_poly = CanonicalPolynomial(mi, can_coeffs)

    # Transform to other polynomial bases
    nwt_poly = CanonicalToNewton(can_poly)()
    lag_poly = CanonicalToLagrange(can_poly)()

    # Compute the integral
    # NOTE: Canonical integration won't work in high degree
    value_can = can_poly.integrate_over()
    value_nwt = nwt_poly.integrate_over()
    value_lag = lag_poly.integrate_over()

    # Assertions
    assert np.isclose(value_can, value_nwt)
    assert np.isclose(value_can, value_lag)

    # Create bounds
    bounds = np.random.rand(SpatialDimension, 2)
    bounds[:, 0] *= -1

    # Compute the integral with bounds
    value_can = can_poly.integrate_over(bounds)
    value_nwt = nwt_poly.integrate_over(bounds)
    value_lag = lag_poly.integrate_over(bounds)

    # Assertions
    assert np.isclose(value_can, value_nwt)
    assert np.isclose(value_can, value_lag)


def test_integrate_over_multiple_polynomials(
    SpatialDimension, PolyDegree, LpDegree
):
    """Test polynomial integration in different basis (sanity check)."""
    # Create a set of Canonical polynomials
    num_polys = 6
    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
    can_coeffs = np.random.rand(len(mi), num_polys)
    can_poly = CanonicalPolynomial(mi, can_coeffs)

    # Transform to other polynomial bases
    nwt_poly = CanonicalToNewton(can_poly)()
    lag_poly = CanonicalToLagrange(can_poly)()

    # Compute the integral
    # NOTE: Canonical integration won't work in high degree
    value_can = can_poly.integrate_over()
    value_nwt = nwt_poly.integrate_over()
    value_lag = lag_poly.integrate_over()

    # Assertions
    assert np.allclose(value_can, value_nwt)
    assert np.allclose(value_can, value_lag)

    # Create bounds
    bounds = np.random.rand(SpatialDimension, 2)
    bounds[:, 0] *= -1

    # Compute the integral with bounds
    value_can = can_poly.integrate_over(bounds)
    value_nwt = nwt_poly.integrate_over(bounds)
    value_lag = lag_poly.integrate_over(bounds)

    # Assertions
    assert np.allclose(value_can, value_nwt)
    assert np.allclose(value_can, value_lag)
