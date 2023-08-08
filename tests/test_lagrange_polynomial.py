"""
Testing module for lagrange_polynomial.py

"""
import numpy as np
import pytest

from conftest import (
    MultiIndices,
    NrPoints,
    NrSimilarPolynomials,
    assert_polynomial_almost_equal,
    build_rnd_coeffs,
    build_rnd_points,
)

from minterpy import CanonicalPolynomial, LagrangePolynomial, MultiIndexSet, Grid
from minterpy.transformations import (
    CanonicalToLagrange,
    LagrangeToCanonical,
    LagrangeToNewton,
)
from minterpy.core.utils import make_complete


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


def test_integrate_over_bounds_invalid_shape(
    SpatialDimension, PolyDegree, LpDegree
):
    """Test polynomial integration with bounds of invalid shape."""
    # Create a Lagrange polynomial
    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
    lag_coeffs = np.random.rand(len(mi))
    lag_poly = CanonicalPolynomial(mi, lag_coeffs)

    # Create bounds (outside the canonical domain of [-1, 1]^M)
    bounds = np.random.rand(SpatialDimension + 3, 2)
    bounds[:, 0] *= -1

    with pytest.raises(ValueError):
        lag_poly.integrate_over(bounds)


def test_integrate_over_bounds_invalid_domain(
    SpatialDimension, PolyDegree, LpDegree
):
    """Test polynomial integration with bounds of invalid domain."""
    # Create a Lagrange polynomial
    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
    lag_coeffs = np.random.rand(len(mi))
    lag_poly = CanonicalPolynomial(mi, lag_coeffs)

    # Create bounds (outside the canonical domain of [-1, 1]^M)
    bounds = 2 * np.ones((SpatialDimension, 2))
    bounds[:, 0] *= -1

    with pytest.raises(ValueError):
        lag_poly.integrate_over(bounds)


def test_integrate_over_bounds_equal(
    SpatialDimension, PolyDegree, LpDegree
):
    """Test polynomial integration with equal bounds (should be zero)."""
    # Create a Lagrange polynomial
    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
    lag_coeffs = np.random.rand(len(mi))
    lag_poly = LagrangePolynomial(mi, lag_coeffs)

    # Create bounds (one of them has lb == ub)
    bounds = np.random.rand(SpatialDimension, 2)
    bounds[:, 0] *= -1
    idx = np.random.choice(SpatialDimension)
    bounds[idx, 0] = bounds[idx, 1]

    # Compute the integral
    ref = 0.0
    value = lag_poly.integrate_over(bounds)

    # Assertion
    assert isinstance(value, float)
    assert np.isclose(ref, value)


def test_integrate_over_bounds_flipped(
    SpatialDimension, PolyDegree, LpDegree
):
    """Test polynomial integration with flipped bounds."""
    # Create a Lagrange polynomial
    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
    lag_coeffs = np.random.rand(len(mi))
    lag_poly = CanonicalPolynomial(mi, lag_coeffs)

    # Compute the integral
    value_1 = lag_poly.integrate_over()

    # Flip bounds
    bounds = np.ones((SpatialDimension, 2))
    bounds[:, 0] *= -1
    bounds[:, [0, 1]] = bounds[:, [1, 0]]

    # Compute the integral with flipped bounds
    value_2 = lag_poly.integrate_over(bounds)

    if np.mod(SpatialDimension, 2) == 1:
        # Odd spatial dimension flips the sign
        assert np.isclose(value_1, -1 * value_2)
    else:
        assert np.isclose(value_1, value_2)


def test_integrate_over_list_as_bounds(
    SpatialDimension, PolyDegree, LpDegree
):
    """Test integrate over with bounds specified with lists."""
    # Create a Lagrange polynomial
    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
    lag_coeffs = np.random.rand(len(mi))
    lag_poly = CanonicalPolynomial(mi, lag_coeffs)

    # Compute the integral
    value_1 = lag_poly.integrate_over()

    # Flip bounds
    bounds = [[-1, 1] for _ in range(SpatialDimension)]

    # Compute the integral with flipped bounds
    value_2 = lag_poly.integrate_over(bounds)

    # Assertion
    assert np.isclose(value_1, value_2)


def test_integrate_over(
    SpatialDimension, PolyDegree, LpDegree
):
    """Test polynomial integration in different basis (sanity check)."""
    # Create a Canonical polynomial
    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
    lag_coeffs = np.random.rand(len(mi))
    lag_poly = LagrangePolynomial(mi, lag_coeffs)

    # Transform to other polynomial bases
    nwt_poly = LagrangeToNewton(lag_poly)()
    can_poly = LagrangeToCanonical(lag_poly)()

    # Compute the integral
    value_lag = lag_poly.integrate_over()
    value_nwt = nwt_poly.integrate_over()
    # NOTE: Canonical integration won't work in high degree
    value_can = can_poly.integrate_over()

    # Assertions
    assert np.isclose(value_lag, value_nwt)
    assert np.isclose(value_lag, value_can)

    # Create bounds
    bounds = np.random.rand(SpatialDimension, 2)
    bounds[:, 0] *= -1

    # Compute the integral with bounds
    value_lag = lag_poly.integrate_over(bounds)
    value_nwt = nwt_poly.integrate_over(bounds)
    value_can = can_poly.integrate_over(bounds)

    # Assertions
    assert np.isclose(value_lag, value_nwt)
    assert np.isclose(value_lag, value_can)


def test_integrate_over_sum_function(
    SpatialDimension, PolyDegree, LpDegree
):
    """Test polynomial integration for a simple sum function."""
    # Create a Lagrange interpolating polynomial
    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
    grd = Grid(mi)
    lag_coeffs = np.sum(grd.unisolvent_nodes, axis=1)
    lag_poly = LagrangePolynomial(mi, lag_coeffs)

    # With the default bounds
    if PolyDegree > 0:
        ref = 0.0
        value = lag_poly.integrate_over()

        # Assertion
        assert np.isclose(ref, value)

    # With non-symmetric bounds (non-cancelling)
    bounds = np.random.rand(SpatialDimension, 2)
    bounds[:, 0] *= -1

    if PolyDegree > 0:
        # Rhe reference from analytical results for non-cancelling bounds
        ref = 0.0
        for i in range(SpatialDimension):
            ref += (
                np.diff(bounds[i] ** 2)
                * np.prod(np.diff(np.delete(bounds, i, axis=0)))
            )
        ref *= 0.5
        value = lag_poly.integrate_over(bounds)

        # Assertion
        assert np.isclose(ref, value)


def test_integrate_over_product_function(
    SpatialDimension, LpDegree,
):
    """Test polynomial integration for a simple product function."""
    fun = lambda xx: np.prod(xx, axis=1)

    # Create a Lagrange interpolating polynomial
    exp = np.ones((1, SpatialDimension), dtype=int)
    exp_completed = make_complete(exp, LpDegree)
    mi = MultiIndexSet(exp_completed, LpDegree)
    grd = Grid(mi)
    lag_coeffs = fun(grd.unisolvent_nodes)
    lag_poly = LagrangePolynomial(mi, lag_coeffs)

    # --- With the default bounds

    # Compute the integral without bounds
    ref = 0.0
    value = lag_poly.integrate_over()

    # Assertion
    assert np.isclose(ref, value)

    # --- With non-symmetric bounds (non-cancelling)

    # Set up bounds
    bounds = np.random.rand(SpatialDimension, 2)
    bounds[:, 0] *= -1

    # Compute the integral with bounds
    ref = 0.5**SpatialDimension * np.prod(np.diff(bounds**2))
    value = lag_poly.integrate_over(bounds)

    # Assertion
    assert np.isclose(ref, value)


def test_integrate_over_multiple_polynomials(
    SpatialDimension, LpDegree,
):
    """Test integration with multiple polynomials."""
    num_polys = 6
    factors = np.arange(1, num_polys + 1)[np.newaxis, :]
    fun = lambda xx: np.prod(xx, axis=1)[:, np.newaxis] * factors

    # Create a Lagrange interpolating polynomial
    exp = np.ones((1, SpatialDimension), dtype=int)
    exp_completed = make_complete(exp, LpDegree)
    mi = MultiIndexSet(exp_completed, LpDegree)
    grd = Grid(mi)
    lag_coeffs = fun(grd.unisolvent_nodes)
    lag_poly = LagrangePolynomial(mi, lag_coeffs)

    # --- With the default bounds

    # Compute the integral without bounds
    ref = 0.0
    value = lag_poly.integrate_over()

    # Assertion
    assert len(value) == num_polys
    assert np.allclose(ref, value)

    # --- With non-symmetric bounds (non-cancelling)

    # Set up bounds
    bounds = np.random.rand(SpatialDimension, 2)
    bounds[:, 0] *= -1

    # Compute the integral with bounds
    ref = factors * 0.5 ** SpatialDimension * np.prod(np.diff(bounds ** 2))
    value = lag_poly.integrate_over(bounds)

    # Assertion
    assert len(value) == num_polys
    assert np.allclose(ref, value)
