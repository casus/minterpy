"""
testing module for canonical_polynomial.py

The subclassing is not tested here, see tesing module `test_polynomial.py`
"""
import numpy as np
import pytest
from conftest import (
    SEED,
    LpDegree,
    MultiIndices,
    NrPoints,
    NrPolynomials,
    NrSimilarPolynomials,
    PolyDegree,
    SpatialDimension,
    BatchSizes,
    assert_polynomial_almost_equal,
    build_rnd_coeffs,
    build_rnd_points,
    build_random_newton_polynom,
)
from numpy.testing import assert_, assert_almost_equal
from minterpy.global_settings import INT_DTYPE
from minterpy.utils import eval_newton_polynomials
from minterpy import Grid

from minterpy import NewtonPolynomial, NewtonToCanonical, CanonicalToNewton, NewtonToLagrange


def test_eval(MultiIndices, NrPoints, NrPolynomials):
    """Test the evaluation of Newton polynomials."""

    coeffs = build_rnd_coeffs(MultiIndices, NrPolynomials)
    poly = NewtonPolynomial(MultiIndices, coeffs)
    pts = build_rnd_points(NrPoints, MultiIndices.spatial_dimension)

    # Evaluate
    res = poly(pts)

    trafo_n2c = NewtonToCanonical(poly)
    canon_poly = trafo_n2c()
    groundtruth = canon_poly(pts)
    assert_almost_equal(res, groundtruth)


def test_eval_batch(MultiIndices, NrPolynomials, BatchSizes):
    """Test the evaluation on Newton polynomials in batches of query points."""

    #TODO: This is a temporary test as the 'batch_size' parameter is not
    #      opened in the higher-level interface, i.e., 'newton_poly(xx)'

    # Create a random coefficient values
    newton_coeffs = build_rnd_coeffs(MultiIndices, NrPolynomials)
    grid = Grid(MultiIndices)
    generating_points = grid.generating_points
    exponents = MultiIndices.exponents

    # Create test query points
    xx = build_rnd_points(421, MultiIndices.spatial_dimension)

    # Evaluate the polynomial in batches
    yy_newton = eval_newton_polynomials(
        xx, newton_coeffs, exponents, generating_points, batch_size=BatchSizes
    )

    # Create a reference results from canonical polynomial evaluation
    newton_poly = NewtonPolynomial(MultiIndices, newton_coeffs)
    canonical_poly = NewtonToCanonical(newton_poly)()
    yy_canonical = canonical_poly(xx)

    # Assert
    assert_almost_equal(yy_newton, yy_canonical)


def test_partial_diff(SpatialDimension, PolyDegree, LpDegree):
    newton_poly = build_random_newton_polynom(SpatialDimension, PolyDegree, LpDegree)

    # Check partial derivative on each dimension by comparing it with Canonical partial derivatives
    for dim in range(SpatialDimension):
        trafo_n2c = NewtonToCanonical(newton_poly)
        canon_poly = trafo_n2c()
        can_diff_poly = canon_poly.partial_diff(dim)
        trafo_c2n = CanonicalToNewton(can_diff_poly)
        newt_can_diff_poly = trafo_c2n()

        newt_diff_poly = newton_poly.partial_diff(dim)

        assert_polynomial_almost_equal(newt_can_diff_poly, newt_diff_poly)

def test_diff(SpatialDimension, PolyDegree, LpDegree, NrPolynomials):
    newton_poly = build_random_newton_polynom(SpatialDimension, PolyDegree, LpDegree, NrPolynomials)

    # A derivative of order zero along all dimensions should be equivalent to the same polynomial
    zero_order_diff_newt = newton_poly.diff(np.zeros(SpatialDimension, dtype=INT_DTYPE))
    assert_polynomial_almost_equal(zero_order_diff_newt, newton_poly)

    # Comparing the gradient with that computed in Canonical basis
    trafo_n2c = NewtonToCanonical(newton_poly)
    canon_poly = trafo_n2c()
    diff_order = np.ones(SpatialDimension, dtype=INT_DTYPE)
    can_diff_poly = canon_poly.diff(diff_order)
    trafo_c2n = CanonicalToNewton(can_diff_poly)
    newt_can_diff_poly = trafo_c2n()

    newt_diff_poly = newton_poly.diff(diff_order)

    assert_polynomial_almost_equal(newt_can_diff_poly, newt_diff_poly)


def test_integrate_over_bounds_invalid_shape(
    SpatialDimension, PolyDegree, LpDegree
):
    """Test polynomial integration with bounds of invalid shape."""
    # Create a Newton polynomial
    nwt_poly = build_random_newton_polynom(
        SpatialDimension, PolyDegree, LpDegree
    )

    # Create bounds (outside the canonical domain of [-1, 1]^M)
    bounds = np.random.rand(SpatialDimension + 3, 2)
    bounds[:, 0] *= -1

    with pytest.raises(ValueError):
        nwt_poly.integrate_over(bounds)


def test_integrate_over_bounds_invalid_domain(
    SpatialDimension, PolyDegree, LpDegree
):
    """Test polynomial integration with bounds of invalid domain."""
    # Create a Newton polynomial
    nwt_poly = build_random_newton_polynom(
        SpatialDimension, PolyDegree, LpDegree
    )

    # Create bounds (outside the canonical domain of [-1, 1]^M)
    bounds = 2 * np.ones((SpatialDimension, 2))
    bounds[:, 0] *= -1

    with pytest.raises(ValueError):
        nwt_poly.integrate_over(bounds)


def test_integrate_over_bounds_equal(
    SpatialDimension, PolyDegree, LpDegree
):
    """Test polynomial integration with equal bounds (should be zero)."""
    # Create a Newton polynomial
    nwt_poly = build_random_newton_polynom(
        SpatialDimension, PolyDegree, LpDegree
    )

    # Create bounds (one of them has lb == ub)
    bounds = np.random.rand(SpatialDimension, 2)
    bounds[:, 0] *= -1
    idx = np.random.choice(SpatialDimension)
    bounds[idx, 0] = bounds[idx, 1]

    # Compute the integral
    ref = 0.0
    value = nwt_poly.integrate_over(bounds)

    # Assertion
    assert np.isclose(ref, value)


def test_integrate_over_bounds_flipped(
    SpatialDimension, PolyDegree, LpDegree
):
    """Test polynomial integration with specified and valid bounds."""
    # Create a Newton polynomial
    nwt_poly = build_random_newton_polynom(
        SpatialDimension, PolyDegree, LpDegree
    )

    # Compute the integral
    value_1 = nwt_poly.integrate_over()

    # Flip bounds
    bounds = np.ones((SpatialDimension, 2))
    bounds[:, 0] *= -1
    bounds[:, [0, 1]] = bounds[:, [1, 0]]

    # Compute the integral with flipped bounds
    value_2 = nwt_poly.integrate_over(bounds)

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
    nwt_poly = build_random_newton_polynom(
        SpatialDimension, PolyDegree, LpDegree
    )

    # Transform to other polynomial bases
    lag_poly = NewtonToLagrange(nwt_poly)()
    can_poly = NewtonToCanonical(nwt_poly)()

    # Compute the integral
    value_nwt = nwt_poly.integrate_over()
    value_lag = lag_poly.integrate_over()
    # NOTE: Canonical integration won't work in high degree
    value_can = can_poly.integrate_over()

    # Assertions
    assert np.isclose(value_nwt, value_lag)
    assert np.isclose(value_nwt, value_can)

    # Create bounds
    bounds = np.random.rand(SpatialDimension, 2)
    bounds[:, 0] *= -1

    # Compute the integral with bounds
    value_lag = lag_poly.integrate_over(bounds)
    value_nwt = nwt_poly.integrate_over(bounds)
    value_can = can_poly.integrate_over(bounds)

    # Assertions
    assert np.isclose(value_nwt, value_lag)
    assert np.isclose(value_nwt, value_can)
