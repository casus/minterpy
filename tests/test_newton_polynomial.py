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

from minterpy import NewtonPolynomial, NewtonToCanonical, CanonicalToNewton


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

def test_diff(SpatialDimension, PolyDegree, LpDegree):
    newton_poly = build_random_newton_polynom(SpatialDimension, PolyDegree, LpDegree)

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
