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
    NrSimilarPolynomials,
    PolyDegree,
    SpatialDimension,
    assert_polynomial_almost_equal,
    build_rnd_coeffs,
    build_rnd_points,
    build_random_newton_polynom,
)
from numpy.testing import assert_, assert_almost_equal
from minterpy.global_settings import INT_DTYPE

from minterpy import NewtonPolynomial, NewtonToCanonical, LagrangeToCanonical, LagrangeToNewton


def test_eval(MultiIndices, NrPoints):
    coeffs = build_rnd_coeffs(MultiIndices)
    poly = NewtonPolynomial(MultiIndices, coeffs)
    pts = build_rnd_points(NrPoints, MultiIndices.spatial_dimension)
    res = poly(pts)

    trafo_n2c = NewtonToCanonical(poly)
    canon_poly = trafo_n2c()
    groundtruth = canon_poly(pts)
    assert_almost_equal(res, groundtruth)

def test_partial_derivative(SpatialDimension, PolyDegree, LpDegree):
    newton_poly = build_random_newton_polynom(SpatialDimension, PolyDegree, LpDegree)

    # Check partial derivative on each dimension by comparing it with Canonical partial derivatives
    for dim in range(SpatialDimension):
        trafo_n2c = NewtonToCanonical(newton_poly)
        canon_poly = trafo_n2c()
        can_deriv_poly = canon_poly.partial_derivative(dim)

        lag_deriv_poly = newton_poly.partial_derivative(dim)
        trafo_l2c = LagrangeToCanonical(lag_deriv_poly)
        can_newt_deriv_poly = trafo_l2c()

        assert_polynomial_almost_equal(can_newt_deriv_poly, can_deriv_poly)

def test_derivative(SpatialDimension, PolyDegree, LpDegree):
    newton_poly = build_random_newton_polynom(SpatialDimension, PolyDegree, LpDegree)

    # A derivative of order zero along all dimensions should be equivalent to the same polynomial
    zero_order_deriv_lagr = newton_poly.derivative(np.zeros(SpatialDimension, dtype=INT_DTYPE))
    trafo_l2n = LagrangeToNewton(zero_order_deriv_lagr)
    zero_order_deriv_newt = trafo_l2n()

    assert_polynomial_almost_equal(zero_order_deriv_newt, newton_poly)

    # Comparing the gradient with that computed in Canonical basis
    trafo_n2c = NewtonToCanonical(newton_poly)
    canon_poly = trafo_n2c()
    deriv_order = np.ones(SpatialDimension, dtype=INT_DTYPE)
    can_deriv_poly = canon_poly.derivative(deriv_order)

    lag_deriv_poly = newton_poly.derivative(deriv_order)
    trafo_l2c = LagrangeToCanonical(lag_deriv_poly)
    can_newt_deriv_poly = trafo_l2c()

    assert_polynomial_almost_equal(can_newt_deriv_poly, can_deriv_poly)