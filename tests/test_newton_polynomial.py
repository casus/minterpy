"""
testing module for canonical_polynomial.py

The subclassing is not tested here, see tesing module `test_polynomial.py`
"""

import numpy as np
import pytest
from conftest import SpatialDimension,PolyDegree,LpDegree,NrSimilarPolynomials,NrPoints, SEED,assert_polynomial_almost_equal,MultiIndices,build_rnd_coeffs,build_rnd_points
from numpy.testing import assert_,assert_almost_equal

from minterpy import NewtonToCanonical,NewtonPolynomial

def test_eval(MultiIndices,NrPoints):
    coeffs = build_rnd_coeffs(MultiIndices)
    poly = NewtonPolynomial(MultiIndices, coeffs)
    pts = build_rnd_points(NrPoints,MultiIndices.spatial_dimension)
    res = poly(pts)

    trafo_n2c = NewtonToCanonical(poly)
    canon_poly = trafo_n2c()
    groundtruth = canon_poly(pts)
    assert_almost_equal(res,groundtruth)
