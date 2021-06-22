"""
testing module for canonical_polynomial.py

The subclassing is not tested here, see tesing module `test_polynomial.py`
"""
import operator as op


import numpy as np
import pytest
from conftest import SpatialDimension,PolyDegree,LpDegree,NrSimilarPolynomials, SEED,assert_polynomial_almost_equal,MultiIndices,build_rnd_coeffs,build_rnd_points
from numpy.testing import assert_,assert_almost_equal

from minterpy import CanonicalPolynomial, MultiIndex


# tests with a single polynomial

def test_neg(MultiIndices,NrSimilarPolynomials):
    coeffs = build_rnd_coeffs(MultiIndices,NrSimilarPolynomials)
    poly = CanonicalPolynomial(coeffs,MultiIndices)
    res = -poly
    groundtruth_coeffs = (-1) * coeffs
    groundtruth = poly.__class__(groundtruth_coeffs,MultiIndices)
    assert_polynomial_almost_equal(res,groundtruth)

nr_pts = [1,2]

@pytest.fixture(params = nr_pts)
def NrPoints(request):
    return request.param

def test_eval(MultiIndices,NrPoints):
    coeffs = build_rnd_coeffs(MultiIndices)
    poly = CanonicalPolynomial(coeffs,MultiIndices)
    pts = build_rnd_points(NrPoints,MultiIndices.spatial_dimension)
    res = poly(pts)
    groundtruth =np.dot(np.prod(np.power(pts[:,None,:],MultiIndices.exponents[None,:,:]),axis=-1),coeffs)
    assert_almost_equal(res,groundtruth)

# tests with two polynomials
# todo:: find out if there are some more sophisticated tests for that
exps1 = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
args1 = np.lexsort(exps1.T, axis=-1)
mi1 = MultiIndex(exps1[args1])
coeffs1 = np.array([1, 2, 3, 4])

exps2 = np.array([[0, 0, 0], [2, 0, 0], [1, 1, 0], [0, 2, 0], [0, 0, 2]])
args2 = np.lexsort(exps2.T, axis=-1)
mi2 = MultiIndex(exps2[args2])
coeffs2 = np.array([1, 2, 3, 4, 5])

polys = [CanonicalPolynomial(coeffs1[args1], mi1),CanonicalPolynomial(coeffs2[args2], mi2)]

@pytest.fixture(params = polys)
def Poly(request):
    return request.param


def test_sub_same_poly(Poly):
    res = Poly - Poly
    groundtruth_coeffs = np.zeros(Poly.coeffs.shape)
    groundtruth = Poly.__class__(groundtruth_coeffs,Poly.multi_index)
    assert_polynomial_almost_equal(res,groundtruth)


P1 = Poly
P2 = Poly

def test_add_different_poly(P1,P2):
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
    groundtruth_multi_index = MultiIndex(groundtruth_multi_index_exponents)
    groundtruth = P1.__class__(groundtruth_coeffs,groundtruth_multi_index)
    assert_polynomial_almost_equal(res,groundtruth)


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
    groundtruth_multi_index = MultiIndex(groundtruth_multi_index_exponents)
    groundtruth = polys[0].__class__(groundtruth_coeffs,groundtruth_multi_index)
    assert_polynomial_almost_equal(res,groundtruth)
