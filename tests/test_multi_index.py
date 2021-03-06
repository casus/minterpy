"""
Test suite for multi_index.py
"""
import numpy as np
import pytest
from conftest import (
    LpDegree,
    PolyDegree,
    SpatialDimension,
    assert_call,
    assert_multi_index_equal,
)
from numpy.testing import assert_, assert_equal, assert_raises

from minterpy import MultiIndexSet
from minterpy.core.utils import find_match_between, get_exponent_matrix


# test initialization
def test_init_from_exponents(SpatialDimension, PolyDegree, LpDegree):
    exponents = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)
    assert_call(MultiIndexSet, exponents)
    assert_call(MultiIndexSet, exponents, lp_degree=LpDegree)


def test_init_fail_from_exponents():
    exponents = get_exponent_matrix(2, 1, 1)
    exponents[0] = 1
    assert_raises(ValueError, MultiIndexSet, exponents)


def test_init_from_degree(SpatialDimension, PolyDegree, LpDegree):
    assert_call(MultiIndexSet.from_degree, SpatialDimension, PolyDegree)
    assert_call(
        MultiIndexSet.from_degree, SpatialDimension, PolyDegree, lp_degree=LpDegree
    )
    multi_index = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

    exponents = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)
    groundtruth = MultiIndexSet(exponents, lp_degree=LpDegree)
    assert_multi_index_equal(multi_index, groundtruth)


def test_init_fail_from_degree():
    assert_raises(TypeError, MultiIndexSet.from_degree, 1.0, 1)
    assert_raises(TypeError, MultiIndexSet.from_degree, 1, 1.0)


# test attributes


def test_attributes(SpatialDimension, PolyDegree, LpDegree):
    exponents = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)
    multi_index = MultiIndexSet(exponents, lp_degree=LpDegree)
    assert_(isinstance(multi_index, MultiIndexSet))
    assert_equal(exponents, multi_index.exponents)
    assert_(multi_index.lp_degree == LpDegree)
    assert_(multi_index.poly_degree == PolyDegree)

    number_of_monomials, dim = exponents.shape
    assert_(len(multi_index) == number_of_monomials)
    assert_(multi_index.spatial_dimension == dim)
