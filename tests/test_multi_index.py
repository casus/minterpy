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
from minterpy.core.utils import (
    find_match_between,
    get_exponent_matrix,
    get_poly_degree,
    is_lexicographically_complete,
)

# test initialization
def test_init_from_exponents(SpatialDimension, PolyDegree, LpDegree):
    exponents = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)
    assert_call(MultiIndexSet, exponents, lp_degree=LpDegree)


def test_init_fail_from_exponents():
    """Test if invalid parameter values raise the expected errors."""
    # --- Non-downward-closed multi-index set exponents
    spatial_dimension, poly_degree, lp_degree = (2, 1, 1)
    exponents = get_exponent_matrix(spatial_dimension, poly_degree, lp_degree)
    exponents[0] = 1
    assert_raises(ValueError, MultiIndexSet, exponents, lp_degree)

    # --- Invalid lp-degree
    exponents = get_exponent_matrix(3, 2, 2)  # arbitrary exponents
    # Zero
    assert_raises(ValueError, MultiIndexSet, exponents, 0.0)
    # Negative value
    assert_raises(ValueError, MultiIndexSet, exponents, -1.0)
    # Invalid type (e.g., string)
    assert_raises(TypeError, MultiIndexSet, exponents, "1.0")


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

    # --- Invalid lp-degree
    # Zero
    assert_raises(ValueError, MultiIndexSet.from_degree, 3, 2, 0.0)
    # Negative
    assert_raises(ValueError, MultiIndexSet.from_degree, 3, 2, -1.0)


# Test methods

def test_add_exponents(SpatialDimension, PolyDegree, LpDegree):
    """Test the add_exponents method of a MultiIndex instance.
    
    Notes
    -----
    - This is related to the fix for Issue #75.
    """
    
    # Create 2 exponents, one twice the polynomial degree of the other
    exponents_1 = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)
    exponents_2 = get_exponent_matrix(SpatialDimension, 2*PolyDegree, LpDegree)

    # Compute the set difference between the larger set and the smaller set
    exponents_diff = np.array(
        list(set(map(tuple, exponents_2)) - set(map(tuple, exponents_1)))
    )

    # Create the multi-index sets
    mi_1 = MultiIndexSet(exponents_1, lp_degree=LpDegree)
    mi_2 = MultiIndexSet(exponents_2, lp_degree=LpDegree)

    # Act: Add the exponents difference to the smaller multi-index set
    mi_added = mi_1.add_exponents(exponents_diff)

    # Assert: The added multi-index must be the same as the big one
    assert_multi_index_equal(mi_added, mi_2)

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


def test_attributes_incomplete_exponents(SpatialDimension, PolyDegree, LpDegree):
    """Test the attributes with an incomplete exponents for MultiIndexSet.
    
    Notes
    -----
    - This is related to the fix for Issue #66.
    """
    exponents = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)

    # Create an incomplete multi-index set
    # by adding a new exponent of higher degree
    exponents_incomplete = np.insert(
        exponents, len(exponents), exponents[-1]+2, axis=0)
    multi_index_incomplete = MultiIndexSet(
        exponents_incomplete, lp_degree=LpDegree)

    # Make sure the incomplete exponents are indeed incomplete
    assert_(not is_lexicographically_complete(exponents_incomplete))

    # Compute the reference polynomial degree given the multi-index set    
    poly_degree = get_poly_degree(exponents_incomplete, LpDegree)

    # Assertion of attributes
    assert_equal(exponents_incomplete, multi_index_incomplete.exponents)
    assert_(multi_index_incomplete.lp_degree == LpDegree)
    assert_(multi_index_incomplete.poly_degree == poly_degree)

    num_of_monomials_incomplete, dim_incomplete = exponents_incomplete.shape
    assert_(len(multi_index_incomplete) == num_of_monomials_incomplete)
    assert_(multi_index_incomplete.spatial_dimension == dim_incomplete)


@pytest.mark.parametrize("spatial_dimension", [1, 2, 3, 7])
@pytest.mark.parametrize("poly_degree", [1, 2, 3, 5])
def test_attributes_from_degree(spatial_dimension, poly_degree, LpDegree):
    """Test the resulting instances from from_degree() constructor.

    Notes
    -----
    - This test is included due to Issue #97. The known breaking cases:
      the spatial dimension 7 and polynomial degree 5
      are explicitly tested and only for this test.
      Otherwise, the test suite would be too time-consuming to run.
    """
    multi_index = MultiIndexSet.from_degree(
        spatial_dimension, poly_degree, LpDegree
    )

    # Assertions
    assert_(isinstance(multi_index, MultiIndexSet))
    assert_(multi_index.lp_degree == LpDegree)
    assert_(multi_index.poly_degree == poly_degree)

    dim = multi_index.exponents.shape[1]
    assert_(multi_index.spatial_dimension == spatial_dimension)
    assert_(dim == spatial_dimension)
