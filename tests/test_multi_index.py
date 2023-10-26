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
from copy import copy
from numpy.testing import assert_, assert_equal, assert_raises

from minterpy import MultiIndexSet
from minterpy.core.utils import (
    get_poly_degree,
    get_exponent_matrix,
    find_match_between,
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

def test_add_exponents_outplace(SpatialDimension, PolyDegree, LpDegree):
    """Test the add_exponents method of a MultiIndex instance outplace.
    
    Notes
    -----
    - This is related to the fix for Issue #75
      and the refactoring of Issue #117.
    """

    # --- Already contained exponents (should return identical exponents)

    # Create a set of exponents
    exponents = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)
    mi = MultiIndexSet(exponents, lp_degree=LpDegree)

    # Add the same set of exponents with default parameter value
    mi_added = mi.add_exponents(exponents)
    # Assertions
    assert_multi_index_equal(mi_added, mi)
    assert mi_added.exponents is exponents

    # Add the same set of exponents with parameter
    mi_added = mi.add_exponents(exponents, inplace=False)
    # Assertions
    assert_multi_index_equal(mi_added, mi)
    assert mi_added.exponents is exponents

    # --- A new set of exponents
    
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

    # Add the exponents difference with default parameter value
    mi_added = mi_1.add_exponents(exponents_diff)
    # Assertion: The added multi-index must be the same as the big one
    assert_multi_index_equal(mi_added, mi_2)

    # Add the exponents difference with parameter
    mi_added = mi_1.add_exponents(exponents_diff, inplace=False)
    # Assertion: The added multi-index must be the same as the big one
    assert_multi_index_equal(mi_added, mi_2)


def test_add_exponents_inplace(SpatialDimension, PolyDegree, LpDegree):
    """Test in-place add_exponents() on the MultiIndexSet instances.

    Notes
    -----
    - This test is related to the refactoring of Issue #117.
    """

    # --- Already contained exponents (should return identical exponents)

    # Create a set of exponents
    exponents = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)
    mi = MultiIndexSet(exponents, lp_degree=LpDegree)
    poly_degree = mi.poly_degree

    # Add the same set of exponents in-place
    mi.add_exponents(exponents, inplace=True)
    # Assertions
    assert mi.exponents is exponents
    assert mi.poly_degree == poly_degree

    # --- New set of exponents

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

    # Add the exponents difference in-place
    mi_1.add_exponents(exponents_diff, inplace=True)
    # Assertion: The added multi-index must be the same as the big one
    assert_multi_index_equal(mi_1, mi_2)


def test_add_exponents_sparse():
    """Test adding exponents to a high-dimensional sparse multi-index set.

    Notes
    -----
    - This test is related to Issue #81.
    """
    # Create a 20-dimensional but sparse exponents
    m = 20
    exponents = np.zeros((m, m), dtype=int)
    exponents[:, 0] = np.arange(m)
    mi = MultiIndexSet(exponents, lp_degree=1.0)

    # Assertion
    assert mi.is_complete

    # Add a new (sparse) element
    new_element = np.zeros(m)
    new_element[0] = m
    mi_added = mi.add_exponents(new_element)

    # Assertions
    assert mi_added.is_complete
    assert mi_added.poly_degree == mi.poly_degree + 1


def test_attributes(SpatialDimension, PolyDegree, LpDegree):
    """Test the attributes of MultiIndexSet instances."""

    # Create a MultiIndexSet from a set of exponents
    exponents = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)
    multi_index = MultiIndexSet(exponents, lp_degree=LpDegree)

    # Assertions
    assert_(isinstance(multi_index, MultiIndexSet))
    assert_equal(exponents, multi_index.exponents)
    assert_(multi_index.lp_degree == LpDegree)
    assert_(multi_index.poly_degree == PolyDegree)

    number_of_monomials, dim = exponents.shape
    assert_(len(multi_index) == number_of_monomials)
    assert_(multi_index.spatial_dimension == dim)

    # Assigning to read-only properties
    with pytest.raises(AttributeError):
        # This is related to Issue #98
        multi_index.lp_degree = LpDegree
        # This is related to Issue #100
        multi_index.poly_degree = PolyDegree


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


def test_make_complete_inplace(SpatialDimension, PolyDegree, LpDegree):
    """Test in-place make_complete() on the MultiIndexSet instances."""
    if PolyDegree < 1:
        # Only test for polynomial degree of higher than 0 (0 always complete)
        return

    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

    # Get the highest multi-index set element
    exponent = np.atleast_2d(mi.exponents[-1])

    # Create a new instance with just a single exponent (incomplete exponent)
    mi_incomplete = MultiIndexSet(exponent, LpDegree)
    assert not mi_incomplete.is_complete

    # Make complete in-place
    mi_incomplete.make_complete(inplace=True)

    # Assertions
    assert mi_incomplete.is_complete
    assert np.all(mi.exponents == mi_incomplete.exponents)


def test_make_complete_outplace(SpatialDimension, PolyDegree, LpDegree):
    """Test out-place make_complete() on the MultiIndexSet instances.

    Notes
    ----
    - This is test is also related to Issue #115; by default, the 'inplace' 
      parameter is set to False and a new MultiIndexSet instance is created.
    """
    if PolyDegree < 1:
        # Only test for polynomial degree of higher than 0 (0 always complete)
        return

    # NOTE: By construction, 'from_degree()' returns a complete set
    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

    # --- Already complete set of exponents (a shallow copy is created)
    mi_complete = mi.make_complete(inplace=False)
    assert mi_complete.exponents is mi.exponents

    # --- Incomplete set of exponents

    # Get the highest multi-index set element
    exponent = np.atleast_2d(mi.exponents[-1])

    # Create a new instance with just a single exponent
    mi_incomplete = MultiIndexSet(exponent, LpDegree)
    assert not mi_incomplete.is_complete

    # Make complete out-place (with the default parameter)
    mi_complete = mi_incomplete.make_complete()

    # Assertions
    assert mi_complete.is_complete
    assert mi_complete is not mi_incomplete
    assert np.all(mi_complete.exponents == mi.exponents)

    # Make complete out-place (with an explicit argument)
    mi_complete = mi_incomplete.make_complete(inplace=False)

    # Assertions
    assert mi_complete.is_complete
    assert mi_complete is not mi_incomplete
    assert np.all(mi_complete.exponents == mi.exponents)
