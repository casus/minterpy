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
    multiply_indices,
)


# test initialization
def test_init_from_exponents(SpatialDimension, PolyDegree, LpDegree):
    """Test the default constructor of MultiIndexSet."""
    # --- Complete exponents
    exponents = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)
    # Assertion
    assert_call(MultiIndexSet, exponents, lp_degree=LpDegree)

    # --- Non-lexicographically sorted (but complete) exponents
    idx = np.arange(len(exponents))
    np.random.shuffle(idx)
    exponents_shuffled = exponents[idx]
    # Assertion
    assert_call(MultiIndexSet, exponents_shuffled, lp_degree=LpDegree)

    # --- Non-lexicographically sorted and incomplete exponents
    idx_size = int(np.ceil(0.5 * len(exponents)))
    idx = np.random.choice(len(exponents), idx_size, replace=False)
    exponents_random = exponents[idx]
    # Assertion
    assert_call(MultiIndexSet, exponents_random, lp_degree=LpDegree)


def test_init_fail_from_exponents():
    """Test if invalid parameter values raise the expected errors."""
    # --- Invalid lp-degree
    # Create arbitrary complete exponents
    exponents = get_exponent_matrix(3, 2, 2)
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
    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

    # Add the same set of exponents with default parameter value
    exponents = mi.exponents
    mi_added = mi.add_exponents(exponents)
    # Assertions
    assert_multi_index_equal(mi_added, mi)

    # Add the same set of exponents with parameter
    mi_added = mi.add_exponents(exponents, inplace=False)
    # Assertions
    assert_multi_index_equal(mi_added, mi)

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
    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
    poly_degree = mi.poly_degree
    exponents = mi.exponents

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


def test_attributes_incomplete_exponents(
    SpatialDimension,
    PolyDegree,
    LpDegree
):
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


# --- make_complete()
def test_make_complete_inplace(SpatialDimension, PolyDegree, LpDegree):
    """Test in-place make_complete() on the MultiIndexSet instances."""
    if PolyDegree < 1:
        # Only test for polynomial degree of higher than 0 (0 always complete)
        return

    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

    # --- Make complete of an already complete set
    exponents = mi.exponents
    mi.make_complete(inplace=True)
    # Assertion: the exponents are identical
    assert mi.exponents is exponents

    # --- Make complete of an incomplete set
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
    assert mi_complete.is_complete
    assert mi_complete is not mi
    assert np.all(mi_complete.exponents == mi.exponents)

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


# --- expand_dim()
def test_expand_dim_invalid(SpatialDimension, PolyDegree, LpDegree):
    """Test invalid dimension expansion (i.e., contraction)."""
    # Create a multi-index set instance
    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

    # Expand the dimension
    new_dimension = SpatialDimension - 1
    # Assertion: Contraction raises an exception
    assert_raises(ValueError, mi.expand_dim, new_dimension)


def test_expand_dim_inplace(SpatialDimension, PolyDegree, LpDegree):
    """Test in-place multi-index set dimension expansion."""
    # Create a multi-index set instance
    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

    # Expand the dimension in-place (same dimension)
    new_dimension = SpatialDimension
    exponents = mi.exponents
    mi.expand_dim(new_dimension, inplace=True)
    # Assertion: identical exponents after expansion
    assert exponents is mi.exponents

    # Expand the dimension in-place (twice the dimension)
    new_dimension = SpatialDimension * 2
    mi.expand_dim(new_dimension, inplace=True)
    # Assertion: new columns are added to the exponents with 0 values
    assert np.all(mi.exponents[:, SpatialDimension:] == 0)


def test_expand_dim_outplace(SpatialDimension, PolyDegree, LpDegree):
    """Test out-place multi-index set dimension expansion."""
    # Create a multi-index set instance
    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

    # Expand the dimension out-place (same dimension)
    new_dimension = SpatialDimension
    expanded_mi = mi.expand_dim(new_dimension)
    # Assertion: exponents after expansion have the same value
    assert np.all(mi.exponents == expanded_mi.exponents)

    # Expand the dimension out-place (twice the dimension)
    new_dimension = SpatialDimension * 2
    expanded_mi = mi.expand_dim(new_dimension)
    # Assertion: new columns are added to the expanded index set with 0 values
    assert np.all(expanded_mi.exponents[:, SpatialDimension:] == 0)


# --- __mul__()
def test_multiplication_diff_deg(SpatialDimension, PolyDegree, LpDegree):
    """Test the multiplication of 2 MultiIndexSet instances with diff. deg."""
    # Problem setup
    m = SpatialDimension
    d_1 = PolyDegree
    d_2 = PolyDegree + np.random.randint(1, 3)
    p = LpDegree
    mi_1 = MultiIndexSet.from_degree(m, d_1, p)
    mi_2 = MultiIndexSet.from_degree(m, d_2, p)

    # MultiIndexSet product
    mi_prod = mi_1 * mi_2
    if p == 1.0:
        # This reference only applies if lp-degree is 1.0
        total_degree = d_1 + d_2  # the sum of degrees
        mi_prod_ref = MultiIndexSet.from_degree(m, total_degree, p)
    else:
        exp_prod = multiply_indices(mi_1.exponents, mi_2.exponents)
        mi_prod_ref = MultiIndexSet(exponents=exp_prod, lp_degree=p)

    # Assertion
    assert_multi_index_equal(mi_prod, mi_prod_ref)


def test_multiplication_diff_dim(SpatialDimension, PolyDegree, LpDegree):
    """Test the multiplication of 2 MultiIndexSet instances with diff. dim."""
    # Problem setup
    d = PolyDegree
    p = LpDegree
    m_1 = SpatialDimension
    m_2 = SpatialDimension + 1
    mi_1 = MultiIndexSet.from_degree(m_1, d, p)
    mi_2 = MultiIndexSet.from_degree(m_2, d, p)

    # MultiIndexSet product
    mi_prod = mi_1 * mi_2
    exp_prod = multiply_indices(mi_1.exponents, mi_2.exponents)
    mi_prod_ref = MultiIndexSet(exponents=exp_prod, lp_degree=p)

    # Assertion
    assert_multi_index_equal(mi_prod, mi_prod_ref)


def test_multiplication_diff_lp(SpatialDimension, PolyDegree):
    """Test the multiplication of 2 MultiIndexSet instances with diff. lp."""
    # Problem setup
    m = SpatialDimension
    d = PolyDegree
    lp_degrees = [0.5, 1.0, 2.0, 3.0, np.inf]
    p_1, p_2 = np.random.choice(lp_degrees, 2)
    mi_1 = MultiIndexSet.from_degree(m, d, p_1)
    mi_2 = MultiIndexSet.from_degree(m, d, p_2)

    # MultiIndexSet product
    mi_prod = mi_1 * mi_2
    exp_prod = multiply_indices(mi_1.exponents, mi_2.exponents)
    lp_prod = max([p_1, p_2])
    mi_prod_ref = MultiIndexSet(exponents=exp_prod, lp_degree=lp_prod)

    # Assertion
    assert_multi_index_equal(mi_prod, mi_prod_ref)
