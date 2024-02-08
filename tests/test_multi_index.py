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
    mi_pair,
)
from copy import copy
from numpy.testing import assert_, assert_equal, assert_raises

from minterpy import MultiIndexSet
from minterpy.core.utils import (
    get_poly_degree,
    get_exponent_matrix,
    find_match_between,
    multiply_indices,
    is_downward_closed,
    is_complete,
    union_indices,
    insert_lexicographically,
    expand_dim,
)


# test initialization
class TestInitDefault:
    """All tests related to the default initialization of MultiIndexSet."""

    def test_complete_exps(self, SpatialDimension, PolyDegree, LpDegree):
        """Test the default constructor."""
        # Create complete exponents
        exponents = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)

        # Assertion
        assert_call(MultiIndexSet, exponents, lp_degree=LpDegree)

    def test_shuffled_exps(self, SpatialDimension, PolyDegree, LpDegree):
        """Test construction from lexicographically unsorted exponents."""
        # Create and shuffle complete exponents
        exponents = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)
        idx = np.arange(len(exponents))
        np.random.shuffle(idx)
        exponents_shuffled = exponents[idx]

        # Assertion
        assert_call(MultiIndexSet, exponents_shuffled, LpDegree)

    def test_incomplete_exps(self, SpatialDimension, PolyDegree, LpDegree):
        """Test construction from incomplete and (lexic) unsorted exponents."""
        # Create and shuffle incomplete exponents
        exponents = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)
        idx_size = int(np.ceil(0.5 * len(exponents)))
        idx = np.random.choice(len(exponents), idx_size, replace=False)
        exponents_random = exponents[idx]

        # Assertion
        assert_call(MultiIndexSet, exponents_random, lp_degree=LpDegree)

    @pytest.mark.parametrize("dimensionality", [1, 3, 4, 5])
    def test_invalid_exps_dimensionality(self, dimensionality):
        """Test construction with exponents array of invalid dimension."""
        # Generate random exponents and lp-degree
        shape = np.random.randint(1, 10, size=dimensionality)
        exponents = np.random.randint(low=1, high=10, size=tuple(shape))
        lp_degree = 5 * np.random.rand()

        # Assertion
        with pytest.raises(ValueError):
            MultiIndexSet(exponents, lp_degree=lp_degree)

    @pytest.mark.parametrize(
        "lp_degree, exception",
        [(0, ValueError),  (-1, ValueError), ("1.0", TypeError)],
    )
    def test_invalid_lp_degree(self, lp_degree, exception):
        """Test construction with invalid lp-degree values."""
        # Generate random dimension and poly. degree
        spatial_dim, poly_degree = np.random.randint(low=1, high=5, size=2)
        lp_degree_ = 5 * np.random.rand()

        # Create arbitrary complete exponents
        exponents = get_exponent_matrix(spatial_dim, poly_degree, lp_degree_)

        # Assertion
        with pytest.raises(exception):
            MultiIndexSet(exponents, lp_degree=lp_degree)


class TestInitFromDegree:
    """All tests related to 'from_degree()' constructor of MultiIndexSet."""
    def test_default(self, SpatialDimension, PolyDegree):
        """Test with the default lp-degree parameter."""
        # Create a multi-index set
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree)

        # Assertions: exponents are complete and downward-closed (Issue #105)
        assert mi.is_complete
        assert mi.is_downward_closed

    def test_lp_degree(self, SpatialDimension, PolyDegree, LpDegree):
        """Test with (valid) lp-degree parameter."""
        # Create a multi-index set
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Assertions: exponents are complete and downward-closed (Issue #105)
        assert mi.is_complete
        assert mi.is_downward_closed

    @pytest.mark.parametrize("spatial_dimension", [1, 1.0, np.array([1])[0]])
    def test_valid_dimension(self, spatial_dimension, PolyDegree, LpDegree):
        """Test with different valid types/values of spatial dimension.

        Notes
        -----
        - This test is related to Issue #77.
        """
        mi = MultiIndexSet.from_degree(spatial_dimension, PolyDegree, LpDegree)

        # Assertions: Type and Value
        assert isinstance(mi.spatial_dimension, int)
        assert mi.spatial_dimension == int(spatial_dimension)

    @pytest.mark.parametrize(
        "spatial_dimension, exception",
        [
            ("1", TypeError),  # string
            (1.5, ValueError),  # non-whole number
            (0, ValueError),  # zero
            (-1, ValueError),  # negative number
            (np.array([1, 2]), ValueError),  # NumPy array
        ],
    )
    def test_invalid_dimension(self, spatial_dimension, exception):
        """Test with different invalid types/values of spatial dimension.

        Notes
        -----
        - This test is related to Issue #77.
        """
        # Generate poly_degree and lp-degree
        poly_deg = np.random.randint(low=1, high=5)
        lp_deg = 5 * np.random.rand()

        with pytest.raises(exception):
            MultiIndexSet.from_degree(spatial_dimension, poly_deg, lp_deg)

    @pytest.mark.parametrize("poly_degree", [0, 0.0, 1, 1.0, np.array([1])[0]])
    def test_valid_poly_degree(self, SpatialDimension, poly_degree, LpDegree):
        """Test construction with different valid types/values of poly. degree.

        Notes
        -----
        - This test is related to Issue #101.
        """
        mi = MultiIndexSet.from_degree(SpatialDimension, poly_degree, LpDegree)

        # Assertions: Type and Value
        assert isinstance(mi.poly_degree, int)
        assert mi.poly_degree == int(poly_degree)

    @pytest.mark.parametrize(
        "poly_degree, exception",
        [
            ("1", TypeError),  # string
            (1.5, ValueError),  # non-whole number
            (-1, ValueError),  # negative number
            (np.array([1, 2]), ValueError),  # NumPy array
        ],
    )
    def test_invalid_poly_degree(self, poly_degree, exception):
        """Test failure due to invalid types/values of poly. degree.

        Notes
        -----
        - This test is related to Issue #101.
        """
        # Generate random dimension and lp-degree
        spatial_dim = np.random.randint(low=1, high=5)
        lp_degree = 5 * np.random.rand()

        # Assertion
        with pytest.raises(exception):
            MultiIndexSet.from_degree(spatial_dim, poly_degree, lp_degree)

    @pytest.mark.parametrize("lp_degree", [1, 1.2, np.array([1])[0]])
    def test_valid_lp_degree(self, SpatialDimension, PolyDegree, lp_degree):
        """Test construction with different types/values of lp-degree."""
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, lp_degree)

        # Assertions: Type and Value
        assert isinstance(mi.lp_degree, float)
        assert mi.lp_degree == float(lp_degree)

    @pytest.mark.parametrize(
        "lp_degree, exception",
        [
            ("1", TypeError),  # string
            (0, ValueError),  # zero
            (-1, ValueError),  # negative number
            (np.array([1, 2]), ValueError),  # NumPy array
        ],
    )
    def test_from_degree_invalid_lp_deg(self, lp_degree, exception):
        """Test failure of calling 'from_degree()' due to invalid lp-degree."""
        # Generate random dimension and poly. degree
        spatial_dim, poly_deg = np.random.randint(low=1, high=5, size=2)

        # Assertion
        with pytest.raises(exception):
            MultiIndexSet.from_degree(spatial_dim, poly_deg, lp_degree)


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
    # Create a complete set
    exponents = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)

    # Create an incomplete multi-index set
    # by adding a new exponent of higher degree
    exponents_incomplete = np.insert(
        exponents, len(exponents), exponents[-1] + 2, axis=0)
    multi_index_incomplete = MultiIndexSet(
        exponents_incomplete, lp_degree=LpDegree)

    # Make sure the incomplete exponents are indeed incomplete
    assert_(not is_complete(exponents_incomplete, PolyDegree, LpDegree))
    assert_(not multi_index_incomplete.is_complete)

    # Compute the reference polynomial degree given the multi-index set
    poly_degree = get_poly_degree(exponents_incomplete, LpDegree)

    # Assertion of attributes
    assert_(multi_index_incomplete.poly_degree == poly_degree)
    assert_(multi_index_incomplete.lp_degree == LpDegree)
    assert_(multi_index_incomplete.spatial_dimension == SpatialDimension)
    assert_equal(exponents_incomplete, multi_index_incomplete.exponents)


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


# --- Instance Methods

# --- MultiIndexSet.add_exponents()
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
    assert mi.is_downward_closed

    # Add a new (sparse) element
    new_element = np.zeros(m)
    new_element[0] = m
    mi_added = mi.add_exponents(new_element)

    # Assertions
    assert mi_added.is_downward_closed
    assert mi_added.poly_degree == mi.poly_degree + 1


# --- make_complete()
def test_make_complete_inplace(SpatialDimension, PolyDegree, LpDegree):
    """Test in-place make_complete() on the MultiIndexSet instances."""
    if PolyDegree < 1:
        # Only test for polynomial degree of higher than 0 (0 always complete)
        return

    # NOTE: By construction, 'from_degree()' returns a complete set
    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
    assert mi.is_complete

    # --- Make complete of an already complete set
    exponents = mi.exponents
    mi.make_complete(inplace=True)
    # Assertion: the exponents are identical object
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
    # NOTE: Shallow copy but due to lex_sort in the default constructor,
    #       a new set of exponents are created.
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


# --- make_downward_closed()
def test_make_downward_closed_inplace(SpatialDimension, PolyDegree, LpDegree):
    """Test in-place make_downward_closed() on the MultiIndexSet instances.

    Notes
    ----
    - This is test is also related to Issue #123.
    """
    if PolyDegree < 1:
        # Only test for polynomial degree of higher than 0 (0 always complete)
        return

    # --- Already downward-closed set of exponents

    # By construction, a complete set is a downward-closed set
    exponents = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)
    # Exclude the last element to break completeness
    mi = MultiIndexSet(exponents[:-1], LpDegree)
    assert mi.is_downward_closed

    # Already downward-closed, but make it downward-closed anyway
    mi.make_downward_closed(inplace=True)
    # Assertion: the exponents are identical object
    exponents = mi.exponents
    assert mi.exponents is exponents

    # ---  Non-downward-closed set of exponents

    # Get the highest multi-index set element from a complete set
    exponents = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)
    exponent = np.atleast_2d(exponents[-1])
    # Create a new instance with just a single exponent
    mi = MultiIndexSet(exponent, LpDegree)
    assert not mi.is_downward_closed

    # Make downward-closed in-place
    mi.make_downward_closed(inplace=True)
    # Assertions
    assert mi.is_downward_closed
    if SpatialDimension > 1 and LpDegree != np.inf:
        assert not mi.is_complete
    else:
        # if LpDegree == inf, a downward-closed set is complete
        assert mi.is_complete


def test_make_downward_closed_outplace(SpatialDimension, PolyDegree, LpDegree):
    """Test out-place make_downward_closed() on the MultiIndexSet instances.

    Notes
    ----
    - This is test is also related to Issue #123.
    """
    if PolyDegree < 1:
        # Only test for polynomial degree of higher than 0 (0 always complete)
        return

    # --- Already downward-closed set of exponents

    # By construction, a complete set is a downward-closed set
    exponents = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)
    # Exclude the last element to break completeness
    mi = MultiIndexSet(exponents[:-1], LpDegree)
    assert mi.is_downward_closed

    # Already downward-closed, but make it downward-closed anyway
    mi_downward_closed = mi.make_downward_closed(inplace=False)

    # Assertions
    assert mi_downward_closed.is_downward_closed
    assert mi_downward_closed is not mi
    # NOTE: Shallow copy but due to lex_sort in the default constructor,
    #       a new set of exponents are created.
    assert np.all(mi_downward_closed.exponents == mi.exponents)

    # --- Non-downward-closed set of exponents

    # Get the highest multi-index set element from a complete set
    exponents = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)
    exponent = np.atleast_2d(exponents[-1])
    # Create a new instance with just a single exponent
    mi_non_downward_closed = MultiIndexSet(exponent, LpDegree)
    assert not mi_non_downward_closed.is_downward_closed

    # Make complete out-place (with the default parameter)
    mi_downward_closed = mi_non_downward_closed.make_downward_closed()
    # Assertions
    assert mi_downward_closed.is_downward_closed
    assert mi_downward_closed is not mi_non_downward_closed

    # Make complete out-place (with an explicit argument)
    mi_downward_closed = mi_non_downward_closed.make_downward_closed(
        inplace=False
    )
    # Assertions
    assert mi_downward_closed.is_downward_closed
    assert mi_downward_closed is not mi_non_downward_closed
    if SpatialDimension > 1 and LpDegree != np.inf:
        assert not mi_downward_closed.is_complete
    else:
        # if LpDegree == inf, a downward-closed set is complete
        assert mi_downward_closed.is_complete


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


# --- __eq__()
def test_equality(SpatialDimension, PolyDegree, LpDegree):
    """Test the equality check between two instances of MultiIndexSet.

    Notes
    -----
    - This test is related to Issue #107.
    """
    # Create two multi-index sets with the same parameters
    mi_1 = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
    mi_2 = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

    # Assertions
    assert mi_1 is not mi_2  # Instances are not identical
    assert mi_1 == mi_2  # Instances are equal in value


def test_inequality():
    """Test the inequality check between MultiIndexSets.

    Notes
    -----
    - This test is related to Issue #107.
    """
    # Create two different multi-index sets (both in exponents and lp-degree)
    mi_1 = MultiIndexSet.from_degree(3, 2, 1)
    mi_2 = MultiIndexSet.from_degree(4, 2, np.inf)

    # Assertion
    assert mi_1 != mi_2


def test_inequality_lp_degree(SpatialDimension, PolyDegree):
    """Test the inequality check between MultiIndexSets w/ diff. lp-degrees.

    Notes
    -----
    - This test is related to Issue #107.
    """
    # Create two sets with the same exponents but different lp-degrees
    exp = get_exponent_matrix(SpatialDimension, PolyDegree, np.inf)
    mi_1 = MultiIndexSet(exp, lp_degree=1.0)
    mi_2 = MultiIndexSet(exp, lp_degree=2.0)

    # Assertion
    assert mi_1 != mi_2


def test_inequality_exponents(SpatialDimension, LpDegree):
    """Test the inequality check between MultiIndexSets w/ diff. exponents.

    Notes
    -----
    - This test is related to Issue #107.
    """
    # Create two sets with different exponents but the same lp-degree.
    exp_1 = get_exponent_matrix(SpatialDimension, 2, np.inf)
    mi_1 = MultiIndexSet(exp_1, lp_degree=np.inf)
    exp_2 = get_exponent_matrix(SpatialDimension, 3, np.inf)
    mi_2 = MultiIndexSet(exp_2, lp_degree=np.inf)

    # Assertion
    assert mi_1 != mi_2


class TestMultiplication:
    """All tests related to the multiplication of MultiIndexSet instances.

    Notes
    -----
    - These tests are related to Issue #119 and #125.
    """
    def test_operator_outplace(self, mi_pair):
        """Multiply two instances via (outplace) operator."""
        # Problem setup
        mi_1, mi_2 = mi_pair

        # Create a reference product
        mi_prod_ref = _create_mi_prod_ref(mi_1, mi_2)

        # Multiply via an operator
        mi_prod_1 = mi_1 * mi_2
        mi_prod_2 = mi_2 * mi_1  # commutativity check

        # Assertions
        assert mi_prod_1 == mi_prod_ref
        assert mi_prod_2 == mi_prod_ref

    def test_operator_inplace(self, mi_pair):
        """Multiply two instances via inplace operator."""
        # Problem setup
        mi_1, mi_2 = mi_pair

        # Create a reference product
        mi_prod_ref = _create_mi_prod_ref(mi_1, mi_2)

        # Multiply via an inplace operator method call
        mi_1 *= mi_2

        # Assertion
        assert mi_1 == mi_prod_ref

    def test_method_outplace(self, mi_pair):
        """Multiply two instances via outplace method."""
        # Problem setup
        mi_1, mi_2 = mi_pair

        # Create a reference product
        mi_prod_ref = _create_mi_prod_ref(mi_1, mi_2)

        # Multiply via a method call
        mi_prod_1 = mi_1.multiply(mi_2)  # default parameter
        mi_prod_2 = mi_1.multiply(mi_2, False)  # positional argument
        mi_prod_3 = mi_1.multiply(mi_2, inplace=False)  # explicit name
        mi_prod_4 = mi_2.multiply(mi_1)  # commutativity check

        # Assertions
        assert mi_prod_1 == mi_prod_ref
        assert mi_prod_2 == mi_prod_ref
        assert mi_prod_3 == mi_prod_ref
        assert mi_prod_4 == mi_prod_ref

    def test_method_inplace(self, mi_pair):
        """Multiply two instances via inplace method."""
        # Problem setup
        mi_1, mi_2 = mi_pair

        # Create a reference product
        mi_prod_ref = _create_mi_prod_ref(mi_1, mi_2)

        # Multiply via an inplace method call
        mi_1.multiply(mi_2, inplace=True)

        # Assertion
        assert mi_1 == mi_prod_ref


class TestUnion:
    """All tests related to taking the union of MultiIndexSet instances.

    Notes
    -----
    - This series of tests is related to Issue #124.
    """
    def test_outplace_operator(self, mi_pair):
        """Take the union via the union operator."""
        # Problem setup
        mi_1, mi_2 = mi_pair

        # Create a reference
        mi_ref = _create_mi_union_ref(mi_1, mi_2)

        # MultiIndexSet union via the operator
        mi_union_1 = mi_1 | mi_2
        mi_union_2 = mi_2 | mi_1  # commutativity check

        # Assertions
        assert mi_ref == mi_union_1
        assert mi_union_1 == mi_union_2

    def test_outplace_method(self, mi_pair):
        """Take the union via the method call."""
        # Problem setup
        mi_1, mi_2 = mi_pair

        # Create a reference
        mi_ref = _create_mi_union_ref(mi_1, mi_2)

        # MultiIndexSet union
        mi_union_1 = mi_1.union(mi_2)  # method call, default
        mi_union_2 = mi_1.union(mi_2, False)  # method call, positional
        mi_union_3 = mi_1.union(mi_2, inplace=False)  # method call, keyword
        mi_union_4 = mi_2.union(mi_1)  # commutativity check

        # Assertions
        assert mi_ref == mi_union_1
        assert mi_union_1 == mi_union_2
        assert mi_union_2 == mi_union_3
        assert mi_union_3 == mi_union_4

    def test_inplace_op(self, mi_pair):
        """Take the union with the inplace operator."""
        # Problem setup
        mi_1, mi_2 = mi_pair

        # Create a reference
        id_mi_1 = id(mi_1)
        mi_ref = _create_mi_union_ref(mi_1, mi_2)

        # MultiIndexSet union
        mi_1 |= mi_2

        # Assertion
        assert mi_ref == mi_1
        assert id_mi_1 == id(mi_1)  # the same object

    def test_inplace_method(self, mi_pair):
        """Take the union via the inplace method call."""
        # Problem setup
        mi_1, mi_2 = mi_pair

        # Create a reference
        mi_ref = _create_mi_union_ref(mi_1, mi_2)

        # MultiIndexSet union
        mi_1.union(mi_2, inplace=True)

        # Assertion
        assert mi_ref == mi_1


def _create_mi_union_ref(mi_1: MultiIndexSet, mi_2: MultiIndexSet):
    """Create a reference of the union of two MultiIndexSets.

    Parameters
    ----------
    mi_1 : MultiIndexSet
        The first operand in the multiplication.
    mi_2 : MultiIndexSet
        The second operand in the multiplication.

    Returns
    -------
    MultiIndexSet
        The reference of two MultiIndexSets product.
    """
    exp_mi_1 = mi_1.exponents
    exp_mi_2 = mi_2.exponents
    if mi_1.spatial_dimension > mi_2.spatial_dimension:
        exp_mi_2 = expand_dim(exp_mi_2, mi_1.spatial_dimension)
    else:
        exp_mi_1 = expand_dim(exp_mi_1, mi_2.spatial_dimension)
    exp_union = insert_lexicographically(exp_mi_1, exp_mi_2)
    lp_union = np.max([mi_1.lp_degree, mi_2.lp_degree])
    mi_union = MultiIndexSet(exponents=exp_union, lp_degree=lp_union)

    return mi_union


def _create_mi_prod_ref(mi_1: MultiIndexSet, mi_2: MultiIndexSet):
    """Create a reference of two MultiIndexSets product.

    Parameters
    ----------
    mi_1 : MultiIndexSet
        The first operand in the multiplication.
    mi_2 : MultiIndexSet
        The second operand in the multiplication.

    Returns
    -------
    MultiIndexSet
        The reference of two MultiIndexSets product.
    """
    m_1 = mi_1.spatial_dimension
    m_2 = mi_2.spatial_dimension
    d_1 = mi_1.poly_degree
    d_2 = mi_2.poly_degree
    lp_1 = mi_1.lp_degree
    lp_2 = mi_2.lp_degree
    exponents_1 = mi_1.exponents
    exponents_2 = mi_2.exponents

    if (m_1 == m_2) and (lp_1 == lp_2 == 1.0):
        # This reference only applies if lp-degree is 1.0 with the same dim.
        total_degree = d_1 + d_2  # the sum of degrees
        m = np.max([m_1, m_2])
        mi_prod_ref = MultiIndexSet.from_degree(m, total_degree, lp_1)

        return mi_prod_ref

    exponents_prod = multiply_indices(exponents_1, exponents_2)
    lp_prod = max([lp_1, lp_2])
    mi_prod_ref = MultiIndexSet(exponents=exponents_prod, lp_degree=lp_prod)

    return mi_prod_ref
