"""
Test suite for instances of the `MultiIndexSet` class.
"""
import copy
import numpy as np
import pytest
from conftest import (
    build_rnd_exponents,
    LpDegree,
    PolyDegree,
    SpatialDimension,
    assert_call,
    mi_pair,
)
from numpy.testing import assert_raises

from minterpy import MultiIndexSet
from minterpy.core.utils import (
    get_exponent_matrix,
    multiply_indices,
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

    @pytest.mark.parametrize("invalid_value", [np.inf, -10, np.nan])
    def test_invalid_exps(self, invalid_value):
        """Test construction from invalid exponents."""
        # Generate exponent with a single invalid value
        exponents = np.random.rand(100, 3)
        exponents[2, 2] = invalid_value

        # Assertion
        with pytest.raises(ValueError):
            MultiIndexSet(exponents, lp_degree=1.0)

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
            (np.array([1, 2]), TypeError),  # NumPy array
        ],
    )
    def test_from_degree_invalid_lp_deg(self, lp_degree, exception):
        """Test failure of calling 'from_degree()' due to invalid lp-degree."""
        # Generate random dimension and poly. degree
        spatial_dim, poly_deg = np.random.randint(low=1, high=5, size=2)

        # Assertion
        with pytest.raises(exception):
            MultiIndexSet.from_degree(spatial_dim, poly_deg, lp_degree)


class TestAttributes:
    """All tests related to attributes of 'MultiIndexSet' instances."""
    def test_complete_exps(self, SpatialDimension, PolyDegree, LpDegree):
        """Test the attributes of MultiIndexSet from complete exponents."""
        # Create a MultiIndexSet from a complete set of exponents
        exponents = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)
        multi_index = MultiIndexSet(exponents, lp_degree=LpDegree)

        # Assertions
        assert isinstance(multi_index, MultiIndexSet)
        assert np.array_equal(exponents, multi_index.exponents)
        assert multi_index.lp_degree == LpDegree
        assert multi_index.poly_degree == PolyDegree
        assert multi_index.spatial_dimension == SpatialDimension
        assert len(multi_index) == len(exponents)
        assert multi_index.is_complete
        assert multi_index.is_downward_closed

        # Assigning to read-only properties
        with pytest.raises(AttributeError):
            # This is related to Issue #98
            multi_index.lp_degree = LpDegree
            # This is related to Issue #100
            multi_index.poly_degree = PolyDegree

    def test_incomplete_exps(self, SpatialDimension, PolyDegree, LpDegree):
        """Test the attributes of MultiIndexSet from incomplete exponents.

        Notes
        -----
        - This is related to the fix for Issue #66.
        """
        # Create a complete set and make it incomplete
        exponents = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)
        # Add a new exponent of a higher degree
        exponents_incomplete = np.insert(
            exponents,
            len(exponents),
            exponents[-1] + 2,
            axis=0,
        )
        mi_incomplete = MultiIndexSet(exponents_incomplete, LpDegree)

        # Assertions
        assert not mi_incomplete.is_complete
        assert mi_incomplete.lp_degree == LpDegree
        assert mi_incomplete.spatial_dimension == SpatialDimension
        assert np.array_equal(mi_incomplete.exponents, exponents_incomplete)

    @pytest.mark.parametrize("spatial_dimension", [1, 2, 3, 7])
    @pytest.mark.parametrize("poly_degree", [1, 2, 3, 5])
    def test_from_degree(self, spatial_dimension, poly_degree, LpDegree):
        """Test the attributes of MultiIndexSet from from_degree() constructor.

        Notes
        -----
        - This test is included due to Issue #97. The known breaking cases:
          the spatial dimension 7 and polynomial degree 5 are explicitly tested
          and only for this test. Otherwise, the test suite would be
          too time-consuming to run.
        """
        # Create an instance from 'from_degree()' constructor
        multi_index = MultiIndexSet.from_degree(
            spatial_dimension,
            poly_degree,
            LpDegree,
        )

        # Assertions
        assert isinstance(multi_index, MultiIndexSet)
        assert multi_index.lp_degree == LpDegree
        assert multi_index.spatial_dimension == spatial_dimension
        assert multi_index.poly_degree == poly_degree

    def test_empty_set(self, LpDegree):
        """Test the attributes of empty MultiIndexSet.

        Notes
        -----
        - This test is related to Issue #132.
        """
        # Create an empty set
        exponents = np.array([[]])
        multi_index = MultiIndexSet(exponents, LpDegree)

        # Assertions
        assert isinstance(multi_index, MultiIndexSet)
        assert multi_index.lp_degree == LpDegree
        assert multi_index.poly_degree is None
        assert multi_index.spatial_dimension == 0
        assert len(multi_index) == 0
        assert not multi_index.is_complete
        assert not multi_index.is_downward_closed

    @pytest.mark.parametrize("spatial_dimension", [0, 1, 5])
    def test_empty_set_dims(self, spatial_dimension, LpDegree):
        """Test the attributes of empty MultiIndexSet with various dimensions.

        Notes
        -----
        - This test is related to Issue #132.
        """
        # Create an empty set
        exponents = np.empty(shape=(0, spatial_dimension), dtype=int)
        multi_index = MultiIndexSet(exponents, LpDegree)

        # Assertions
        assert isinstance(multi_index, MultiIndexSet)
        assert np.array_equal(multi_index.exponents, exponents)
        assert multi_index.lp_degree == LpDegree
        assert multi_index.poly_degree is None
        assert multi_index.spatial_dimension == spatial_dimension
        assert len(multi_index) == 0
        assert not multi_index.is_complete
        assert not multi_index.is_downward_closed


class TestCopy:
    """All tests related to copy and deepcopy of MultiIndexSet instances.

    Notes
    -----
    - These tests are related to Issue #106.
    """
    def test_copy(self, SpatialDimension, PolyDegree, LpDegree):
        """Test creating a shallow copy."""
        # Create a multi-index set
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Create a shallow copy
        mi_copy = copy.copy(mi)

        # Assertions
        assert mi_copy == mi
        assert mi_copy is not mi
        assert np.shares_memory(mi.exponents, mi_copy.exponents)

    def test_deepcopy(self, SpatialDimension, PolyDegree, LpDegree):
        """Test creating a deep copy."""
        # Create a multi-index set
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Create a deep copy
        mi_deepcopy = copy.deepcopy(mi)

        # Assertions
        assert mi_deepcopy == mi
        assert mi_deepcopy is not mi
        assert not np.shares_memory(mi.exponents, mi_deepcopy.exponents)


# --- Instance Methods

class TestAddExponents:
    """All tests related to add_exponents() method."""
    def test_outplace_identical(self, SpatialDimension, PolyDegree, LpDegree):
        """Test adding identical exponents of a MultiIndex instance outplace.
    
        Notes
        -----
        - This is related to the fix for Issue #75
          and the refactoring of Issue #117.
        """
        # Create a set of exponents
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Add the same set of exponents with default parameter value
        exponents = mi.exponents
        mi_added = mi.add_exponents(exponents)
        # Assertion
        assert mi_added == mi

        # Add the same set of exponents with parameter
        mi_added = mi.add_exponents(exponents, inplace=False)
        # Assertion
        assert mi_added == mi

    def test_outplace(self, SpatialDimension, PolyDegree, LpDegree):
        """Test adding exponents of a MultiIndex instance outplace.

        Notes
        -----
        - This is related to the fix for Issue #75
          and the refactoring of Issue #117.
        """
        # Create 2 exponents, one twice the polynomial degree of the other
        exponents_1 = get_exponent_matrix(
            SpatialDimension, PolyDegree, LpDegree
        )
        exponents_2 = get_exponent_matrix(
            SpatialDimension, 2 * PolyDegree, LpDegree
        )

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
        assert mi_added == mi_2

        # Add the exponents difference with parameter
        mi_added = mi_1.add_exponents(exponents_diff, inplace=False)
        # Assertion: The added multi-index must be the same as the big one
        assert mi_added == mi_2

    def test_inplace_identical(self, SpatialDimension, PolyDegree, LpDegree):
        """Test adding identical exponents of a MultiIndex instance in-place.

        Notes
        -----
        - This test is related to the refactoring of Issue #117.
        """
        # Create a set of exponents
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        poly_degree = mi.poly_degree
        exponents = mi.exponents

        # Add the same set of exponents in-place
        mi.add_exponents(exponents, inplace=True)
        # Assertions
        assert mi.exponents is exponents
        assert mi.poly_degree == poly_degree

    def test_inplace(self, SpatialDimension, PolyDegree, LpDegree):
        """Test adding exponents of a MultiIndex instance in-place.

        Notes
        -----
        - This test is related to the refactoring of Issue #117.
        """
        # Create 2 exponents, one twice the polynomial degree of the other
        exponents_1 = get_exponent_matrix(
            SpatialDimension, PolyDegree, LpDegree
        )
        exponents_2 = get_exponent_matrix(
            SpatialDimension, 2 * PolyDegree, LpDegree
        )

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
        assert mi_1 == mi_2

    def test_sparse(self):
        """Test adding exponents to a high-dimensional sparse MultiIndexSet.

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

    def test_empty_set_outplace(self, SpatialDimension):
        """Test adding exponents to an empty MultiIndexSet out-place.

        Notes
        -----
        - This test is related to Issue #132.
        """
        # Create an empty set
        exponents = np.empty(shape=(0, SpatialDimension), dtype=int)
        mi = MultiIndexSet(exponents, lp_degree=1.0)

        # Add an exponent
        new_exponent = np.ones(shape=(SpatialDimension,))
        mi_added = mi.add_exponents(new_exponent)

        # Assertions
        assert len(mi_added) == 1
        assert np.array_equal(mi_added.exponents[0], new_exponent)

        # Add several exponents
        new_exponents = np.eye(SpatialDimension, dtype=int)
        mi_added = mi.add_exponents(new_exponents)

        # Assertions
        assert len(mi_added) == SpatialDimension
        assert np.array_equal(mi_added.exponents, new_exponents)

    def test_empty_set_inplace(self, SpatialDimension):
        """Test adding exponents to an empty MultiIndexSet in-place.

        Notes
        -----
        - This test is related to Issue #132.
        """
        # Create an empty set
        exponents = np.empty(shape=(0, SpatialDimension), dtype=int)
        mi = MultiIndexSet(exponents, lp_degree=1.0)

        # Add an exponent
        new_exponent = 5 * np.ones(shape=(SpatialDimension,))
        mi.add_exponents(new_exponent, inplace=True)

        # Assertions
        assert len(mi) == 1
        assert np.array_equal(mi.exponents[0], new_exponent)

        # Add several exponents
        new_exponents = np.eye(SpatialDimension, dtype=int)
        mi.add_exponents(new_exponents, inplace=True)

        # Assertions
        assert len(mi) == SpatialDimension + 1
        assert np.array_equal(mi.exponents[:-1], new_exponents)


class TestMakeComplete:
    """All tests related to the 'make_complete()' method."""
    def test_complete_inplace(self, SpatialDimension, PolyDegree, LpDegree):
        """Test in-place make_complete() on complete MultiIndexSet's."""
        if PolyDegree < 1:
            # Only test for polynomial degree > 0 (0 always complete)
            return

        # NOTE: By construction, 'from_degree()' returns a complete set
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        assert mi.is_complete

        # --- Make complete of an already complete set
        exponents = mi.exponents
        mi.make_complete(inplace=True)
        # Assertion: the exponents are identical object
        assert mi.exponents is exponents

    def test_incomplete_inplace(self, SpatialDimension, PolyDegree, LpDegree):
        """Test in-place make_complete() on incomplete MultiIndexSet's."""
        if PolyDegree < 1:
            # Only test for polynomial degree > 0 (0 always complete)
            return

        # Create a complete multi-index set
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # --- Make complete of an incomplete set
        # Get the highest multi-index set element
        exponent = np.atleast_2d(mi.exponents[-1])
        # Create a new instance with just one exponent (incomplete exponent)
        mi_incomplete = MultiIndexSet(exponent, LpDegree)
        assert not mi_incomplete.is_complete
        # Make complete in-place
        mi_incomplete.make_complete(inplace=True)
        # Assertions
        assert mi_incomplete.is_complete
        assert mi_incomplete == mi

    def test_complete_outplace(self, SpatialDimension, PolyDegree, LpDegree):
        """Test out-place make_complete() on complete MultiIndexSet's.

        Notes
        ----
        - This test is also related to Issue #115; by default, the 'inplace'
          parameter is set to False and a new instance is created.
        """
        if PolyDegree < 1:
            # Only test for polynomial degree of > 0 (0 always complete)
            return

        # NOTE: By construction, 'from_degree()' returns a complete set
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # --- Already complete set of exponents (a shallow copy is created)
        mi_complete = mi.make_complete(inplace=False)
        assert mi_complete.is_complete
        assert mi_complete is not mi
        # NOTE: Shallow copy but due to lex_sort in the default constructor,
        #       a new set of exponents are created.
        assert mi_complete == mi

    def test_incomplete_outplace(self, SpatialDimension, PolyDegree, LpDegree):
        """Test out-place make_complete() on incomplete MultiIndexSet's.

        Notes
        ----
        - This test is also related to Issue #115; by default, the 'inplace'
          parameter is set to False and a new instance is created.
        """
        if PolyDegree < 1:
            # Only test for polynomial degree of > 0 (0 always complete)
            return

        # Create a complete multi-index set
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

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
        assert mi_complete == mi

        # Make complete out-place (with an explicit argument)
        mi_complete = mi_incomplete.make_complete(inplace=False)

        # Assertions
        assert mi_complete.is_complete
        assert mi_complete is not mi_incomplete
        assert mi_complete == mi

    @pytest.mark.parametrize("spatial_dimension", [0, 1, 5])
    def test_empty_set_inplace(self, spatial_dimension, LpDegree):
        """Test in-place 'make_complete()' on empty MultiIndexSet's.

        Notes
        -----
        - This test is related to Issue #132.
        """
        # Create an empty set
        exponent = np.empty((0, spatial_dimension))
        mi = MultiIndexSet(exponent, LpDegree)

        # Assertion
        with pytest.raises(ValueError):
            mi.make_complete(inplace=True)

    @pytest.mark.parametrize("spatial_dimension", [0, 1, 5])
    def test_empty_set_outplace(self, spatial_dimension, LpDegree):
        """Test out-place 'make_complete()' on empty MultiIndexSet's.

        Notes
        -----
        - This test is related to Issue #132.
        """
        # Create an empty set
        exponent = np.empty((0, spatial_dimension))
        mi = MultiIndexSet(exponent, LpDegree)

        # Assertion
        with pytest.raises(ValueError):
            # with default parameter
            mi.make_complete()
        with pytest.raises(ValueError):
            # with parameter
            mi.make_complete(inplace=False)


class TestMakeDownwardClosed:
    """All tests related to 'make_downward_closed()' MultiIndexSet's.

    Notes
    -----
    - These tests are related to Issue #123.
    """
    def test_already_inplace(self, SpatialDimension, PolyDegree, LpDegree):
        """Test on already downward-closed MultiIndexSet's in-place."""
        # Only test for polynomial degree of higher than 0 (0 always complete)
        if PolyDegree < 1:
            return

        # By construction, a complete set is a downward-closed set
        exponents = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)
        mi = MultiIndexSet(exponents, LpDegree)

        # Already downward-closed, but make it downward-closed anyway
        mi.make_downward_closed(inplace=True)

        # Assertions: the exponents are identical object
        exponents = mi.exponents
        assert mi.exponents is exponents
        assert mi.is_downward_closed

    def test_non_inplace(self, SpatialDimension, PolyDegree, LpDegree):
        """Test on non downward-closed MultiIndexSet's out-place."""
        # Only test for polynomial degree of higher than 0 (0 always complete)
        if PolyDegree < 1:
            return

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

    def test_already_outplace(self, SpatialDimension, PolyDegree, LpDegree):
        """Test on already downward closed MultiIndexSet's out-place."""
        # Only test for polynomial degree of higher than 0 (0 always complete)
        if PolyDegree < 1:
            return

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
        assert mi_downward_closed == mi

    def test_non_outplace(self, SpatialDimension, PolyDegree, LpDegree):
        """Test on non downward-closed MultiIndexSet's out-place."""
        # Only test for polynomial degree of higher than 0 (0 always complete)
        if PolyDegree < 1:
            return

        # Get the highest multi-index set element from a complete set
        exponents = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)
        exponent = np.atleast_2d(exponents[-1])
        # Create a new instance with just a single exponent
        mi_non_downward_closed = MultiIndexSet(exponent, LpDegree)
        assert not mi_non_downward_closed.is_downward_closed

        # Make complete out-place (with the default parameter)
        mi_downward_closed_1 = mi_non_downward_closed.make_downward_closed()
        # Make complete out-place (with an explicit argument)
        mi_downward_closed_2 = mi_non_downward_closed.make_downward_closed(
            inplace=False
        )

        # Assertions
        assert mi_downward_closed_1.is_downward_closed
        assert mi_downward_closed_2.is_downward_closed
        assert mi_downward_closed_1 is not mi_non_downward_closed
        assert mi_downward_closed_2 is not mi_non_downward_closed
        if SpatialDimension > 1 and LpDegree != np.inf:
            assert not mi_downward_closed_1.is_complete
            assert not mi_downward_closed_2.is_complete
        else:
            # if LpDegree == inf, a downward-closed set is complete
            assert mi_downward_closed_1.is_complete
            assert mi_downward_closed_2.is_complete

    @pytest.mark.parametrize("spatial_dimension", [0, 1, 5])
    def test_empty_set_inplace(self, spatial_dimension, LpDegree):
        """Test in-place 'make_downward_closed()' on empty MultiIndexSet's.

        Notes
        -----
        - This test is related to Issue #132.
        """
        # Create an empty set
        exponent = np.empty((0, spatial_dimension))
        mi = MultiIndexSet(exponent, LpDegree)

        # Assertion
        with pytest.raises(ValueError):
            mi.make_downward_closed(inplace=True)

    @pytest.mark.parametrize("spatial_dimension", [0, 1, 5])
    def test_empty_set_outplace(self, spatial_dimension, LpDegree):
        """Test in-place 'make_downward_closed()' on empty MultiIndexSet's.

        Notes
        -----
        - This test is related to Issue #132.
        """
        # Create an empty set
        exponent = np.empty((0, spatial_dimension))
        mi = MultiIndexSet(exponent, LpDegree)

        # Assertion
        with pytest.raises(ValueError):
            # with default parameter
            mi.make_downward_closed()
        with pytest.raises(ValueError):
            # with parameter
            mi.make_downward_closed(inplace=False)


class TestExpandDim:
    """All tests related to the expand_dim() method."""
    def test_invalid(self, SpatialDimension, PolyDegree, LpDegree):
        """Test invalid dimension expansion (i.e., contraction)."""
        # Create a multi-index set instance
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Expand the dimension
        new_dimension = SpatialDimension - 1
        # Assertion: Contraction raises an exception
        assert_raises(ValueError, mi.expand_dim, new_dimension)

    def test_inplace(self, SpatialDimension, PolyDegree, LpDegree):
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

    def test_outplace(self, SpatialDimension, PolyDegree, LpDegree):
        """Test out-place multi-index set dimension expansion."""
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

    @pytest.mark.parametrize("spatial_dimension", [0, 1, 5])
    def test_empty_set_inplace(self, spatial_dimension, LpDegree):
        """Test in-place dimension expansion of empty sets."""
        # Create an empty set
        exponents = np.empty(shape=(0, spatial_dimension), dtype=int)
        mi = MultiIndexSet(exponents, LpDegree)

        # Expand the dimension in-place (same dimension)
        new_dimension = spatial_dimension
        exponents = mi.exponents
        mi.expand_dim(new_dimension, inplace=True)
        # Assertions
        assert exponents is mi.exponents  # identical exponents after expansion
        assert mi.spatial_dimension == new_dimension

        # Expand the dimension in-place (twice the dimension)
        new_dimension = spatial_dimension * 2
        mi.expand_dim(new_dimension, inplace=True)
        # Assertion
        assert mi.spatial_dimension == new_dimension

    @pytest.mark.parametrize("spatial_dimension", [0, 1, 5])
    def test_empty_set_outplace(self, spatial_dimension, LpDegree):
        """Test out-place dimension expansion of empty sets."""
        # Create an empty sets
        exponents = np.empty(shape=(0, spatial_dimension), dtype=int)
        mi = MultiIndexSet(exponents, LpDegree)

        # Expand the dimension out-place (same dimension)
        new_dimension = spatial_dimension
        expanded_mi = mi.expand_dim(new_dimension)
        # Assertions: exponents after expansion have the same value
        assert np.array_equal(mi.exponents, expanded_mi.exponents)
        assert expanded_mi.spatial_dimension == new_dimension

        # Expand the dimension out-place (twice the dimension)
        new_dimension = spatial_dimension * 2
        expanded_mi = mi.expand_dim(new_dimension)
        assert expanded_mi.spatial_dimension == new_dimension


class TestEquality:
    """All tests related to equality check.

    Notes
    -----
    - These tests are related to Issue #107.
    """
    def test_non_empty_set(self, SpatialDimension, PolyDegree, LpDegree):
        """Test the equality check between two instances of non-empty sets."""
        # Create two multi-index sets with the same parameters
        mi1 = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        mi2 = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Assertions: not identical but equal in value
        assert mi1 is not mi2
        assert mi1 == mi2

    @pytest.mark.parametrize("spatial_dimension", [0, 1, 5])
    def test_empty_set(self, spatial_dimension, LpDegree):
        """Test the equality check between two instances of empty sets.

        Notes
        -----
        - This test is related to Issue #132.
        """
        # Create an empty set
        exponent = np.empty((0, spatial_dimension))
        mi_1 = MultiIndexSet(exponent, LpDegree)
        mi_2 = MultiIndexSet(exponent, LpDegree)

        # Assertions: not identical but equal in value
        assert mi_1 is not mi_2
        assert mi_1 == mi_2


class TestInequality:
    """All tests related to inequality check.

    Notes
    -----
    - These tests are related to Issue #107.
    """
    def test_inequality(self):
        """Test the inequality check between MultiIndexSets."""
        # Create two different multi-index sets (both in exponents and lp-degree)
        mi_1 = MultiIndexSet.from_degree(3, 2, 1)
        mi_2 = MultiIndexSet.from_degree(4, 2, np.inf)

        # Assertion
        assert mi_1 != mi_2

    def test_lp_degree(self, SpatialDimension, PolyDegree):
        """Test the inequality check between MultiIndexSets w/ diff. lp-degrees.
        """
        # Create two sets with the same exponents but different lp-degrees
        exp = get_exponent_matrix(SpatialDimension, PolyDegree, np.inf)
        mi_1 = MultiIndexSet(exp, lp_degree=1.0)
        mi_2 = MultiIndexSet(exp, lp_degree=2.0)

        # Assertion
        assert mi_1 != mi_2

    def test_exponents(self, SpatialDimension, LpDegree):
        """Test the inequality check between MultiIndexSets w/ diff. exponents.
        """
        # Create two sets with different exponents but the same lp-degree.
        exp_1 = get_exponent_matrix(SpatialDimension, 2, np.inf)
        mi_1 = MultiIndexSet(exp_1, lp_degree=np.inf)
        exp_2 = get_exponent_matrix(SpatialDimension, 3, np.inf)
        mi_2 = MultiIndexSet(exp_2, lp_degree=np.inf)

        # Assertion
        assert mi_1 != mi_2

    @pytest.mark.parametrize("spatial_dimension", [0, 1, 5])
    def test_empty_set(self, spatial_dimension, LpDegree):
        """Test the equality check between two instances of empty sets.

        Notes
        -----
        - This test is related to Issue #132.
        """
        # Create an empty set
        exponent = np.empty((0, spatial_dimension))
        mi_1 = MultiIndexSet(exponent, lp_degree=1.0)
        mi_2 = MultiIndexSet(exponent, lp_degree=2.0)

        # Assertion
        assert mi_1 != mi_2


class TestMultiplication:
    """All tests related to the multiplication of MultiIndexSet instances.

    Notes
    -----
    - These tests are related to Issue #119, #125, and #132.
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


class TestContainmentCheck:
    """Tests about checking the containment of an element in MultiIndexSet.

    Notes
    -----
    - This series of tests is related to Issue #128.
    """
    def test_in(self, SpatialDimension, PolyDegree, LpDegree):
        """Check the containment of an element."""
        # Problem setup
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Take a random element
        idx = np.random.choice(len(mi))

        # Assertion: an element of a set must be contained by the set
        assert mi.exponents[idx] in mi

    def test_not_in(self, SpatialDimension, PolyDegree, LpDegree):
        """Check the negative containment of an element."""
        # Problem setup
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Take the highest index
        index = np.copy(mi.exponents[-1])  # copy, to modify
        index += 1  # Ensure the element is not in the set

        # Assertion
        assert index not in mi
        assert np.array([], dtype=int) not in mi  # empty

    def test_list(self, SpatialDimension, PolyDegree, LpDegree):
        """Check using lists as operand."""
        # Problem setup
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Take a random element
        idx = np.random.choice(len(mi))
        indices = mi.exponents[idx].tolist()

        # Assertion: an element of a set must be contained by the set
        assert indices in mi

    def test_wrong_type(self, SpatialDimension, PolyDegree, LpDegree):
        """Check using operand of a wrong type."""
        # Problem setup
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Test element
        indices = "a"

        # Assertion
        assert indices not in mi

    @pytest.mark.parametrize("spatial_dimension", [2, 3, 4])
    def test_wrong_dimension(self, spatial_dimension):
        """Check using operand of a wrong dimension, but can be expanded."""
        # Problem setup
        mi = MultiIndexSet.from_degree(spatial_dimension, 2, np.inf)

        # Test elements
        dim = mi.exponents.shape[1]
        indices_1 = np.ones(dim-1, dtype=int)  # smaller dimension
        indices_2 = np.zeros(dim+1, dtype=int)  # larger dimension

        # Assertions: Automatic dimension expansion
        assert indices_1 in mi
        assert indices_2 in mi

    def test_squeeze(self, SpatialDimension, PolyDegree, LpDegree):
        """Check using an operand with a squeezable dimension."""
        # Problem setup
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Take an element and increase dimension
        idx = np.random.choice(len(mi))
        indices = np.expand_dims(mi.exponents[idx], axis=0)

        # Assertion
        assert indices in mi

    def test_squeeze_list(self, SpatialDimension, PolyDegree, LpDegree):
        """Check using a list operand with a squeezable dimension."""
        # Problem setup
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Take an element and increase dimension
        idx = np.random.choice(len(mi))
        indices = np.expand_dims(mi.exponents[idx], axis=0).tolist()

        # Assertion
        assert indices in mi

    def test_multiple_elems(self, SpatialDimension, PolyDegree, LpDegree):
        """Check using an operand with multiple index elements."""
        # Problem setup
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Take random elements
        if len(mi) > 1:
            idx = np.random.randint(len(mi), size=2)
            indices = mi.exponents[idx]

            # Assertion
            assert indices not in mi

    def test_multiple_elems_list(self, SpatialDimension, PolyDegree, LpDegree):
        """Check using an operand with multiple index elements as list."""
        # Problem setup
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Take random elements
        if len(mi) > 1:
            idx = np.random.randint(len(mi), size=2)
            indices = mi.exponents[idx].tolist()

            # Assertion
            assert indices not in mi


class TestUnion:
    """All tests related to taking the union of MultiIndexSet instances.

    Notes
    -----
    - This series of tests is related to Issue #124 and #132.
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

    def test_inplace_operator(self, mi_pair):
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


class TestSubset:
    """All tests related to checking the subset of MultiIndexSet's."""
    def test_empty_set(self, SpatialDimension, PolyDegree, LpDegree):
        """Test that an empty set is a subset of any set.

        Notes
        -----
        - This test is related to Issue #132.
        """
        # Create a multi-index set (not empty)
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Create an empty set
        mi_empty = MultiIndexSet(np.array([[]]), LpDegree)

        # Assertions
        assert mi_empty.is_subset(mi)
        assert mi_empty <= mi
        assert not mi.is_subset(mi_empty)
        assert not mi <= mi_empty

    def test_method_same_dims(self, SpatialDimension, PolyDegree):
        """Test checking subset of the same dimension via a method call.

        Notes
        -----
        - This test is related to Issue #129.
        """
        # Create multi-index sets
        mi_1 = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, 1.0)
        mi_2 = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, 2.0)
        mi_3 = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, np.inf)

        # Assertions
        assert mi_1.is_subset(mi_1)
        assert mi_1.is_subset(mi_2)
        assert mi_2.is_subset(mi_3)
        assert mi_2.is_subset(mi_3, expand_dim=True)  # with parameter

    def test_operator_same_dims(self, SpatialDimension, PolyDegree):
        """Test checking subset of the same dimension via an operator.

        Notes
        -----
        - This test is related to Issue #129.
        """
        # Create multi-index sets
        mi_1 = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, 1.0)
        mi_2 = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, 2.0)
        mi_3 = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, np.inf)

        # Assertions
        assert mi_1 <= mi_1
        assert mi_1 <= mi_2
        assert mi_2 <= mi_3

    def test_method_diff_dims(self, PolyDegree, LpDegree):
        """Test checking subset of different dimensions via a method call.

        Notes
        -----
        - This test is related to Issue #129.
        """
        # Create multi-index sets
        mi_1 = MultiIndexSet.from_degree(1, PolyDegree, LpDegree)
        mi_2 = MultiIndexSet.from_degree(2, PolyDegree, LpDegree)
        mi_3 = MultiIndexSet.from_degree(3, PolyDegree, LpDegree)

        # Assertions
        assert mi_1.is_subset(mi_2, expand_dim=True)
        assert mi_2.is_subset(mi_3, expand_dim=True)

        with pytest.raises(ValueError):
            mi_1.is_subset(mi_2, expand_dim=False)

    @pytest.mark.parametrize("spatial_dimension", [2, 3, 4])
    def test_method_not_subset(self, spatial_dimension, LpDegree):
        """Test checking not a subset via a method call.

        Notes
        -----
        - This test is related to Issue #129.
        """
        # Create multi-index sets
        num_points = 50
        exps_1 = build_rnd_exponents(spatial_dimension, num_points)
        exps_2 = build_rnd_exponents(spatial_dimension, num_points)
        mi_1 = MultiIndexSet(exps_1, LpDegree)
        mi_2 = MultiIndexSet(exps_2, LpDegree)

        # Assertions
        assert not mi_1.is_subset(mi_2)
        assert not mi_2.is_subset(mi_1)

    @pytest.mark.parametrize("spatial_dimension", [2, 3, 4])
    def test_operator_not_subset(self, spatial_dimension, LpDegree):
        """Test checking not a subset via an operator.

        Notes
        -----
        - This test is related to Issue #129.
        """
        # Create multi-index sets
        num_points = 20
        exps_1 = build_rnd_exponents(spatial_dimension, num_points)
        exps_2 = build_rnd_exponents(spatial_dimension, num_points)
        mi_1 = MultiIndexSet(exps_1, LpDegree)
        mi_2 = MultiIndexSet(exps_2, LpDegree)

        # Assertions
        assert not mi_1 <= mi_2
        assert not mi_2 <= mi_1

    def test_not_subset_diff_dims(self):
        """Test checking not subset of different dimensions via a method call.

        Notes
        -----
        - This test is related to Issue #129.
        - This test specifically checks for different dimensions.
        """
        # Create multi-index sets
        mi_1 = MultiIndexSet.from_degree(1, 2, 1.0)
        mi_2 = MultiIndexSet.from_degree(2, 2, 2.0)
        mi_3 = MultiIndexSet.from_degree(3, 2, 3.0)

        # Assertions
        assert not mi_3.is_subset(mi_2, expand_dim=True)
        assert not mi_2.is_subset(mi_1, expand_dim=True)


class TestPropSubset:
    """All tests related to checking the proper subset of MultiIndexSet's.

    Notes
    -----
    - These tests are related to Issue #129.
    """
    def test_empty_set(self, SpatialDimension, PolyDegree, LpDegree):
        """Test that an empty set is a proper subset of any set, except itself.

        Notes
        -----
        - This test is related to Issue #132.
        """
        # Create a multi-index set (not empty)
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Create an empty set
        mi_empty = MultiIndexSet(np.array([[]]), LpDegree)

        # Assertions
        assert mi_empty.is_propsubset(mi)
        assert mi_empty < mi
        assert not mi.is_propsubset(mi_empty)
        assert not mi < mi_empty
        # Not with itself
        assert not mi_empty.is_propsubset(mi_empty)
        assert not mi_empty < mi_empty

    def test_method_same_dims(self, SpatialDimension, LpDegree):
        """Test checking proper subset of the same dimension via a method call.
        """
        # Create multi-index sets
        mi_1 = MultiIndexSet.from_degree(SpatialDimension, 2, LpDegree)
        mi_2 = MultiIndexSet.from_degree(SpatialDimension, 4, LpDegree)

        # Assertions
        assert not mi_1.is_propsubset(mi_1)  # With itself
        assert mi_1.is_propsubset(mi_2)
        assert mi_1.is_propsubset(mi_2, expand_dim=True)

    def test_operator_same_dims(self, SpatialDimension, LpDegree):
        """Test checking proper subset of the same dimension via an operator.
        """
        # Create multi-index sets
        mi_1 = MultiIndexSet.from_degree(SpatialDimension, 2, LpDegree)
        mi_2 = MultiIndexSet.from_degree(SpatialDimension, 4, LpDegree)

        # Assertions
        assert not mi_1 < mi_1  # With itself
        assert mi_1 < mi_2

    def test_method_diff_dims(self, LpDegree):
        """Test checking proper subset of diff. dimensions via a method call.
        """
        # Create multi-index sets
        mi_1 = MultiIndexSet.from_degree(1, 2, LpDegree)
        mi_2 = MultiIndexSet.from_degree(2, 4, LpDegree)

        # Assertions
        assert mi_1.is_propsubset(mi_2, expand_dim=True)
        with pytest.raises(ValueError):
            mi_1.is_propsubset(mi_2, expand_dim=False)

    def test_identity(self, SpatialDimension, PolyDegree, LpDegree):
        """Test checking identity from proper subset checking."""
        # Create two identical multi-index set
        mi1 = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        mi2 = copy.copy(mi1)

        # Assertion: This only applies if the dimension is the same
        assert mi1 <= mi2
        assert not mi1 < mi2
        assert mi1 == mi2
        assert mi2 <= mi1
        assert not mi2 < mi1
        assert mi1 == mi2

    @pytest.mark.parametrize("spatial_dimension", [2, 3, 4])
    def test_method_not_propsubset(self, spatial_dimension, LpDegree):
        """Test checking not a proper subset via a method call."""
        # Create multi-index sets
        num_points = 50
        exps_1 = build_rnd_exponents(spatial_dimension, num_points)
        exps_2 = build_rnd_exponents(spatial_dimension, num_points)
        mi_1 = MultiIndexSet(exps_1, LpDegree)
        mi_2 = MultiIndexSet(exps_2, LpDegree)

        # Assertions
        assert not mi_1.is_propsubset(mi_2)
        assert not mi_2.is_propsubset(mi_1)

    @pytest.mark.parametrize("spatial_dimension", [2, 3, 4])
    def test_operator_not_propsubset(self, spatial_dimension, LpDegree):
        """Test checking not a proper subset via an operator."""
        # Create multi-index sets
        num_points = 20
        exps_1 = build_rnd_exponents(spatial_dimension, num_points)
        exps_2 = build_rnd_exponents(spatial_dimension, num_points)
        mi_1 = MultiIndexSet(exps_1, LpDegree)
        mi_2 = MultiIndexSet(exps_2, LpDegree)

        # Assertions
        assert not mi_1 < mi_2
        assert not mi_2 < mi_1


class TestSuperset:
    """All tests related to checking the superset of MultiIndexSet's."""
    def test_empty_set(self, SpatialDimension, PolyDegree, LpDegree):
        """Test that any set is a superset of the empty set.

        Notes
        -----
        - This test is related to Issue #132.
        """
        # Create a multi-index set (not empty)
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Create an empty set
        mi_empty = MultiIndexSet(np.array([[]]), LpDegree)

        # Assertions
        assert mi.is_superset(mi_empty)
        assert not mi_empty.is_superset(mi)

    def test_method_same_dims(self, SpatialDimension, PolyDegree):
        """Test checking superset of the same dimension via a method call.

        Notes
        -----
        - This test is related to Issue #129.
        """
        # Create multi-index sets
        mi_1 = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, 1.0)
        mi_2 = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, 2.0)
        mi_3 = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, np.inf)

        # Assertions
        assert mi_3.is_superset(mi_2)
        assert mi_2.is_superset(mi_1)
        assert mi_1.is_superset(mi_1)
        assert mi_3.is_superset(mi_2, expand_dim=True)  # with parameter

    def test_operator_same_dims(self, SpatialDimension, PolyDegree):
        """Test checking superset of the same dimension via an operator.

        Notes
        -----
        - This test is related to Issue #129.
        """
        # Create multi-index sets
        mi_1 = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, 1.0)
        mi_2 = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, 2.0)
        mi_3 = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, np.inf)

        # Assertions
        assert mi_3 >= mi_2
        assert mi_2 >= mi_1
        assert mi_1 >= mi_1

    def test_method_diff_dims(self, PolyDegree, LpDegree):
        """Test checking superset of different dimensions via a method call.

        Notes
        -----
        - This test is related to Issue #129.
        """
        # Create multi-index sets
        mi_1 = MultiIndexSet.from_degree(1, PolyDegree, LpDegree)
        mi_2 = MultiIndexSet.from_degree(2, PolyDegree, LpDegree)
        mi_3 = MultiIndexSet.from_degree(3, PolyDegree, LpDegree)

        # Assertions
        assert mi_3.is_superset(mi_2, expand_dim=True)
        assert mi_2.is_superset(mi_1, expand_dim=True)

        with pytest.raises(ValueError):
            mi_3.is_superset(mi_2, expand_dim=False)

    @pytest.mark.parametrize("spatial_dimension", [2, 3, 4])
    def test_method_not_subset(self, spatial_dimension, LpDegree):
        """Test checking not a superset via a method call.

        Notes
        -----
        - This test is related to Issue #129.
        """
        # Create multi-index sets
        num_points = 50
        exps_1 = build_rnd_exponents(spatial_dimension, num_points)
        exps_2 = build_rnd_exponents(spatial_dimension, num_points)
        mi_1 = MultiIndexSet(exps_1, LpDegree)
        mi_2 = MultiIndexSet(exps_2, LpDegree)

        # Assertions
        assert not mi_1.is_superset(mi_2)
        assert not mi_2.is_superset(mi_1)

    @pytest.mark.parametrize("spatial_dimension", [2, 3, 4])
    def test_operator_not_subset(self, spatial_dimension, LpDegree):
        """Test checking not a superset via an operator.

        Notes
        -----
        - This test is related to Issue #129.
        """
        # Create multi-index sets
        num_points = 20
        exps_1 = build_rnd_exponents(spatial_dimension, num_points)
        exps_2 = build_rnd_exponents(spatial_dimension, num_points)
        mi_1 = MultiIndexSet(exps_1, LpDegree)
        mi_2 = MultiIndexSet(exps_2, LpDegree)

        # Assertions
        assert not mi_1 >= mi_2
        assert not mi_2 >= mi_1


class TestPropSuperset:
    """All tests related to checking the proper superset of MultiIndexSet's.

    Notes
    -----
    - These tests are related to Issue #129.
    """
    def test_empty_set(self, SpatialDimension, PolyDegree, LpDegree):
        """Test that any set is a proper superset of an empty set.

        Notes
        -----
        - This test is related to Issue #132.
        """
        # Create a multi-index set (not empty)
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Create an empty set
        mi_empty = MultiIndexSet(np.array([[]]), LpDegree)

        # Assertions
        assert mi.is_propsuperset(mi_empty)
        assert mi > mi_empty
        assert not mi_empty.is_propsuperset(mi)
        assert not mi_empty > mi
        # Empty set is not a proper superset of itself
        assert not mi_empty.is_propsuperset(mi_empty)
        assert not mi_empty < mi_empty

    def test_method_same_dims(self, SpatialDimension, LpDegree):
        """Test checking proper superset of the same dim. via a method call.
        """
        # Create multi-index sets
        mi_1 = MultiIndexSet.from_degree(SpatialDimension, 2, LpDegree)
        mi_2 = MultiIndexSet.from_degree(SpatialDimension, 4, LpDegree)

        # Assertions
        assert not mi_1.is_propsuperset(mi_1)  # With itself
        assert mi_2.is_propsuperset(mi_1)
        assert mi_2.is_propsuperset(mi_1, expand_dim=True)

    def test_operator_same_dims(self, SpatialDimension, LpDegree):
        """Test checking proper superset of the same dimension via an operator.
        """
        # Create multi-index sets
        mi_1 = MultiIndexSet.from_degree(SpatialDimension, 2, LpDegree)
        mi_2 = MultiIndexSet.from_degree(SpatialDimension, 4, LpDegree)

        # Assertions
        assert not mi_1 > mi_1  # With itself
        assert mi_2 > mi_1

    def test_method_diff_dims(self, LpDegree):
        """Test checking proper superset of diff. dimensions via a method call.
        """
        # Create multi-index sets
        mi_1 = MultiIndexSet.from_degree(1, 2, LpDegree)
        mi_2 = MultiIndexSet.from_degree(2, 4, LpDegree)

        # Assertions
        assert mi_2.is_propsuperset(mi_1, expand_dim=True)
        with pytest.raises(ValueError):
            mi_2.is_propsuperset(mi_1, expand_dim=False)

    def test_identity(self, SpatialDimension, PolyDegree, LpDegree):
        """Test checking identity from proper superset checking."""
        # Create two identical multi-index set
        mi1 = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        mi2 = copy.copy(mi1)

        # Assertion: This only applies if the dimension is the same
        assert mi1 >= mi2
        assert not mi1 > mi2
        assert mi1 == mi2
        assert mi2 >= mi1
        assert not mi2 > mi1
        assert mi1 == mi2

    @pytest.mark.parametrize("spatial_dimension", [2, 3, 4])
    def test_method_not_propsuperset(self, spatial_dimension, LpDegree):
        """Test checking not a proper superset via a method call."""
        # Create multi-index sets
        num_points = 50
        exps_1 = build_rnd_exponents(spatial_dimension, num_points)
        exps_2 = build_rnd_exponents(spatial_dimension, num_points)
        mi_1 = MultiIndexSet(exps_1, LpDegree)
        mi_2 = MultiIndexSet(exps_2, LpDegree)

        # Assertions
        assert not mi_1.is_propsuperset(mi_2)
        assert not mi_2.is_propsuperset(mi_1)

    @pytest.mark.parametrize("spatial_dimension", [2, 3, 4])
    def test_operator_not_propsuperset(self, spatial_dimension, LpDegree):
        """Test checking not a proper superset via an operator."""
        # Create multi-index sets
        num_points = 20
        exps_1 = build_rnd_exponents(spatial_dimension, num_points)
        exps_2 = build_rnd_exponents(spatial_dimension, num_points)
        mi_1 = MultiIndexSet(exps_1, LpDegree)
        mi_2 = MultiIndexSet(exps_2, LpDegree)

        # Assertions
        assert not mi_1 > mi_2
        assert not mi_2 > mi_1


class TestDisjoint:
    """All tests related to disjoint check.

    Notes
    -----
    - These tests are related to Issue #129.
    """
    def test_empty_set(self, SpatialDimension, PolyDegree, LpDegree):
        """Test that an empty set is disjoint with every set, inc. itself."""
        # Create an empty set
        mi_empty = MultiIndexSet(np.array([[]]), LpDegree)
        # Create a complete multi-index set
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Assertion
        assert mi_empty.is_disjoint(mi_empty)
        assert mi_empty.is_disjoint(mi)
        assert mi.is_disjoint(mi_empty)
        assert mi.is_disjoint(mi_empty, expand_dim=False)

    def test_not_disjoint_same_dim(self, SpatialDimension, LpDegree):
        """Test not disjoint sets having the same dimension."""
        # Create two sets whose intersection is not zero (known a priori)
        mi_1 = MultiIndexSet.from_degree(SpatialDimension, 2, LpDegree)
        mi_2 = MultiIndexSet.from_degree(SpatialDimension, 4, LpDegree)

        # Assertion
        assert not mi_1.is_disjoint(mi_2)
        assert not mi_2.is_disjoint(mi_1)

    def test_disjoint_same_dim(self, SpatialDimension, LpDegree):
        """Test disjoint sets having the same dimension."""
        # Create two disjoint sets (known a priori)
        exps_1 = np.eye(SpatialDimension)
        exps_2 = 2 * np.eye(SpatialDimension)
        mi_1 = MultiIndexSet(exps_1, LpDegree)
        mi_2 = MultiIndexSet(exps_2, LpDegree)

        # Assertions
        assert mi_1.is_disjoint(mi_2)
        assert mi_2.is_disjoint(mi_1)

    def test_not_disjoint_diff_dims(self, PolyDegree, LpDegree):
        """Test not disjoint sets having different dimensions."""
        # Create two sets
        mi_1 = MultiIndexSet.from_degree(2, PolyDegree, LpDegree)
        mi_2 = MultiIndexSet.from_degree(4, PolyDegree, LpDegree)

        # Assertions
        # No dimension expansion
        with pytest.raises(ValueError):
            mi_1.is_disjoint(mi_2)
        with pytest.raises(ValueError):
            mi_2.is_disjoint(mi_1)
        # Dimension expansion
        assert not mi_1.is_disjoint(mi_2, expand_dim=True)
        assert not mi_2.is_disjoint(mi_1, expand_dim=True)

    def test_disjoint_diff_dims(selfself, LpDegree):
        """Test disjoint sets having different dimensions."""
        # Create two disjoint sets (known a priori)
        exps_1 = np.eye(5)
        exps_2 = 2 * np.eye(6)
        mi_1 = MultiIndexSet(exps_1, LpDegree)
        mi_2 = MultiIndexSet(exps_2, LpDegree)

        # Assertion
        # No dimension expansion
        with pytest.raises(ValueError):
            mi_1.is_disjoint(mi_2)
        with pytest.raises(ValueError):
            mi_2.is_disjoint(mi_1)
        # Dimension expansion
        assert mi_1.is_disjoint(mi_2, expand_dim=True)
        assert mi_2.is_disjoint(mi_1, expand_dim=True)


class TestPrint:
    """All tests related to the string representation of MultiIndexSet.

    Notes
    -----
    - These tests are related to Issue #133.
    """
    def test_repr(self, SpatialDimension, PolyDegree, LpDegree):
        """Test __repr__ string representation."""
        # Create a multi-index set
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # __repr__()
        out = repr(mi)
        # exponents are printed with some additional spaces.
        exps = "\n".join([f"  {_}" for _ in repr(mi.exponents).splitlines()])

        # Assertions
        assert f"lp_degree={mi.lp_degree}" in out
        assert exps in out

    def test_str(self, SpatialDimension, PolyDegree, LpDegree):
        """Test __str__ string representation."""
        # Create a multi-index set
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # __str__()
        out = str(mi)

        # Assertions
        assert f"m={mi.spatial_dimension}" in out
        assert f"n={mi.poly_degree}" in out
        assert f"p={mi.lp_degree}" in out
        assert str(mi.exponents) in out

    @pytest.mark.parametrize("spatial_dimension", [0, 1, 2, 3])
    def test_repr_empty(self, spatial_dimension, LpDegree):
        """Test __repr__ string representation of empty set."""
        # Create an empty set
        exponent = np.empty((0, spatial_dimension), dtype=int)
        mi = MultiIndexSet(exponent, LpDegree)

        # __repr__()
        out = repr(mi)

        # Assertions
        assert repr(mi.exponents) in out
        assert f"lp_degree={mi.lp_degree}" in out


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
    if len(mi_1) == 0:
        exp_union = exp_mi_2
    elif len(mi_2) == 0:
        exp_union = exp_mi_1
    else:
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

    is_empty = len(mi_1) == 0 or len(mi_2) == 0
    if (m_1 == m_2) and (lp_1 == lp_2 == 1.0) and not is_empty:
        # This reference only applies if lp-degree is 1.0 with the same dim.
        total_degree = d_1 + d_2  # the sum of degrees
        m = np.max([m_1, m_2])
        mi_prod_ref = MultiIndexSet.from_degree(m, total_degree, lp_1)

        return mi_prod_ref

    lp_prod = max([lp_1, lp_2])

    if is_empty:
        exponents_prod = np.empty((0, np.max((m_1, m_2))))
    else:
        exponents_prod = multiply_indices(exponents_1, exponents_2)

    mi_prod_ref = MultiIndexSet(exponents=exponents_prod, lp_degree=lp_prod)

    return mi_prod_ref
