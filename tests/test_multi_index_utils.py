"""
Test suite for multi_index_utils.py
"""
import numpy as np
import pytest
from conftest import (
    LpDegree,
    PolyDegree,
    SpatialDimension,
    assert_call,
    build_rnd_exponents,
)
from numpy.testing import assert_, assert_equal, assert_raises

from minterpy.core.utils import (
    _gen_multi_index_exponents,
    find_match_between,
    expand_dim,
    get_exponent_matrix,
    gen_backward_neighbors,
    gen_missing_backward_neighbors,
    insert_lexicographically,
    is_complete,
    is_disjoint,
    is_downward_closed,
    make_complete,
    make_downward_closed,
    make_derivable,
    multiply_indices,
    lex_sort,
    union_indices,
)
from minterpy.global_settings import NOT_FOUND
from minterpy.jit_compiled_utils import (
    all_indices_are_contained,
    is_lex_sorted,
    is_index_contained,
    is_lex_smaller_or_equal,
    search_lex_sorted,
)
from minterpy.global_settings import NOT_FOUND

MIN_POLY_DEG = 1
MAX_POLY_DEG = 5
SEED = 12345678


def get_random_dim(spatial_dimension, seed=None):
    """Pick randomly a dimension.

    Parameters
    ----------
    spatial_dimension : int
        Number of spatial dimensions.
    seed : int, optional
        Seed number for the pseudo-random number generator.

    Returns
    -------
    int
        A random integer for dimension selection.
    """
    if seed is None:
        seed = SEED
    rng = np.random.default_rng(seed)

    if spatial_dimension == 1:
        rnd_dim = 0
    else:
        rnd_dim = rng.integers(0, spatial_dimension - 1)

    return rnd_dim


def get_lex_bigger(index, bigger_by_1=False, seed=None):
    """Get a lexicographically bigger multi-indices from a given one.

    Parameters
    ----------
    index : :class:`numpy:numpy.ndarray`
        Given multi-indices, a two-dimensional array of integers.
    bigger_by_1 :  bool
        The next larger( dimension-wise) multi-indices.
    seed : int, optional
        Seed number for the pseudo-random number generator.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        A lexicographically bigger multi-indices.
    """
    rnd_dim = get_random_dim(len(index), seed=seed)
    out = index.copy()
    out[rnd_dim] += 1
    if not bigger_by_1:
        # Setting all previous entries to 0, increasing index only in one dim.
        out[:rnd_dim] = 0
    return out


# fixtures for number of monomials
number_of_monomials = [1, 2]


@pytest.fixture(params=number_of_monomials)
def NumOfMonomials(request):
    return request.param


# test utilities

def test_call_get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree):
    
    # The function can be called
    assert_call(get_exponent_matrix, SpatialDimension, PolyDegree, LpDegree)

    # The function can be called with finite integral float as poly. degree
    # NOTE: This test is related to the fix for Issue #65
    poly_degree = float(PolyDegree)
    assert_call(get_exponent_matrix, SpatialDimension, poly_degree, LpDegree)
    
    # The function can't be called non-finite integral float
    # NOTE: This test is related to the fix for Issue #65
    poly_degree = PolyDegree + np.random.rand(1)[0]
    assert_raises(
        ValueError,
        get_exponent_matrix,
        SpatialDimension,
        poly_degree,
        LpDegree
        )


# --- is_lex_smaller_or_equal()
def test_lex_smaller_or_equal(SpatialDimension, PolyDegree):
    """Test lexicographically comparing two different multi-index elements."""
    # Create a random multi-indices
    if PolyDegree == 0:
        indices_1 = np.zeros(SpatialDimension, dtype=int)
    else:
        indices_1 = np.random.randint(0, PolyDegree, SpatialDimension)

    # Assertion: Equal multi-indices
    assert is_lex_smaller_or_equal(indices_1, indices_1)

    # Create a lexicographically bigger multi-indices
    indices_2 = get_lex_bigger(indices_1)

    # Assertions
    assert is_lex_smaller_or_equal(indices_1, indices_2)
    assert not is_lex_smaller_or_equal(indices_2, indices_1)


# --- is_lex_sorted()
def test_is_lex_sorted(SpatialDimension, PolyDegree, LpDegree):
    """Test checking whether a multi-index set is lexicographically sorted."""
    # --- By construction, below is lexicographically ordered
    indices = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)

    # Assertion
    assert is_lex_sorted(indices)


def test_is_lex_sorted_single():
    """Test if a single entry multi-index set is lexicographically sorted."""
    index = np.random.randint(1, 5, (1, 10))

    # Assertion: Always lexicographical
    assert is_lex_sorted(index)


def test_is_lex_sorted_random():
    """Test if a random integer array is lexicographically sorted."""
    # Generate randomly, big enough so there's no chance it will be sorted
    indices = np.random.randint(1, 5, (5, 12))

    # Assertion - Not lexicographical
    assert not is_lex_sorted(indices)


def test_is_lex_sorted_duplicates():
    """Test if a multi-index set with duplicate entries is lexicographical."""
    indices = np.array([[0, 0], [2, 0], [2, 0]])

    # Assertion: Not lexicographical
    assert not is_lex_sorted(indices)


# --- lex_sort()
def test_lex_sort(SpatialDimension, PolyDegree, LpDegree):
    """Test lexicographically sorting a shuffled multi-index set."""

    # --- Create a complete multi-indices (no duplication)
    indices = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)

    # Shuffle the indices
    idx = np.arange(len(indices))
    np.random.shuffle(idx)
    indices_shuffled = indices[idx]

    # Sort lexicographically
    indices_lexsorted = lex_sort(indices_shuffled)
    # Assertions
    assert is_lex_sorted(indices_lexsorted)
    assert np.all(indices == indices_lexsorted)

    # --- Introduce duplicate entries
    idx_2 = np.arange(len(indices))
    np.random.shuffle(idx_2)
    indices_shuffled = np.concatenate((indices[idx], indices[idx_2]))

    # Sort lexicographically and remove duplicate entries
    indices_lexsorted = lex_sort(indices_shuffled)
    # Assertions
    assert is_lex_sorted(indices_lexsorted)
    assert np.all(indices == indices_lexsorted)


# --- search_lex_sorted()
def test_search_lex_sorted(SpatialDimension, PolyDegree, LpDegree):
    """Test searching a set of multi-indices for a given index.

    Notes
    -----
    - This test is related to the refactoring as described in Issue #121.
    """
    # Create a complete multi-indices
    indices = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)

    # --- An index is contained in the set
    idx_ref = int(np.random.randint(0, len(indices), 1))  # must be int
    index_ref = indices[idx_ref]
    idx = search_lex_sorted(indices, index_ref)
    # Assertion
    assert idx_ref == idx

    # --- An index is not contained in the set
    index_ref = indices[-1] + 10
    idx = search_lex_sorted(indices, index_ref)
    # Assertion: Use global indicator if an element is not found
    assert idx == NOT_FOUND


# --- is_index_contained()
def test_is_index_contained(SpatialDimension, PolyDegree, LpDegree):
    """Test checking whether an entry is contained in a multi-index set."""

    # Create a complete multi-indices
    indices = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)

    # --- An index is contained in the set
    idx_ref = int(np.random.randint(0, len(indices), 1))  # must be int
    index_ref = indices[idx_ref]
    # Assertion
    assert is_index_contained(indices, index_ref)

    # --- An index is not contained in the set
    index_ref = indices[-1] + 10  # won't be in the set
    # Assertion: Use global indicator if an element is not found
    assert not is_index_contained(indices, index_ref)


# --- is_complete()
def test_is_complete(SpatialDimension, PolyDegree, LpDegree):
    """Test if a multi-index set is complete."""
    # --- A complete set
    indices = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)
    # Assertion: The function always returns complete set of multi-indices
    assert is_complete(indices, PolyDegree, LpDegree)

    # --- Not a complete set
    if PolyDegree > 0:
        # NOTE: Only applies for PolyDegree > 0 (== 0 has only 1 element)
        # Remove the largest multi-index element
        indices = np.delete(indices, -1, axis=0)
        # Assertion: Completeness lost
        assert not is_complete(indices, PolyDegree, LpDegree)


class TestIsDisjoint:
    """All tests related to disjoint check between to multi-index arrays.

    Notes
    -----
    - These tests are related to Issue #129.
    """
    def test_disjoint(self, SpatialDimension):
        """Test two disjoint sets."""
        indices_1 = np.eye(SpatialDimension, dtype=int)
        indices_2 = 2 * np.eye(SpatialDimension, dtype=int)

        # Assertions
        assert is_disjoint(indices_1, indices_2)
        assert is_disjoint(indices_2, indices_1)

    def test_not_disjoint(self, SpatialDimension, LpDegree):
        """Test two not disjoint sets."""
        # Create two not disjoint sets
        indices_1 = get_exponent_matrix(SpatialDimension, 2, LpDegree)
        indices_2 = get_exponent_matrix(SpatialDimension, 4, LpDegree)

        # Assertions
        assert not is_disjoint(indices_1, indices_2)
        assert not is_disjoint(indices_2, indices_1)

    def test_not_disjoint_diff_dims(self, SpatialDimension, LpDegree):
        """Test two not disjoint sets having different dimensions."""
        # Create two sets
        indices_1 = np.eye(2, dtype=int)
        indices_2 = np.eye(4, dtype=int)

        # Assertions
        # Different dimension automatically disjoint
        assert is_disjoint(indices_1, indices_2)
        assert is_disjoint(indices_2, indices_1)
        # ...unless the dimension is expanded
        assert not is_disjoint(indices_1, indices_2, True)
        assert not is_disjoint(indices_2, indices_1, True)


# --- is_downward_closed()
def test_is_downward_closed(SpatialDimension, PolyDegree, LpDegree):
    """Test if a multi-index set is downward-closed."""
    # --- A complete set is a downward-closed set
    indices = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)
    # Assertion
    assert is_downward_closed(indices)

    # --- Without the largest element, a set remains downward-closed
    if PolyDegree > 0:
        # NOTE: Only applies for PolyDegree > 0 (== 0 has only 1 element)
        assert is_downward_closed(np.delete(indices, -1, axis=0))

    # --- Not downward-closed
    if PolyDegree > 0:
        # NOTE: Only applies for PolyDegree > 0 (== 0 has only 1 element)
        # Assertion: Without the lexicographically smallest element
        assert not is_downward_closed(np.delete(indices, 0, axis=0))

    if PolyDegree > 1:
        # NOTE: Only applies for PolyDegree > 1 (== 1 has at least 2 elements)
        # Assertion: Without the 2nd lexicographically smallest element
        assert not is_downward_closed(np.delete(indices, 1, axis=0))


def test_insert_indices(SpatialDimension, PolyDegree, LpDegree):
    exponents = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)
    number_of_monomials, dim = exponents.shape

    # TODO: is it necessary to do this for every element?
    for idx, exp_vect in enumerate(exponents):

        exponents2 = insert_lexicographically(exponents, exp_vect)
        assert_equal(exponents2, exponents)

        incomplete_exponents = np.delete(exponents, idx, axis=0)
        restored_exponents = insert_lexicographically(exponents, exp_vect)
        assert_equal(restored_exponents, exponents)

    # bigger indices should be inserted at the end
    largest_exponent_vector = exponents[-1, :]  # last / biggest exponent vector
    bigger_exponent_vector = get_lex_bigger(largest_exponent_vector)
    enlarged_exponents = insert_lexicographically(exponents, bigger_exponent_vector)
    assert_(enlarged_exponents.shape[0] == (number_of_monomials + 1))
    assert_(enlarged_exponents.shape[1] == dim)
    assert_equal(enlarged_exponents[:-1], exponents)


# --- make_complete()
def test_make_complete(SpatialDimension, PolyDegree, LpDegree):
    exponents = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)
    assert_(is_downward_closed(exponents))

    # Remove the first row and then make the exponents complete
    # NOTE: Only applies for PolyDegree > 0
    if PolyDegree > 0:
        exp_vect = exponents[0]
        incomplete_exponents = np.delete(exponents, 0, axis=0)
        # completion should be identical to "make derivable"
        # after just deleting a single exponent vector
        completed_exponents1 = make_derivable(incomplete_exponents)
        completed_exponents2 = make_complete(incomplete_exponents, LpDegree)

        assert is_downward_closed(completed_exponents1)
        assert is_downward_closed(completed_exponents2)

        assert_equal(exp_vect, completed_exponents1[0, :])
        assert_equal(exp_vect, completed_exponents2[0, :])

    # Remove the second row and the make the exponents complete
    # NOTE: Only applies for PolyDegree > 1
    #       (otherwise, the exponents are still complete)
    if PolyDegree > 1:
        exp_vect = exponents[1]
        incomplete_exponents = np.delete(exponents, 1, axis=0)
        # completion should be identical to "make derivable"
        # after just deleting a single exponent vector
        completed_exponents1 = make_derivable(incomplete_exponents)
        completed_exponents2 = make_complete(incomplete_exponents, LpDegree)

        assert is_downward_closed(completed_exponents1)
        assert is_downward_closed(completed_exponents2)

        assert_equal(exp_vect, completed_exponents1[1, :])
        assert_equal(exp_vect, completed_exponents2[1, :])

    # Insertion of a term that belongs to a higher-degree set
    # NOTE: This test is related to the fix for Issue #65
    incomplete_exponents = np.insert(
        exponents, len(exponents), exponents[-1]+2, axis=0
    )
    # Make sure that the incomplete set is indeed incomplete
    assert_(not is_downward_closed(incomplete_exponents))
    # Make the set complete
    completed_exponents = make_complete(incomplete_exponents, LpDegree)
    # Completed set must be lexicographically ordered
    assert_(is_lex_sorted(completed_exponents))
    # Completed set must be complete
    assert_(is_downward_closed(completed_exponents))


# --- make_downward_closed()
def test_make_downward_closed(SpatialDimension):
    """Test the routine to make multi-indices downward-closed.

    Notes
    -----
    - This test is related to Issue #123.
    """
    # Create a random multi-index set
    if SpatialDimension > 1:
        indices = build_rnd_exponents(SpatialDimension, 5)
        indices = lex_sort(indices)
    else:
        # Make sure the elements are not downward-closed
        indices = np.arange(100)[:, np.newaxis]
        idx = np.random.choice(len(indices), len(indices)//2, replace=False)
        indices = indices[np.sort(idx)]

    assert not is_downward_closed(indices)

    # Make downward-closed
    downward_closed = make_downward_closed(indices)

    # Assertion
    assert is_downward_closed(downward_closed)


def test_make_downward_closed_lp_degree_inf(SpatialDimension, PolyDegree):
    """Test the routine to make multi-indices downward-closed with lp inf

    Notes
    -----
    - In case the largest multi-index element has the same value across spatial
      dimensions, the corresponding downward-closed set is the same as
      the complete set with respect to the lp-degree inf and with
      the aforementioned value as the polynomial degree.
    - This test is related to Issue #123.
    """
    # Set the maximum element and make it downward-closed
    indices = PolyDegree * np.ones(SpatialDimension, dtype=int)
    indices = indices[np.newaxis, :]
    indices = make_downward_closed(indices)

    # Create the corresponding complete set
    indices_ref = get_exponent_matrix(SpatialDimension, PolyDegree, np.inf)

    # Assertion: downward-closed set is the complete set.
    assert np.all(indices == indices_ref)


def test_all_indices_are_contained(SpatialDimension, PolyDegree, LpDegree):
    exponents = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)

    # NOTE: Only applies for PolyDegree > 0 (== 0 has nothing to remove)
    # TODO: is it necessary to do this for every element?
    if PolyDegree > 0:
        for idx in range(exponents.shape[0]):
            incomplete_exponents = np.delete(exponents, idx, axis=0)
            assert_(
                not all_indices_are_contained(
                    exponents, incomplete_exponents
                )
            )
            assert_(all_indices_are_contained(incomplete_exponents, exponents))

    # when adding indices, not all indices are contained <-- ???
    largest_exponent_vector = exponents[-1, :]  # last / biggest exponent vector
    bigger_exponent_vector = get_lex_bigger(largest_exponent_vector)
    enlarged_exponents = insert_lexicographically(
        exponents, bigger_exponent_vector
    )
    assert_(not all_indices_are_contained(enlarged_exponents, exponents))
    # but the other way round should hold:
    assert_(all_indices_are_contained(exponents, enlarged_exponents))


# --- expand_dim()
def test_expand_dim_invalid_shape():
    """Test expanding the dimension of an exponent or grid array
    of an invalid shape.
    """
    # 1D
    xx = np.random.rand(10)
    # Assertion: too few dimension
    assert_raises(ValueError, expand_dim, xx, 2, None)

    # 3D
    xx = np.random.rand(10, 3).reshape(1, 3, 10)
    # Assertion: too many dimension
    assert_raises(ValueError, expand_dim, xx, xx.shape[1] + 1, None)


def test_expand_dim_contraction(SpatialDimension):
    """Test expanding the dimension of an exponent or grid array
    with an invalid target dimension.
    """
    # Create an input array
    xx = np.random.rand(100, SpatialDimension)

    # Assertion: invalid target number of columns (contraction)
    assert_raises(ValueError, expand_dim, xx, xx.shape[1] - 1, None)


def test_expand_dim_invalid_new_values(SpatialDimension):
    """Test expanding the dimension of an exponent or grid array
    with a set of invalid new values.
    """
    # Create an input array
    xx = np.random.rand(100, SpatialDimension)
    new_dim = SpatialDimension * 3
    xx_new = np.random.rand(new_dim - SpatialDimension - 1)

    # Assertion: number of new values mismatches with the expanded columns
    assert_raises(ValueError, expand_dim, xx, new_dim, xx_new)


def test_expand_dim_no_expansion(SpatialDimension):
    """Test expanding the dimension of an exponent or grid array
    to the current number of dimensions.
    """
    # Create an input array
    xx = np.random.rand(100, SpatialDimension)
    xx_expanded = expand_dim(xx, SpatialDimension)

    # Assertion: identical array because the number of columns remains
    assert xx is xx_expanded


def test_expand_dim(SpatialDimension):
    """Test expanding the dimension of an exponent or grid array."""
    # Create an input array
    num_rows = 100
    xx = np.random.rand(num_rows, SpatialDimension)

    # Expand the column
    new_dim = SpatialDimension * 3
    xx_expanded = expand_dim(xx, new_dim)

    # Assertion: By default, expanded columns are filled with 0
    assert np.all(xx_expanded[:, SpatialDimension:] == 0)


def test_expand_dim_new_values(SpatialDimension):
    """Test expanding the dimension of an exponent or grid array column
    with a given set of values.
    """
    # Create an input array
    num_rows = 100
    xx = np.random.rand(num_rows, SpatialDimension)

    # Expand the column
    new_dim = SpatialDimension * 3
    diff_cols = new_dim - SpatialDimension
    new_values = np.random.rand(diff_cols)
    xx_expanded = expand_dim(xx, new_dim, new_values)

    # Assertion
    assert np.all(xx_expanded[:, SpatialDimension:] == new_values)


# --- is_index_contained()
def test_is_index_contained(SpatialDimension, PolyDegree, LpDegree):
    """Test checking whether an entry is contained in a multi-index set."""

    # Create a complete multi-indices
    indices = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)

    # --- An index is contained in the set
    idx_rnd = int(np.random.randint(0, len(indices), 1))  # must be int
    index_rnd = indices[idx_rnd]
    # Assertion
    assert is_index_contained(indices, index_rnd)

    # --- An index is not contained in the set
    index_ref = indices[-1] + 10
    idx = search_lex_sorted(indices, index_ref)
    # Assertion: Use global indicator if an element is not found
    assert not is_index_contained(indices, index_ref)


class TestMultiplyIndices:
    """All tests related to the multiplication of array of multi-indices.

    Notes
    -----
    - These tests are related to Issue #119.
    """
    def test_same_dimension(self, SpatialDimension):
        """Multiplication of arrays of the same dimension."""
        # Create random multi-indices of the same dimension
        indices_1 = build_rnd_exponents(SpatialDimension, 5)
        indices_2 = build_rnd_exponents(SpatialDimension, 2)

        # Create a ref. from a naive implementation (only for the same dim.)
        ref_prod = []
        for index_1 in indices_1:
            for index_2 in indices_2:
                ref_prod.append(index_1 + index_2)
        ref_prod = lex_sort(np.array(ref_prod))

        # Multiply the indices
        indices_prod_1 = multiply_indices(indices_1, indices_2)
        # Commutativity check
        indices_prod_2 = multiply_indices(indices_2, indices_1)

        # Assertions
        assert np.array_equal(ref_prod, indices_prod_1)
        assert np.array_equal(ref_prod, indices_prod_2)

    def test_different_dimension(self, SpatialDimension, PolyDegree, LpDegree):
        """Multiplication of arrays of different dimensions."""
        # Randomly generate dimension difference
        diff_dim = np.random.randint(low=1, high=3)
        dim_1 = SpatialDimension
        dim_2 = SpatialDimension + diff_dim
        indices_1 = get_exponent_matrix(dim_1, PolyDegree, LpDegree)
        indices_2 = get_exponent_matrix(dim_2, PolyDegree, LpDegree)

        # Multiply indices
        indices_prod_1 = multiply_indices(indices_1, indices_2)
        # Check commutativity
        indices_prod_2 = multiply_indices(indices_2, indices_1)

        # Assertions: The additional dimensions in the product only have values
        # from the set with the higher dimension.
        unique_values_1 = np.unique(indices_prod_1[:, -diff_dim])
        unique_values_2 = np.unique(indices_prod_2[:, -diff_dim])

        assert np.all(np.unique(indices_2[:, -diff_dim]) == unique_values_1)
        assert np.all(np.unique(indices_2[:, -diff_dim]) == unique_values_2)


class TestUnionIndices:
    """All tests related to taking the union of multi-indices.

    Notes
    -----
    - These tests are related to Issue #124.
    """
    def test_same_dimension(self, SpatialDimension):
        """The union of two sets of multi-indices with the same dimension."""
        # Create random multi-indices of the same dimension
        indices_1 = build_rnd_exponents(SpatialDimension, 5)
        indices_2 = build_rnd_exponents(SpatialDimension, 2)

        # Create a reference
        ref_union = np.concatenate((indices_1, indices_2), axis=0)
        ref_union = lex_sort(ref_union)

        # Union of the indices
        indices_union_1 = union_indices(indices_1, indices_2)
        # Check commutativity
        indices_union_2 = union_indices(indices_2, indices_1)

        # Assertions
        assert np.all(ref_union == indices_union_1)
        assert np.all(ref_union == indices_union_2)

    def test_different_dimension(self, SpatialDimension, PolyDegree, LpDegree):
        """The union of two sets of multi-indices with different dimension."""
        # Randomly generate dimension difference
        diff_dim = np.random.randint(low=1, high=3)
        dim_1 = SpatialDimension
        dim_2 = SpatialDimension + diff_dim
        indices_1 = get_exponent_matrix(dim_1, PolyDegree, LpDegree)
        indices_2 = get_exponent_matrix(dim_2, PolyDegree, LpDegree)

        # Take the union of indices
        indices_union_1 = union_indices(indices_1, indices_2)
        # Check commutativity
        indices_union_2 = union_indices(indices_2, indices_1)

        # Assertions: The additional dimensions in the union only have values
        # from the set with the higher dimension.
        unique_values_1 = np.unique(indices_union_1[:, -diff_dim])
        unique_values_2 = np.unique(indices_union_2[:, -diff_dim])

        assert np.all(np.unique(indices_2[:, -diff_dim]) == unique_values_1)
        assert np.all(np.unique(indices_2[:, -diff_dim]) == unique_values_2)


# --- gen_backward_neighbors()
def test_gen_backward_neighbors(SpatialDimension, PolyDegree, LpDegree):
    """Test generating backward neighbors given a multi-index element."""
    # Create a complete multi-index set
    indices = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)

    # Select an index randomly
    idx = int(np.random.randint(0, len(indices), 1))
    index = indices[idx]

    # Non-lazy evaluation of getting the backward neighbors
    backward_neighbors_ref = index - np.eye(SpatialDimension, dtype=int)
    backward_neighbors_ref = backward_neighbors_ref[
        ~np.any(backward_neighbors_ref < 0, axis=1)
    ]

    # Lazy evaluation of getting the backward neighbors (via a generator)
    backward_neighbors = gen_backward_neighbors(index)
    pairs = zip(backward_neighbors, backward_neighbors_ref)
    for backward_neighbor, backward_neighbor_ref in pairs:
        assert np.all(backward_neighbor == backward_neighbor_ref)


# --- gen_missing_backward_neighbors()
def test_gen_missing_backward_neighbors(
    SpatialDimension, PolyDegree, LpDegree
):
    """Test generating all missing backward neighbors of a multi-index set.

    Notes
    -----
    - This test is related to the issue described in Issue #122.
    """
    # Create a complete multi-index set
    indices = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)

    # --- No missing backward neighbors
    missing_backward_neighbors = gen_missing_backward_neighbors(indices)
    assert not len(list(missing_backward_neighbors))

    # --- Remove one element
    idx = 0  # Taking the first element will cause indices not downward-closed
    index = indices[idx]
    indices_1 = np.delete(indices, idx, axis=0)

    missing_backward_neighbors = gen_missing_backward_neighbors(indices_1)

    for missing_backward_neighbor in missing_backward_neighbors:
        # Assertion: must be equal to what's removed
        assert np.all(missing_backward_neighbor == index)

    # --- Remove two elements
    if len(indices) > 2:
        idx = [0, 1]
        index = indices[idx]
        indices_2 = np.delete(indices, idx, axis=0)

        missing_backward_neighbors = gen_missing_backward_neighbors(indices_2)

        for missing_backward_neighbor in missing_backward_neighbors:
            # Assertion: must be equal to what's removed
            assert np.any(np.all(missing_backward_neighbor == index, axis=1))
