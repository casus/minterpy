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
    insert_lexicographically,
    is_lexicographically_complete,
    make_complete,
    make_derivable,
    multiply_indices,
    lex_sort,
)
from minterpy.jit_compiled_utils import (
    all_indices_are_contained,
    is_lex_sorted,
    index_is_contained,
    is_lex_smaller_or_equal,
)

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


def test_index_is_contained(SpatialDimension, PolyDegree, LpDegree):
    exponents = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)
    number_of_monomials, dim = exponents.shape
    assert_(dim == SpatialDimension)

    # TODO: is it necessary to do this for every element?
    for exponent_vector in exponents:
        assert_(index_is_contained(exponents, exponent_vector))

    largest_exponent_vector = exponents[-1, :]  # last / biggest exponent vector
    bigger_exponent_vector = get_lex_bigger(largest_exponent_vector)
    assert_(not index_is_contained(exponents, bigger_exponent_vector))

    deleted_exponent_vector = exponents[0]
    exponents2 = np.delete(exponents, 0, axis=0)
    assert_(not index_is_contained(exponents2, deleted_exponent_vector))


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
def test_lexicographical_ordering(SpatialDimension, PolyDegree, LpDegree):
    """Test checking whether a multi-index set is lexicographically ordered."""
    # --- By construction, below is lexicographically ordered
    indices = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)

    # Assertion
    assert is_lex_sorted(indices)


def test_lexicographical_ordering_single():
    """Test if a single entry multi-index set is lexicographically ordered."""
    index = np.random.randint(1, 5, (1, 10))

    # Assertion - Always, lexicographical
    assert is_lex_sorted(index)


def test_lexicographical_ordering_random():
    """Test if a random integer array is lexicographically ordered."""
    indices = np.random.randint(1, 5, (5, 10))

    # Assertion - Not lexicographical
    assert not is_lex_sorted(indices)


def test_lexicographical_ordering_duplicates():
    """Test if a multi-index set with duplicate entries is lexicographical."""
    indices = np.array([[0, 0], [2, 0], [2, 0]])

    # Assertion - Not lexicographical
    assert not is_lex_sorted(indices)


def test_is_lexicographically_complete(SpatialDimension, PolyDegree, LpDegree):
    exponents = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)
    assert_(is_lexicographically_complete(exponents))

    # NOTE: Only applies for PolyDegree > 0 (== 0 has nothing to remove)
    if PolyDegree > 0:
        # deleting the smallest vector destroys completeness
        assert_(not is_lexicographically_complete(np.delete(exponents, 0, axis=0)))

        # deleting the first row of exponents of PolyDegree > 1
        if PolyDegree > 1:
            assert_(
                not is_lexicographically_complete(
                    np.delete(exponents, 1, axis=0)
                )
            )

        # deleting the biggest vector maintains completeness
        assert_(is_lexicographically_complete(np.delete(exponents, -1, axis=0)))

    # TODO: shall not use functions which are not tested!
    # bigger_exponent_vector = expomemts[-1,:].copy()  # independent copy!
    # bigger_exponent_vector[0] += 2  # introduces "hole"
    # multi_index2 = multi_index.add_exponents(bigger_exponent_vector)
    # exponents2 = multi_index2.exponents
    # assert not is_lexicographically_complete(exponents2)
    # if multi_index2.is_complete:
    # assert not multi_index2.is_complete


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
    assert_(is_lexicographically_complete(exponents))

    # Remove the first row and then make the exponents complete
    # NOTE: Only applies for PolyDegree > 0
    if PolyDegree > 0:
        exp_vect = exponents[0]
        incomplete_exponents = np.delete(exponents, 0, axis=0)
        # completion should be identical to "make derivable"
        # after just deleting a single exponent vector
        completed_exponents1 = make_derivable(incomplete_exponents)
        completed_exponents2 = make_complete(incomplete_exponents, LpDegree)

        assert is_lexicographically_complete(completed_exponents1)
        assert is_lexicographically_complete(completed_exponents2)

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

        assert is_lexicographically_complete(completed_exponents1)
        assert is_lexicographically_complete(completed_exponents2)

        assert_equal(exp_vect, completed_exponents1[1, :])
        assert_equal(exp_vect, completed_exponents2[1, :])

    # Insertion of a term that belongs to a higher-degree set
    # NOTE: This test is related to the fix for Issue #65
    incomplete_exponents = np.insert(
        exponents, len(exponents), exponents[-1]+2, axis=0
    )
    # Make sure that the incomplete set is indeed incomplete
    assert_(not is_lexicographically_complete(incomplete_exponents))
    # Make the set complete
    completed_exponents = make_complete(incomplete_exponents, LpDegree)
    # Completed set must be lexicographically ordered
    assert_(is_lex_sorted(completed_exponents))
    # Completed set must be complete
    assert_(is_lexicographically_complete(completed_exponents))


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


# --- multiply_indices()
def test_multiply_indices_same_dim(SpatialDimension):
    """Test the multiplication of MultiIndexSet instances with the same dim.

    Notes
    -----
    - This test is related to Issue #119.
    """
    # Create random multi-indices of the same dimension
    indices_1 = build_rnd_exponents(SpatialDimension, 5)
    indices_2 = build_rnd_exponents(SpatialDimension, 2)

    # Create a reference from a naive implementation (only for the same dim.)
    ref_prod = []
    for index_1 in indices_1:
        for index_2 in indices_2:
            ref_prod.append(index_1 + index_2)
    ref_prod = lex_sort(np.array(ref_prod))

    # Multiply the indices
    indices_prod = multiply_indices(indices_1, indices_2)

    # Assertion
    assert np.all(ref_prod == indices_prod)


def test_multiply_indices_diff_dim(SpatialDimension, PolyDegree, LpDegree):
    """Test the multiplication of MultiIndexSet instances with different dims.

    With different dimension, the values of the last dimensions take the values
    from the indices of the higher dimension.

    Notes
    -----
    - This is related to Issue #119.
    """
    # --- One dimension difference
    dim_1 = SpatialDimension
    dim_2 = SpatialDimension + 1
    indices_1 = get_exponent_matrix(dim_1, PolyDegree, LpDegree)
    indices_2 = get_exponent_matrix(dim_2, PolyDegree, LpDegree)
    # Multiply the indices
    indices_prod = multiply_indices(indices_1, indices_2)
    # Assertion: The additional dimension in the product only has values
    #            from the set with the higher dimension.
    assert np.all(
        np.unique(indices_2[:, -1]) == np.unique(indices_prod[:, -1])
    )

    # --- Two dimensions difference
    dim_3 = SpatialDimension + 2
    indices_3 = get_exponent_matrix(dim_3, PolyDegree, LpDegree)
    # Multiply the indices
    indices_prod = multiply_indices(indices_1, indices_3)
    # Assertions: the additional two dimensions only have values from the set
    #             with the higher dimension.
    assert np.all(
        np.unique(indices_3[:, -2]) == np.unique(indices_prod[:, -2])
    )
    assert np.all(
        np.unique(indices_3[:, -1]) == np.unique(indices_prod[:, -1])
    )


# --- lex_sort()
def test_sort_lexicographically(SpatialDimension, PolyDegree, LpDegree):
    """Test sorting a shuffled multi-index set."""

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
