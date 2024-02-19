"""
Here we store the core utilities of `minterpy`.
"""
from __future__ import annotations

import numpy as np

from decimal import Decimal, ROUND_HALF_UP
from itertools import product
from math import ceil
from typing import Iterable, no_type_check

from minterpy.global_settings import DEFAULT_LP_DEG, INT_DTYPE, NOT_FOUND
from minterpy.jit_compiled_utils import (
    fill_match_positions,
    is_index_contained,
    is_lex_smaller_or_equal,
    search_lex_sorted,
)
from minterpy.utils import cartesian_product, lp_norm, lp_sum

# if TYPE_CHECKING:
#    from .tree import MultiIndexTree


def get_poly_degree(exponents: np.ndarray, lp_degree: float) -> int:
    """Get the polynomial degree from a multi-index set for a given lp-degree.
    
    Parameters
    ----------
    exponents : np.ndarray
        An array of exponents of a multi-index set.
    
    lp_degree : float
        The lp-degree of the multi-index set.

    Returns
    -------
    int
        The polynomial degree from the multi-index set
        for the given lp-degree.
    """
    norms = lp_norm(exponents, lp_degree, axis=1)
    max_norm = np.max(norms)
    # NOTE: Don't do round half to even (default from round())
    max_norm_int = int(Decimal(max_norm).quantize(0, ROUND_HALF_UP))
    if np.isclose(max_norm, max_norm_int):
        # The nearest integer is equivalent to the maximum norm
        # Difference are insignificant, take the integer
        return max_norm_int
    else:
        # Take the ceiling to include all index set elements
        # with smaller lp-norm
        # NOTE: math.ceil() returns int, np.ceil() returns float
        return ceil(max_norm)


def get_exponent_matrix(
    spatial_dimension: int, poly_degree: int, lp_degree: float | int
) -> np.ndarray:
    """
    Generate exponent matrix.

    :param spatial_dimension: Dimension of the domain space.
    :type spatial_dimension: int

    :param poly_degree: Polynomial degree described by the resulting exponent matrix.
    :type poly_degree: int

    :param lp_degree: Degree of the used lp-norm. This can be any integer bigger than one or `np.inf`.
    :type lp_degree: int or float

    :return: List of exponent arrays for all monomials, whose lp-norm of the exponents is smaller or equal to the given `poly_degree`. The list is given as a `np.ndarray` with the shape `(number_of_monomials, spatial_dimension)` and the list is lexicographically ordered.
    :rtype: np.ndarray

    """
    # Validate poly_degree, allow float if finite with integral value
    if isinstance(poly_degree, float):
        if not poly_degree.is_integer():
            raise ValueError(
                f"poly_degree needs to be a whole number! <{poly_degree} given."
            )

    if poly_degree == 0:
        right_choices = np.zeros((1, spatial_dimension), dtype=INT_DTYPE)

        return right_choices

    if lp_degree == np.inf:
        right_choices = cartesian_product(
            *[np.arange(poly_degree + 1, dtype=INT_DTYPE)] * spatial_dimension
        )
    else:
        candidates_without_diag = cartesian_product(
            *[np.arange(poly_degree, dtype=INT_DTYPE)] * spatial_dimension
        )
        candidates = np.vstack(
            (
                candidates_without_diag,
                np.diag([INT_DTYPE(poly_degree)] * spatial_dimension),
            )
        )
        cond = lp_sum(candidates, lp_degree) <= poly_degree ** lp_degree
        right_choices = candidates[cond]
    lex_idx = np.lexsort(right_choices.T)
    return right_choices[lex_idx]


NORM_FCT = lp_norm


def _gen_multi_index_exponents_recur(m, n, gamma, gamma2, lp_degree):
    """DEPRECATED. only for reference. TODO remove
    NOTE: this is slow for larger problem instances!

    build multi index "gamma" depending on lp_degree
    NOTE: multi indices are being called "alpha" in the newest interpolation paper

    BUG: lp = infinity does not create complete equidistant grids ((n+1)^m entries)
    very memory inefficient implementation!

    :param m:
    :param n:
    :param gamma:
    :param gamma2:
    :param lp_degree:
    :return:
    """

    # NOTE: these (and only these) copy operations are required! (tested)
    gamma2 = gamma2.copy()
    gamma0 = gamma.copy()
    gamma0[0, m - 1] += 1

    out = []
    norm = NORM_FCT(gamma0.reshape(-1), lp_degree)
    if norm < n and m > 1:
        o1 = _gen_multi_index_exponents_recur(m - 1, n, gamma, gamma, lp_degree)
        o2 = _gen_multi_index_exponents_recur(m, n, gamma0, gamma0, lp_degree)
        out = np.concatenate([o1, o2], axis=0)
    elif norm < n and m == 1:
        out = np.concatenate(
            [gamma2, _gen_multi_index_exponents_recur(m, n, gamma0, gamma0, lp_degree)],
            axis=0,
        )
    elif norm == n and m > 1:
        out = np.concatenate(
            [
                _gen_multi_index_exponents_recur(m - 1, n, gamma, gamma, lp_degree),
                gamma0,
            ],
            axis=0,
        )
    elif norm == n and m == 1:
        out = np.concatenate([gamma2, gamma0], axis=0)
    elif norm > n:
        norm_ = NORM_FCT(gamma.reshape(-1), lp_degree)
        if norm_ < n and m > 1:
            for j in range(1, m):
                gamma0 = gamma  # TODO simplify
                gamma0[j - 1] = gamma0[j - 1] + 1  # gamm0 -> 1121 broken
                if NORM_FCT(gamma0.reshape(-1), lp_degree) <= n:
                    gamma2 = np.concatenate(
                        [
                            gamma2,
                            _gen_multi_index_exponents_recur(
                                j, n, gamma0, gamma0, lp_degree
                            ),
                        ],
                        axis=0,
                    )
            out = gamma2
        elif m == 1:
            out = gamma2
        elif norm_ <= n:
            out = gamma
    return out


def _gen_multi_index_exponents(spatial_dimension, poly_degree, lp_degree):
    """
    creates the array of exponents for the MultiIndex class
    DEPRECATED: reference implementations for tests TODO remove
    """
    gamma_placeholder = np.zeros((1, spatial_dimension), dtype=INT_DTYPE)
    exponents = _gen_multi_index_exponents_recur(
        spatial_dimension, poly_degree, gamma_placeholder, gamma_placeholder, lp_degree
    )
    return exponents


def expand_dim(
    xx: np.ndarray,
    new_dim: int,
    new_values: np.ndarray = None,
) -> np.ndarray:
    """Expand the dimension of a given 2D array filled with given values.

    Parameters
    ----------
    xx : :class:`numpy:numpy.ndarray`
        Input array (exponents array or interpolating grid array) which will
        be expanded; it must be a two-dimensional array.
    new_dim : int
        The target dimension up to which the array will be expanded.
        The value must be larger than or equal to the dimension of the current
        array.
    new_values : :class:`numpy:numpy.ndarray`, optional
       The new values for the expanded dimensions; the values will be tiled
       to fill in the expanded dimensions.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        Exponents or grid array with expanded dimension (i.e., additional
        columns).

    Raises
    ------
    ValueError
        If the number of dimension of the input array is not equal to 2;
        if the target number of columns is less than the number of columns
        of the input array;
        or if the number of values for the new columns is inconsistent
        (not equal the number of rows of the input array).

    Notes
    -----
    - The term `dimension` here refers to dimensionality of exponents or
      interpolating grid; in other words, it refers to the number of columns
      of such arrays.

    Examples
    --------
    >>> array = np.array([[0, 0], [1, 0], [0, 1]])  # 2 columns / "dimensions"
    >>> expand_dim(array, 4)  # expand to 4 columns / "dimensions"
    array([[0, 0, 0, 0],
           [1, 0, 0, 0],
           [0, 1, 0, 0]])
    >>> expand_dim(array, 4, np.array([3, 2]))  # expand with tiled values
    array([[0, 0, 3, 2],
           [1, 0, 3, 2],
           [0, 1, 3, 2]])
    """
    # Check the dimension of the input array
    if xx.ndim != 2:
        raise ValueError(
            f"The exponent or grid array must be of dimension 2! "
            f"Instead got {xx.ndim}."
        )

    # Get the shape of the input array
    num_rows, num_columns = xx.shape

    # --- Dimension contraction (smaller target), raises an exception
    if new_dim < num_columns:
        # TODO maybe build a reduce fun. which removes dims where all exps 0
        raise ValueError(
            f"Can't expand the exponent or grid array dimension "
            f"from {num_columns} to {new_dim}."
        )

    # --- No dimension expansion (same target)
    if new_dim == num_columns:
        # Return the input array (identical)
        return xx

    # --- Dimension expansion
    diff_dim = new_dim - num_columns
    if new_values is None:
        new_values = np.zeros(
            (num_rows, diff_dim),
            dtype=xx.dtype
        )
    else:
        new_values = np.atleast_1d(new_values)
        if len(new_values) != diff_dim:
            raise ValueError(
                f"The given set of new values {new_values} does not have "
                f"enough elements to fill the extra columns! "
                f"<{diff_dim}> required, got <{len(new_values)}> instead."
            )

        # Tile the new values according to the shape of the input array
        new_values = np.require(
            np.tile(new_values, (num_rows, 1)), dtype=xx.dtype
        )

    return np.append(xx, new_values, axis=1)


def iterate_indices(indices: np.ndarray | Iterable[np.ndarray]) -> Iterable[np.ndarray]:
    if isinstance(indices, np.ndarray) and indices.ndim == 1:  # just a single index
        yield indices
    else:  # already iterable as is:
        yield from indices


def gen_backward_neighbors(index: np.ndarray) -> Iterable[np.ndarray]:
    """Yield the backward neighbors of a given multi-index set element.

    Parameters
    ----------
    index : :class:`numpy:numpy.ndarray`
        Multi-index set element (i.e., a vector of multi-indices),
        a one-dimensional array of length ``m``, where ``m`` is the spatial
        dimensions.

    Returns
    -------
    `Iterable` [:class:`numpy:numpy.ndarray`]
        A generator that yields all the backward neighbors (i.e., all vectors
        of multi-indices "smaller by 1") of the given multi-index element.

    Examples
    --------
    >>> my_backward_neighbors = gen_backward_neighbors(np.array([1, 1, 2]))
    >>> for my_backward_neighbor in my_backward_neighbors:
    ...     print(my_backward_neighbor)
    [0 1 2]
    [1 0 2]
    [1 1 1]
    """
    spatial_dimension = len(index)
    for m in range(spatial_dimension):
        index_in_dim = index[m]
        if index_in_dim == 0:
            # a value in a dimension is zero, ignore
            continue
        backward_neighbor = index.copy()
        backward_neighbor[m] = index_in_dim - 1

        yield backward_neighbor


def gen_missing_backward_neighbors(
    indices: np.ndarray
) -> Iterable[np.ndarray]:
    """Yield all the missing backward neighbors for an array of multi-indices.

    Parameters
    ----------
    indices : :class:`numpy:numpy.ndarray`
        Array of lexicographically sorted multi-indices, a two-dimensional
        non-negative integer array of shape ``(N, m)``, where ``N`` is
        the number of multi-indices and ``m`` is the number
        of spatial dimensions.

    Returns
    -------
    `Iterable` [:class:`numpy:numpy.ndarray`]
        A generator that yields all the missing backward neighbors
        (i.e., all the missing vectors of multi-indices "smaller by 1")
        of the array of multi-indices.

    Notes
    -----
    - This function strictly requires a lexicographically sorted array of
      multi-indices but, as a utility function, it does not check for them
      for efficiency reason. Higher-level functions that call this function
      must make sure that the array is sorted beforehand.

    Examples
    --------
    >>> my_missing_neighbors = gen_missing_backward_neighbors(np.array([[3]]))
    >>> for missing_neighbor in my_missing_neighbors:
    ...     print(missing_neighbor)
    [2]
    >>> my_missing_neighbors = gen_missing_backward_neighbors(
    ...     np.array([[1, 1, 1]])
    ... )
    >>> for missing_neighbor in my_missing_neighbors:
    ...     print(missing_neighbor)
    [0 1 1]
    [1 0 1]
    [1 1 0]
    """
    # Set up a caching system
    cache = set()

    # Loop over the indices
    nr_indices = len(indices)
    for i in reversed(range(nr_indices)):
        # Start with the lexicographically largest element
        index = indices[i]
        for backward_index in gen_backward_neighbors(index):
            # All vectors of multi-index "smaller by 1"
            indices2search = indices[:i]
            contained = is_index_contained(indices2search, backward_index)

            # Check the cache
            backward_index_tpl = tuple(backward_index)  # must be hashable
            in_cache = backward_index_tpl in cache

            if not contained and not in_cache:
                # Update the cache
                cache.add(backward_index_tpl)

                yield backward_index


def is_complete(
    indices: np.ndarray,
    poly_degree: int,
    lp_degree: float
) -> bool:
    """Check if an array of multi-indices is complete w.r.t poly- & lp-degree.

    Parameters
    ----------
    indices : :class:`numpy:numpy.ndarray`
        Array of lexicographically sorted multi-indices, a two-dimensional
        non-negative integer array of shape ``(N, m)``, where ``N`` is
        the number of multi-indices and ``m`` is the number
        of spatial dimensions.
    poly_degree : int
        Polynomial degree supported by the multi-index set.
    lp_degree : float
        :math:`p` in the :math:`l_p`-norm with respect to which
        the multi-index set is defined.

    Returns
    -------
    bool
        ``True`` if the multi-index set is complete and ``False``
        otherwise. The function also returns ``False`` if the array of
        multi-indices is not lexicographically sorted (including containing
        any duplicate entries).

    Notes
    -----
    - For a definition of completeness, refer to the relevant
      :doc:`section </how-to/multi-index-set/multi-index-set-complete>`
      of the Minterpy Documentation.

    Examples
    --------
    >>> my_indices = np.array([
    ... [0, 0, 0],
    ... [1, 0, 0],
    ... [0, 1, 0],
    ... [1, 1, 0],
    ... [0, 0, 1],
    ... [1, 0, 1],
    ... [0, 1, 1],
    ... [1, 1, 1]])
    >>> is_complete(my_indices, poly_degree=1, lp_degree=np.inf)
    True
    >>> is_complete(my_indices, poly_degree=1, lp_degree=2.0)
    False
    >>> is_complete(my_indices, poly_degree=2, lp_degree=1.0)
    False
    """
    m = indices.shape[1]  # spatial dimensions
    indices_complete = get_exponent_matrix(m, poly_degree, lp_degree)

    try:
        if np.allclose(indices, indices_complete):
            return True
    except ValueError:
        # Possibly because inconsistent shape of arrays in the comparison above
        return False

    return False


def is_disjoint(
    indices_1: np.ndarray,
    indices_2: np.ndarray,
    expand_dim_: bool = False,
) -> bool:
    """Check if an array of multi-indices is disjoint with another.

    Parameters
    ----------
    indices_1 : :class:`numpy:numpy.ndarray`
        Two-dimensional integer array of shape ``(N1, m)``, where ``N1`` is
        the number of multi-indices and ``m`` is the number
        of spatial dimensions.
    indices_2 : :class:`numpy:numpy.ndarray`
        Two-dimensional integer array of shape ``(N2, m)``, where ``N2`` is
        the number of multi-indices and ``m`` is the number
        of spatial dimensions.
    expand_dim_ : bool, optional
        Flag to allow the spatial dimension (i.e., the number of columns) of
        the indices to be expanded if there is a difference.

    Returns
    -------
    bool
        ``True`` if the multi-index sets are disjoint and ``False``
        otherwise. The function also returns ``True`` if the number of columns
        are different and ``expand_dim`` is set to ``False``.

    Examples
    --------
    >>> my_indices_1 = np.array([
    ... [0, 0, 0],
    ... [1, 0, 0],
    ... [1, 1, 1]])
    >>> my_indices_2 = np.array([
    ... [0, 0, 0],
    ... [1, 1, 1]])
    >>> is_disjoint(my_indices_1, my_indices_2)
    False
    >>> my_indices_3 = np.array([
    ... [2, 0, 1],
    ... [1, 2, 1]])
    >>> is_disjoint(my_indices_1, my_indices_3)
    True
    >>> my_indices_4 = np.array([
    ... [0, 0],
    ... [1, 0]])
    >>> is_disjoint(my_indices_1, my_indices_4)
    True
    >>> is_disjoint(my_indices_1, my_indices_4, expand_dim_=True)
    False
    """
    m_1 = indices_1.shape[1]
    m_2 = indices_2.shape[1]
    if expand_dim_:
        if m_1 < m_2:
            indices_1 = expand_dim(indices_1, m_2)
        if m_1 > m_2:
            indices_2 = expand_dim(indices_2, m_1)
    else:
        if m_1 != m_2:
            return True

    # So the search is carried out with the smaller array
    n_1 = indices_1.shape[0]
    n_2 = indices_2.shape[0]
    if n_1 > n_2:
        main_indices = indices_1
        search_indices = indices_2
    else:
        main_indices = indices_2
        search_indices = indices_1

    for search_index in search_indices:
        if search_lex_sorted(main_indices, search_index) != NOT_FOUND:
            return False

    return True


def is_downward_closed(indices: np.ndarray) -> bool:
    """Check if an array of multi-indices is downward-closed.

    Parameters
    ----------
    indices : :class:`numpy:numpy.ndarray`
        Array of lexicographically sorted multi-indices, a two-dimensional
        non-negative integer array of shape ``(N, m)``, where ``N`` is
        the number of multi-indices and ``m`` is the number
        of spatial dimensions.

    Returns
    -------
    bool
        ``True`` if the multi-index set is downward-closed and ``False``
        otherwise. The function also returns ``False`` if the array of
        multi-indices is not lexicographically sorted (including containing
        any duplicate entries).

    Notes
    -----
    - A multi-index is downward-closed if it does not contain "hole" across
      its spatial dimensions.
    - Some synonyms for a downward-closed multi-index set are:
      "monotonic", "lower", or "lexicographically complete".

    Examples
    --------
    >>> my_indices = np.array([
    ... [0, 0, 0],
    ... [1, 0, 0],
    ... [0, 0, 1]])
    >>> is_downward_closed(my_indices)
    True
    >>> my_indices = np.array([
    ... [0, 0, 0],
    ... [1, 0, 0],
    ... [0, 2, 0]])  # missing [0, 1, 0]
    >>> is_downward_closed(my_indices)
    False
    """
    for _ in gen_missing_backward_neighbors(indices):
        # A "hole" is found, i.e., a missing backward neighbor
        return False

    return True


@no_type_check
def list_insert_single(
    list_of_indices: list[np.ndarray],
    index2insert: np.ndarray,
    check_for_inclusion: bool = True,
):
    """inserts a single index into a given list of indices maintaining lexicographical ordering"""
    # TODO check length of single index to insert!
    if list_of_indices is None:
        raise TypeError
    nr_of_indices = len(list_of_indices)
    if nr_of_indices == 0:
        list_of_indices.insert(0, index2insert)
        return

    insertion_idx = (
        nr_of_indices  # default: insert at the last position, ATTENTION: -1 not working
    )
    for i, contained_index in enumerate(list_of_indices):
        if is_lex_smaller_or_equal(index2insert, contained_index):
            insertion_idx = i
            break

    # no insertion when an equal entry exists already
    # TODO check_for_inclusion
    if not np.array_equal(contained_index, index2insert):
        list_of_indices.insert(insertion_idx, index2insert)


@no_type_check
def to_index_list(indices: np.ndarray | Iterable[np.ndarray]) -> list[np.ndarray]:
    if type(indices) is list:
        return indices  # already is a list
    list_of_indices = list(iterate_indices(indices))  # include all existing indices
    return list_of_indices


def to_index_array(list_of_indices: list[np.ndarray]) -> np.ndarray:
    # NOTE: shape is: (N, m)
    index_array = np.array(list_of_indices, dtype=INT_DTYPE)
    return index_array


# def list_based(func):
#     """ TODO decorator for working with list of indices rather than with static numpy arrays"""
#     @functools.wraps(func)
#     def list_based_wrapper(*args, **kwargs):
#         to_index_list(
#         value = func(*args, **kwargs)
#         to_index_array(
#         return value
#     return list_based_wrapper


def insert_lexicographically(
    indices: list[np.ndarray] | np.ndarray,
    indices2insert: Iterable[np.ndarray] | None,
) -> np.ndarray:
    """inserts possibly multiple index vectors into a given array of indices maintaining lexicographical ordering

    exploits ordering for increased efficiency:
     inserts one vector after another to maintain ordering

    NOTE: is expected to return the same identical array instance
     if all exponents are already contained!
    """
    if indices2insert is None:
        return indices
    nr_exponents = len(indices)
    list_of_indices = None  # avoid creating a list when there is no index to insert
    for i, index2insert in enumerate(iterate_indices(indices2insert)):
        if i == 0:  # initialise list
            list_of_indices = to_index_list(indices)
        list_insert_single(list_of_indices, index2insert)
    if (
        list_of_indices is None or len(list_of_indices) == nr_exponents
    ):  # len(list_of_indices) or len(indices)
        # no index has been inserted.
        # ATTENTION: return the previous array in order to easily compare for equality!
        return indices
    index_array = to_index_array(list_of_indices)
    return index_array


def insert_partial_derivatives(list_of_indices, exponent_vector):
    for deriv_vect in gen_backward_neighbors(
        exponent_vector
    ):  # all vectors "smaller by 1"
        list_insert_single(list_of_indices, deriv_vect)


def make_derivable(indices: np.ndarray) -> np.ndarray:
    """inserts all missing multi index vectors "smaller by one" """
    list_of_indices: list[int] = []
    nr_exponents, spatial_dimension = indices.shape
    for i in reversed(range(nr_exponents)):  # start with the biggest multi index
        contained_exponent_vector = indices[i, :]
        list_insert_single(list_of_indices, contained_exponent_vector)
        insert_partial_derivatives(list_of_indices, contained_exponent_vector)
    index_array = to_index_array(list_of_indices)
    return index_array


def make_complete(exponents: np.ndarray, lp_degree: float) -> np.ndarray:
    """Create a complete exponents from a given array of exponents.

    A complete set of exponents contains all monomials, whose :math:`l_p`-norm
    of the exponents are smaller or equal to the polynomial degree of the set.

    Parameters
    ----------
    exponents : :class:`numpy:numpy.ndarray`
        A given exponent array to be completed.
    lp_degree : :py:class:`float`
        A given :math:`l_p` of the :math:`l_p`-norm (i.e., the lp-degree)
        for the completion.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The complete exponents with respect to the given ``exponents`` and
        ``lp_degree``.

    Notes
    -----
    - The polynomial degree is inferred from the given exponents with
      respect to the required ``lp_degree``. It is the smallest polynomial
      degree with respect to the ``lp_degree`` to contain the given exponents.

    Examples
    --------
    >>> exponent = np.array([[2, 2]])
    >>> lp_degree = 2.0
    >>> make_complete(exponent, lp_degree)  # Complete w.r.t the lp_degree
    array([[0, 0],
           [1, 0],
           [2, 0],
           [3, 0],
           [0, 1],
           [1, 1],
           [2, 1],
           [0, 2],
           [1, 2],
           [2, 2],
           [0, 3]])
    """
    poly_degree = get_poly_degree(exponents, lp_degree)
    spatial_dimension = exponents.shape[-1]

    complete_exponents = get_exponent_matrix(
        spatial_dimension, poly_degree, lp_degree
    )

    return complete_exponents


def make_downward_closed(indices: np.ndarray) -> np.ndarray:
    """Make an array of multi-indices downward-closed.

    Parameters
    ----------
    indices : :class:`numpy:numpy.ndarray`
        Array of lexicographically sorted multi-indices, a two-dimensional
        non-negative integer array of shape ``(N, m)``, where ``N`` is
        the number of multi-indices and ``m`` is the number of spatial
        dimensions.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        Array of downward-closed lexicographically sorted multi-indices,
        a two-dimensional array of shape ``(N1, m)``, where ``N1`` is
        the number of multi-indices in the downward-closed set and
        ``m`` is the number of spatial dimensions.

    Notes
    -----
    - This function strictly requires a lexicographically sorted array of
      multi-indices but, as a utility function, it does not check for them
      for efficiency reason. Higher-level functions that call this function
      must make sure that the array is sorted beforehand.

    Examples
    --------
    >>> my_indices = np.array([
    ... [0, 0],
    ... [2, 0],  # Jump from [1, 0]
    ... [0, 3],  # Jump from [0, 1] and [0, 2]
    ... ])
    >>> make_downward_closed(my_indices)
    array([[0, 0],
           [1, 0],
           [2, 0],
           [0, 1],
           [0, 2],
           [0, 3]])
    >>> my_indices = np.array([[1, 1, 1]])  # must be two-dimensional
    >>> make_downward_closed(my_indices)
    array([[0, 0, 0],
           [1, 0, 0],
           [0, 1, 0],
           [1, 1, 0],
           [0, 0, 1],
           [1, 0, 1],
           [0, 1, 1],
           [1, 1, 1]])
    """
    # Set up a set to store final indices while avoiding duplications
    indices_set = set([tuple(index) for index in indices])

    # Initialize the primary iteration condition (input indices)
    backward_neighbors = indices
    # Set the early break parameter
    break_length = -np.inf
    while not is_downward_closed(backward_neighbors):
        # All missing backward neighbors will eventually become
        # downward-closed; this is a cheaper break condition
        # than checking for the downward-closeness of the whole set
        # everytime the set is updated with the missing backward neighbors.

        # Backward neighbors of the current backward neighbors
        backward_neighbors = _missing_backward_neighbors(backward_neighbors)

        # Update the multi-index set
        indices_tuple = [tuple(index) for index in backward_neighbors]
        indices_set.update(indices_tuple)

        # Early breaking: If the length of indices remains the same after
        # two consecutive iterations.
        if break_length == len(indices_set):
            break
        break_length = len(indices_set)

    # Make sure the downward-indices are lexicographically sorted.
    indices_out = lex_sort(np.array(list(indices_set)))

    return indices_out


def find_match_between(
    smaller_idx_set: np.ndarray, larger_idx_set: np.ndarray
) -> np.ndarray:
    """find the positions of each multi_index of the smaller set in the larger set

    NOTE: both sets are required to be ordered lexicographically!
    NOTE: the smaller set is required to be a subset of the larger set
    NOTE: this function is required for 'rebasing' a polynomial onto complete multi indices (map the coefficients)
    """
    nr_exp_smaller, spatial_dimension = smaller_idx_set.shape
    positions = np.empty(nr_exp_smaller, dtype=INT_DTYPE)
    fill_match_positions(larger_idx_set, smaller_idx_set, positions)
    return positions


def lex_sort(indices: np.ndarray) -> np.ndarray:
    """Lexicographically sort an array of multi-indices.

    Parameters
    ----------
    indices : :class:`numpy:numpy.ndarray`
        Two-dimensional array of multi-indices to be sorted.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        Lexicographically sorted array, having the same type as ``indices``.
        Only unique entries are kept in the sorted array.

    Examples
    --------
    >>> xx = np.array([
    ... [0, 1, 2, 3],
    ... [0, 1, 0, 1],
    ... [0, 0, 0, 0],
    ... [0, 0, 1, 1],
    ... [0, 0, 0, 0],
    ... ])
    >>> lex_sort(xx)  # Sort and remove duplicates
    array([[0, 0, 0, 0],
           [0, 1, 0, 1],
           [0, 0, 1, 1],
           [0, 1, 2, 3]])
    """
    indices_unique = np.unique(indices, axis=0)
    indices_lex_sorted = indices_unique[np.lexsort(indices_unique.T)]

    return indices_lex_sorted


def multiply_indices(
    indices_1: np.ndarray,
    indices_2: np.ndarray
) -> np.ndarray:
    """Multiply an array of multi-indices with another array of multi-indices.

    Parameters
    ----------
    indices_1 : :class:`numpy:numpy.ndarray`
        Two-dimensional array of multi-indices of shape ``(N1, m2)`` where
        ``N1`` is the number of multi-indices and ``m2`` is the number of
        dimensions of the first array.
    indices_2 : :class:`numpy:numpy.ndarray`
        Another two-dimensional array of multi-indices of shape ``(N2, m2)``
        where ``N2`` is the number of multi-indices and ``m2`` is the number
        of dimensions of the second array.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The product of ``indices_1`` with ``indices_2`` of shape ``(N3, m3)``
        where ``m3`` is the maximum between ``m1`` and ``m2``. The product
        array is lexicographically sorted.

    Examples
    --------
    >>> my_indices_1 = np.array([
    ... [0, 0],
    ... [1, 0],
    ... [0, 1],
    ... [1, 1],
    ... ])
    >>> my_indices_2 = np.array([
    ... [0, 0],
    ... [1, 0],
    ... [0, 1],
    ... ])
    >>> multiply_indices(my_indices_1, my_indices_2)
    array([[0, 0],
           [1, 0],
           [2, 0],
           [0, 1],
           [1, 1],
           [2, 1],
           [0, 2],
           [1, 2]])
    >>> my_indices_3 = np.array([
    ... [0],
    ... [1],
    ... [2],
    ... ])
    >>> multiply_indices(my_indices_2, my_indices_3)  # Different dimension
    array([[0, 0],
           [1, 0],
           [2, 0],
           [3, 0],
           [0, 1],
           [1, 1],
           [2, 1]])
    >>> my_indices_4 = np.empty((0, 2))
    >>> multiply_indices(my_indices_3, my_indices_4)  # empty set
    array([], shape=(0, 2), dtype=int32)
    """
    # --- Adjust the dimension
    m_1 = indices_1.shape[1]
    m_2 = indices_2.shape[1]
    if m_1 < m_2:
        indices_1 = expand_dim(indices_1, m_2)
    if m_1 > m_2:
        indices_2 = expand_dim(indices_2, m_1)

    # --- Take the cross product (maybe expensive for large arrays of indices)
    prod = list(product(indices_1, indices_2))
    # The product may be empty
    prod = np.array([np.sum(np.array(i), axis=0) for i in prod])
    if len(prod) != 0:
        prod = lex_sort(prod)
    else:
        prod = np.empty((0, np.max([m_1, m_2])), dtype=INT_DTYPE)

    return prod


def union_indices(
    indices_1: np.ndarray,
    indices_2: np.ndarray,
) -> np.ndarray:
    """Create a union between two arrays of multi-indices.

    Parameters
    ----------
    indices_1 : :class:`numpy:numpy.ndarray`
        Two-dimensional array of multi-indices of shape ``(N1, m2)`` where
        ``N1`` is the number of multi-indices and ``m2`` is the number of
        dimensions of the first array.
    indices_2 : :class:`numpy:numpy.ndarray`
        Another two-dimensional array of multi-indices of shape ``(N2, m2)``
        where ``N2`` is the number of multi-indices and ``m2`` is the number
        of dimensions of the second array.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The union of ``indices_1`` and ``indices_2`` of shape ``(N3, m3)``
        where ``m3`` is the maximum between ``m1`` and ``m2``.
        The union is lexicographically sorted.

    Examples
    --------
    >>> my_indices_1 = np.array([
    ... [0, 0],
    ... [1, 0],
    ... [0, 1],
    ... [1, 1],
    ... ])
    >>> my_indices_2 = np.array([
    ... [0, 0],
    ... [1, 0],
    ... [0, 1],
    ... ])
    >>> union_indices(my_indices_1, my_indices_2)
    array([[0, 0],
           [1, 0],
           [0, 1],
           [1, 1]])
    >>> my_indices_3 = np.array([
    ... [0],
    ... [1],
    ... [2],
    ... ])
    >>> union_indices(my_indices_1, my_indices_3)  # Different dimension
    array([[0, 0],
           [1, 0],
           [2, 0],
           [0, 1],
           [1, 1]])
    >>> my_indices_4 = np.empty((0, 1), dtype=int)
    >>> union_indices(my_indices_3, my_indices_4)
    array([[0],
           [1],
           [2]])
    """
    # --- Adjust the dimension
    m_1 = indices_1.shape[1]
    m_2 = indices_2.shape[1]
    if m_1 < m_2:
        indices_1 = expand_dim(indices_1, m_2)
    if m_1 > m_2:
        indices_2 = expand_dim(indices_2, m_1)

    # --- Take the union between the two indices then lexicographically sort it
    union = np.concatenate((indices_1, indices_2), axis=0)
    union = lex_sort(union)

    return union


def _missing_backward_neighbors(indices: np.ndarray) -> np.ndarray:
    """Create a lexicographically sorted array of missing backward neighbors.

    Parameters
    ----------
    indices : :class:`numpy:numpy.ndarray`
        Array of lexicographically sorted multi-indices, a two-dimensional
        non-negative integer array of shape ``(N, m)``, where ``N`` is
        the number of multi-indices and ``m`` is the number of spatial
        dimensions.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        Array of missing backward neighbors (i.e., all the missing vectors
        of multi-indices "smaller by 1") of the array of multi-indices.
        If ``indices`` has no backward neighbor, an empty array is returned.
    """
    backward_neighbors = gen_missing_backward_neighbors(indices)
    # Convert generator to a NumPy array
    backward_neighbors = np.array(list(backward_neighbors))
    if len(backward_neighbors) > 0:
        return lex_sort(backward_neighbors)

    return backward_neighbors


if __name__ == "__main__":
    import doctest
    doctest.testmod()
