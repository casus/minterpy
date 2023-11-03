"""
Here we store the core utilities of `minterpy`.
"""
from __future__ import annotations

from typing import Iterable, no_type_check

import numpy as np
from math import ceil
from decimal import Decimal, ROUND_HALF_UP

from minterpy.global_settings import DEFAULT_LP_DEG, INT_DTYPE
from minterpy.jit_compiled_utils import (
    fill_match_positions,
    index_is_contained,
    lex_smaller_or_equal,
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


def gen_partial_derivatives(exponent_vector: np.ndarray):
    """yields the exponent vectors of all partial derivatives of a given exponent vector

    NOTE: all exponent vectors "smaller by 1"
    """
    spatial_dimension = len(exponent_vector)
    for m in range(spatial_dimension):
        exponent_in_dim = exponent_vector[m]
        if exponent_in_dim == 0:  # partial derivative is 0
            continue
        partial_derivative = exponent_vector.copy()
        partial_derivative[m] = exponent_in_dim - 1
        yield partial_derivative


def gen_missing_derivatives(indices: np.ndarray) -> Iterable[np.ndarray]:
    """yields all the partial derivative exponent vectors missing from the given set of multi indices

    ATTENTION: the input indices must have lexicographical ordering
    ATTENTION: duplicate partial derivatives will be generated
    """
    nr_exponents, spatial_dimension = indices.shape
    for n in reversed(
        range(nr_exponents)
    ):  # start with the lexicographically biggest index
        exp_vect = indices[n, :]
        for deriv_vect in gen_partial_derivatives(
            exp_vect
        ):  # all vectors "smaller by 1"
            # NOTE: looking for the index starting from the last index would be faster TODO
            indices2search = indices[:n, :]
            if not index_is_contained(indices2search, deriv_vect):
                yield deriv_vect


def is_lexicographically_complete(indices: np.ndarray) -> bool:
    """tells weather an array of indices contains all possible lexicographically smaller indices

    ATTENTION: the input indices must have lexicographical ordering
    :returns False if there is a missing multi index vector "smaller by 1"
    """
    for _ in gen_missing_derivatives(indices):
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
        if lex_smaller_or_equal(index2insert, contained_index):
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
    for deriv_vect in gen_partial_derivatives(
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
