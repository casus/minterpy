"""
Here we store the core utilities of `minterpy`.
"""
from __future__ import annotations

from typing import Iterable, no_type_check

import numpy as np
from math import ceil

from minterpy.global_settings import DEFAULT_LP_DEG, INT_DTYPE
from minterpy.jit_compiled_utils import (
    fill_match_positions,
    index_is_contained,
    lex_smaller_or_equal,
)
from minterpy.utils import cartesian_product, lp_norm, lp_sum

# if TYPE_CHECKING:
#    from .tree import MultiIndexTree


def _get_poly_degree(exponents: np.ndarray, lp_degree: float) -> int:
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
    # NOTE: math.ceil() returns int, np.ceil() returns float
    return ceil(np.max(norms))

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


def _expand_dim(grid_nodes, target_dim, point_pinned=None):
    """
    Expansion of a given array to a given dimension, where the additional dimensions will filled with tiles of given elements

    Parameters
    ----------
    grid_nodes : array_like (shape=(len_arr,dim))
        Array which shall be expanded.
    target_dim : np.int
        Dimension up to the array shall be expanded to. Needs to be bigger or equal than the current dimension of array.
    point_pinned : optional np.ndarray
    """
    grid_len, dim = grid_nodes.shape
    grid_dtype = grid_nodes.dtype
    if target_dim < dim:
        # TODO maybe build a reduce function which removes dims where all exps are 0
        raise ValueError(f"Can't expand grid from dim {dim} to dim {target_dim}.")
    if target_dim == dim:
        return grid_nodes
    num_expand_dim = target_dim - dim
    if point_pinned is None:
        new_dim_exps = np.zeros((grid_len, num_expand_dim), dtype=grid_dtype)
    else:
        point_pinned = np.atleast_1d(point_pinned)
        if len(point_pinned) is not num_expand_dim:
            raise ValueError(
                f"Given point_pinned {point_pinned} has not enough elements to fill the extra dimensions! <{num_expand_dim}> required."
            )
        new_dim_exps = np.require(
            np.tile(point_pinned, (grid_len, 1)), dtype=grid_dtype
        )
    return np.concatenate((grid_nodes, new_dim_exps), axis=1)


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


def make_complete(indices: np.ndarray, lp_degree: float = None) -> np.ndarray:
    """Make a given array of exponents complete.

    :param indices: The exponent array to be completed.
    :type indices: np.ndarray
    :param lp_degree: lp-degree for the completation. Optional, the default is given by `minterpy.DEFAULT_LP_DEG`.
    :type lp_degree: int (optional)

    :return: Completed version of the input exponents.
    :rtype: np.ndarray

    """
    if lp_degree is None:
        lp_degree = DEFAULT_LP_DEG
    poly_degree = _get_poly_degree(indices, lp_degree)
    spatial_dimension = indices.shape[-1]
    return get_exponent_matrix(spatial_dimension, poly_degree, lp_degree)


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


def verify_lp_deg(lp_degree):
    if lp_degree is None:
        lp_degree = DEFAULT_LP_DEG
    # TODO remove
    # elif lp_degree != DEFAULT_LP_DEG:
    #     if lp_degree % 1.0 != 0.0:
    #         raise ValueError('the given l_p-degree cannot be interpreted as integer')
    #     lp_degree = int(lp_degree)
    return lp_degree
