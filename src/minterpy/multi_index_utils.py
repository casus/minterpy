from math import gamma
from typing import Iterable, List, Optional, Union
from warnings import warn

import numpy as np
from scipy.special import binom

from minterpy.global_settings import DEFAULT_LP_DEG, INT_DTYPE
from minterpy.jit_compiled_utils import (fill_exp_matrix, fill_match_positions,
                                         index_is_contained,
                                         lex_smaller_or_equal)
from minterpy.utils import lp_norm


def _get_poly_degree(exponents, lp):
    norms = lp_norm(exponents, lp, axis=1)
    return norms.max()


def lp_hypersphere_vol(spacial_dimension, radius, p):
    # https://en.wikipedia.org/wiki/Volume_of_an_n-ball#Balls_in_Lp_norms
    n = spacial_dimension
    r = radius
    return (2 * gamma(1 / p + 1) * r) ** n / gamma(n / p + 1)


def get_exponent_matrix(
    spatial_dimension: int, poly_degree: int, lp_degree: Union[float, int]
) -> np.ndarray:
    """creates an array of "multi-indices" symmetric in each dimension

    pre allocate the right amount of memory
    then starting from the 0 vector, lexicographically "count up" and add all valid exponent vectors

    NOTE: this has only been tested for up to dim 4 and up to deg 5.
        be aware that the preallocated array might be too small
        (<- will raise an IndexError by accessing the array out of bounds)!

    :param spatial_dimension: the dimensionality m
    :param poly_degree: the highest exponent which should occur in the exponent matrix
    :param lp_degree: the value for p of the l_p-norm
    :return: an array of shape (m, x) with all possible exponent vectors v with: l_p-norm(v) < poly_degree
        sorted lexicographically.
    """
    n = poly_degree
    m = spatial_dimension
    p = lp_degree
    p = float(p)  # float dtype expected

    max_nr_exp = (n + 1) ** m
    if p < 0.0:
        raise ValueError("values for p must be larger than 0.")
    # for some special cases it is known how many distinct exponent vectors exist:
    if p == 1.0:
        nr_expected_exponents = binom(m + n, n)
    elif p == np.inf or m == 1 or n == 1:
        # in dimension 1 all lp-degrees are equal!
        nr_expected_exponents = max_nr_exp
    else:
        # for values p << 2 the estimation based on the hypersphere volume tends to underestimate
        # the required amount of space! -> clip values for estimation.
        # TODO find more precise upper bound.
        # -> increases the volume in order to make sure there truly is enough memory being allocated
        p_estimation = max(p, 1.7)
        vol_lp_sphere = lp_hypersphere_vol(m, radius=1.0, p=p_estimation)
        vol_hypercube = 2 ** m
        vol_fraction = vol_lp_sphere / vol_hypercube
        nr_expected_exponents = max_nr_exp * vol_fraction

    nr_expected_exponents = int(nr_expected_exponents)
    if nr_expected_exponents > 1e6:
        raise ValueError(
            f"trying to generate exponent matrix with {max_nr_exp} entries."
        )
    exponents = np.empty((nr_expected_exponents, m), dtype=INT_DTYPE)
    nr_filled_exp = fill_exp_matrix(exponents, p, n)

    # NOTE: validity checked by tests:
    # if nr_expected_exponents == nr_filled_exp and lp_degree != np.inf:
    #     raise ValueError('potentially not enough memory has been allocated to fit all valid exponent vectors! '
    #                      'check increasing the `incr_factor`')
    # just use relevant part:
    out = exponents[:nr_filled_exp, :]
    #  NOTE: since tiny_array is a view onto huge_array, so long as a reference to tiny_array exists the full
    #  big memory allocation will remain. creating an independent copy however will take up resources!
    #  cf. http://numpy-discussion.10968.n7.nabble.com/Numpy-s-policy-for-releasing-memory-td1533.html
    #  NOTE: will be slow for very large arrays. since everything needs not be copied!
    if nr_filled_exp * 2 < nr_expected_exponents:
        warn(
            f"more than double the required memory has been allocated ({nr_filled_exp} filled "
            f"<< {nr_expected_exponents} allocated). inefficient!"
        )
        out = out.copy()  # independent copy
        del exponents  # free up memory
    return out


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


def _expand_dim(exps, target_dim):
    """
    Expansion of exponents with zeros up to a given dimension

    Parameters
    ----------
    exps : array_like (shape=(dim,len_mi))
        Array of exponents from MultiIndex.
    target_dim : np.int
        Dimension up to the exponents shall be expanded to. Needs to be bigger or equal than the current dimension of exps.
    """
    mi_len, dim = exps.shape
    exps_dtype = exps.dtype
    if target_dim < dim:
        # TODO maybe build a reduce function which removes dims where all exps are 0
        raise ValueError(
            f"Can't expand multi_index from dim {dim} to dim {target_dim}."
        )
    if target_dim == dim:
        return exps
    num_expand_dim = target_dim - dim
    new_dim_exps = np.zeros((mi_len, num_expand_dim), dtype=exps_dtype)
    return np.concatenate((exps, new_dim_exps), axis=1)


def iterate_indices(
    indices: Union[np.ndarray, Iterable[np.ndarray]]
) -> Iterable[np.ndarray]:
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


def get_partial_derivatives(exponent_vector: np.ndarray) -> np.ndarray:
    """compiles the exponent vectors of all partial derivatives of a given exponent vector"""
    return np.array(list(gen_partial_derivatives(exponent_vector)), dtype=INT_DTYPE)


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


def list_insert_single(
    list_of_indices: List[np.ndarray],
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


def to_index_list(indices: Union[np.ndarray, Iterable[np.ndarray]]) -> List[np.ndarray]:
    if type(indices) is list:
        return indices  # already is a list
    list_of_indices = list(iterate_indices(indices))  # include all existing indices
    return list_of_indices


def to_index_array(list_of_indices: List[np.ndarray]) -> np.ndarray:
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
    indices: Union[List[np.ndarray], np.ndarray],
    indices2insert: Optional[Iterable[np.ndarray]],
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


def sort_lexicographically(indices: Iterable[np.ndarray]) -> np.ndarray:
    return insert_lexicographically([], indices)


def insert_partial_derivatives(list_of_indices, exponent_vector):
    for deriv_vect in gen_partial_derivatives(
        exponent_vector
    ):  # all vectors "smaller by 1"
        list_insert_single(list_of_indices, deriv_vect)


def make_derivable(indices: np.ndarray) -> np.ndarray:
    """inserts all missing multi index vectors "smaller by one" """
    list_of_indices = []
    nr_exponents, spatial_dimension = indices.shape
    for i in reversed(range(nr_exponents)):  # start with the biggest multi index
        contained_exponent_vector = indices[i, :]
        list_insert_single(list_of_indices, contained_exponent_vector)
        insert_partial_derivatives(list_of_indices, contained_exponent_vector)
    index_array = to_index_array(list_of_indices)
    return index_array


def make_complete(indices: np.ndarray) -> np.ndarray:
    """inserts ALL possible missing smaller multi index vectors

    -> fills up the "holes"
    ATTENTION: calling this with a complete index set is very inefficient!
    """
    list_of_indices = to_index_list(indices)
    ptr = -1  # start with the biggest multi index
    while (
        1
    ):  # add the partial derivatives of the vector at the current pointer (will be added in front)
        exp2check = list_of_indices[ptr]
        list_insert_single(list_of_indices, exp2check)  # insert the vector itself
        insert_partial_derivatives(
            list_of_indices, exp2check
        )  # insert the its partial derivatives
        first_entry_checked = len(list_of_indices) + ptr == 0
        if first_entry_checked:  # stop when the first entry has been processed
            break
        ptr -= 1  # move to the next smaller index and repeat
    index_array = to_index_array(list_of_indices)
    return index_array


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
