"""
Module containing several numba optimized functions.
"""

import numpy as np
from numba import b1, njit, void

from minterpy.global_settings import (
    B_TYPE,
    F_1D,
    F_2D,
    FLOAT,
    I_1D,
    I_2D,
    INT,
    FLOAT_DTYPE,
    NOT_FOUND,
)


@njit(void(F_2D, F_2D, I_2D, F_2D), cache=True)
def can_eval_mult(x_multiple, coeffs, exponents, result_placeholder):
    """Naive evaluation of polynomials in canonical basis.

    - ``m`` spatial dimension
    - ``k`` number of points
    - ``N`` number of monomials
    - ``p`` number of polynomials

    :param x_multiple: numpy array with coordinates of points where polynomial is to be evaluated.
                       The shape has to be ``(k x m)``.
    :param coeffs: numpy array of polynomial coefficients in canonical basis. The shape has to be ``(N x p)``.
    :param exponents: numpy array with exponents for the polynomial. The shape has to be ``(N x m)``.
    :param result_placeholder: placeholder numpy array where the results of evaluation are stored.
                               The shape has to be ``(k x p)``.

    Notes
    -----
    This is a naive evaluation; a more numerically accurate approach would be to transform to Newton basis and
    using the newton evaluation scheme.

    Multiple polynomials in the canonical basis can be evaluated at once by having a 2D coeffs array. It is assumed
    that they all have the same set of exponents.

    """
    nr_coeffs, nr_polys = coeffs.shape
    r = result_placeholder
    nr_points, _ = x_multiple.shape
    for i in range(nr_coeffs):  # each monomial
        exp = exponents[i, :]
        for j in range(nr_points):  # evaluated on each point
            x = x_multiple[j, :]
            monomial_value = np.prod(np.power(x, exp))
            for k in range(nr_polys):  # reuse intermediary results
                c = coeffs[i, k]
                r[j, k] += c * monomial_value


# NOTE: the most "fine grained" functions must be defined first
# in order for Numba to properly infer the function types

@njit(void(F_1D, I_2D, F_2D, I_1D, F_2D, F_1D), cache=True)  # O(Nm)
def eval_newton_monomials_single(
    x_single,
    exponents,
    generating_points,
    max_exponents,
    products_placeholder,
    monomials_placeholder,
):
    """Precomputes the value of all given Newton basis polynomials at a point.

    Core of the fast polynomial evaluation algorithm.
    - ``m`` spatial dimension
    - ``N`` number of monomials
    - ``n`` maximum exponent in each dimension

    :param x_single: coordinates of the point. The shape has to be ``m``.
    :param exponents: numpy array with exponents for the polynomial. The shape has to be ``(N x m)``.
    :param generating_points: generating points used to generate the grid. The shape is ``(n x m)``.
    :param max_exponents: array with maximum exponent in each dimension. The shape has to be ``m``.
    :param products_placeholder: a numpy array for storing the (chained) products.
    :param monomials_placeholder: a numpy array of length N for storing the values of all Newton basis polynomials.

    Notes
    -----
    Precompute all the (chained) products required during newton evaluation. Complexity is ``O(mN)``.
    This precomputation is coefficient agnostic.
    Results are only stored in the placeholder arrays. Nothing is returned.
    """

    # NOTE: the maximal exponent might be different in every dimension,
    #    in this case the matrix becomes sparse (towards the end)
    # NOTE: avoid index shifting during evaluation (has larger complexity than pre-computation!)
    #    by just adding one empty row in front. ATTENTION: these values must not be accessed!
    #    -> the exponents of each monomial ("alpha") then match the indices of the required products

    # Create the products matrix
    m = exponents.shape[1]
    for i in range(m):
        max_exp_in_dim = max_exponents[i]
        x_i = x_single[i]
        prod = 1.0
        for j in range(max_exp_in_dim):  # O(n)
            # TODO there are n+1 1D grid values, the last one will never be used!?
            p_ij = generating_points[j, i]
            prod *= x_i - p_ij
            # NOTE: shift index by one
            exponent = j + 1  # NOTE: otherwise the result type is float
            products_placeholder[exponent, i] = prod

    # evaluate all Newton polynomials. O(Nm)
    N = exponents.shape[0]
    for j in range(N):
        # the exponents of each monomial ("alpha")
        # are the indices of the products which need to be multiplied
        newt_mon_val = 1.0  # required as multiplicative identity
        for i in range(m):
            exp = exponents[j, i]
            # NOTE: an exponent of 0 should not cause a multiplication
            # (inefficient, numerical instabilities)
            if exp > 0:
                newt_mon_val *= products_placeholder[exp, i]
        monomials_placeholder[j] = newt_mon_val
    #NOTE: results have been stored in the numpy arrays. no need to return anything.


@njit(void(F_2D, I_2D, F_2D, I_1D, F_2D, F_2D, B_TYPE), cache=True)
def eval_newton_monomials_multiple(
    xx: np.ndarray,
    exponents: np.ndarray,
    generating_points: np.ndarray,
    max_exponents: np.ndarray,
    products_placeholder: np.ndarray,
    monomials_placeholder: np.ndarray,
    triangular: bool
):
    """Evaluate the Newton monomials at multiple query points.

    The following notations are used below:

    - :math:`m`: the spatial dimension of the polynomial
    - :math:`p`: the (maximum) degree of the polynomial in any dimension
    - :math:`n`: the number of elements in the multi-index set (i.e., monomials)
    - :math:`\mathrm{nr_{points}}`: the number of query (evaluation) points
    - :math:`\mathrm{nr_polynomials}`: the number of polynomials with different
      coefficient sets of the same multi-index set

    :param xx: numpy array with coordinates of points where polynomial is to be evaluated.
              The shape has to be ``(k x m)``.
    :param exponents: numpy array with exponents for the polynomial. The shape has to be ``(N x m)``.
    :param generating_points: generating points used to generate the grid. The shape is ``(n x m)``.
    :param max_exponents: array with maximum exponent in each dimension. The shape has to be ``m``.
    :param products_placeholder: a numpy array for storing the (chained) products.
    :param monomials_placeholder: placeholder numpy array where the results of evaluation are stored.
                               The shape has to be ``(k x p)``.
    :param triangular: whether the output will be of lower triangular form or not.
                       -> will skip the evaluation of some values
    :return: the value of each Newton polynomial at each point. The shape will be ``(k x N)``.

    Notes
    -----
    - The memory footprint for evaluating the Newton monomials iteratively
      single query point at a time is smaller than evaluationg all the Newton
      monomials on all query points before multiplying it with the coefficients.
    """

    n_points = xx.shape[0]

    # By default, all exponents are "active" unless xx are unisolvent nodes
    active_exponents = exponents
    # Iterate each query points and evaluate the Newton monomials
    for idx in range(n_points):

        x_single = xx[idx, :]

        # Get the row view of the monomials placeholder;
        # this would be the evaluation results of a single query point
        monomials_placeholder_single = monomials_placeholder[idx]

        if triangular:
            # TODO: Refactor this, this is triangular because
            #       the monomials are evaluated at the unisolvent nodes
            # When evaluated on unisolvent nodes,
            # some values will be a priori 0
            n_active_polys = idx + 1
            # Only some exponents are active
            active_exponents = exponents[:n_active_polys, :]
            # IMPORTANT: initialised empty. set all others to 0!
            monomials_placeholder_single[n_active_polys:] = 0.0
            # Only modify the non-zero entries
            monomials_placeholder_single = \
                monomials_placeholder_single[:n_active_polys]

        # Evaluate the Newton monomials on a single query point
        # NOTE: Due to "view" access, the whole 'monomials_placeholder' will
        # be modified
        eval_newton_monomials_single(
            x_single,
            active_exponents,
            generating_points,
            max_exponents,
            products_placeholder,
            monomials_placeholder_single
        )


@njit(void(F_2D, F_2D, I_2D), cache=True)
def compute_vandermonde_n2c(V_n2c, nodes, exponents):
    """Computes the Vandermonde matrix.

    - ``m`` spatial dimension
    - ``N`` number of monomials

    :param V_n2c: the placeholder array to store the Vandermonde matrix. The shape has to be ``(N x N)``.
    :param nodes: the unisolvent nodes
    :param exponents:  numpy array with exponents for the polynomial. The shape has to be ``(N x m)``.

    """
    num_monomials, spatial_dimension = exponents.shape
    for i in range(num_monomials):
        for j in range(1, num_monomials):
            for d in range(spatial_dimension):
                V_n2c[i, j] *= nodes[i, d] ** exponents[j, d]


@njit(b1(I_1D, I_1D), cache=True)
def lex_smaller_or_equal(index1: np.ndarray, index2: np.ndarray) -> bool:
    """Compares whether multi-index 1 is lexicographically smaller than or equal to multi-index 2.

    - ``m`` spatial dimension

    :param index1: a multi-index entry of shape ``m``.
    :param index2: another multi-index entry.
    :return: ``True`` if ``index1 <= index2`` (lexicographically), otherwise ``False``.

    """
    spatial_dimension = len(index1)
    for m in range(spatial_dimension - 1, -1, -1):  # from last to first dimension
        if index1[m] > index2[m]:
            return False
        if index1[m] < index2[m]:
            return True
    return True  # all equal


@njit(B_TYPE(I_2D), cache=True)
def have_lexicographical_ordering(indices: np.ndarray) -> bool:
    """Checks if an array of indices is ordered lexicographically

    - ``m`` spatial dimension
    - ``N`` number of monomials

    :param indices: array of multi-indices with shape ``(N x m)``.
    :return: ``True`` if indices are lexicographically ordered, ``False`` otherwise.

    """
    nr_exponents, spatial_dimension = indices.shape
    if nr_exponents <= 1:
        return True
    i1 = indices[0, :]
    for n in range(1, nr_exponents):
        i2 = indices[n, :]
        if lex_smaller_or_equal(i2, i1):
            return False
        if np.all(i1 == i2):  # duplicates are not allowed
            return False
        i1 = i2
    return True


@njit(INT(I_2D, I_1D), cache=True)
def get_match_idx(indices: np.ndarray, index: np.ndarray) -> int:
    """Finds the position of a multi index entry within an exponent matrix.

    - ``m`` spatial dimension
    - ``N`` number of monomials

    :param indices: array of multi indices with shape ``(N x m)``.
    :param index: one multi index entry with shape ``m``.
    :return: if ``index`` is present in ``indices``, the position (array index) where it is found is returned,
             otherwise a global constant ``NOT_FOUND`` is returned.

    Notes
    -----
    Exploits the lexicographical order of the indices to abort early -> not testing all indices.
    Time complexity: O(mN).

    """
    nr_exponents, spatial_dimension = indices.shape
    if nr_exponents == 0:
        return NOT_FOUND
    m = len(index)
    if m != spatial_dimension:
        raise ValueError("dimensions do not match.")
    out = NOT_FOUND
    for i in range(nr_exponents):  # O(N)
        contained_index = indices[i, :]
        if lex_smaller_or_equal(index, contained_index):  # O(m)
            # i is now pointing to the (smallest) index which is lexicographically smaller or equal
            # the two indices are equal iff the contained index is also smaller or equal than the query index
            # NOTE: testing for equality directly is faster, but only in the case of equality (<- rare!)
            #   most of the times the index won't be smaller and the check can be performed with fewer comparisons
            is_equal = lex_smaller_or_equal(contained_index, index)
            if is_equal:  # found the position of the index
                out = i
            break  # stop looking (an even bigger index cannot be equal)
    return out


@njit(B_TYPE(I_2D, I_1D), cache=True)
def index_is_contained(indices: np.ndarray, index: np.ndarray) -> bool:
    """Checks if a multi index entry is present in a set of indices.

    - ``m`` spatial dimension
    - ``N`` number of monomials

    :param indices: array of multi indices with shape ``(N x m)``.
    :param index: one multi index entry with shape ``m``.
    :return: ``True`` if the ``index`` is present in ``indices``, ``False`` otherwise.

    Notes
    -----
    Exploits the lexicographical order of the indices to abort early -> not testing all indices.

    """
    return get_match_idx(indices, index) != NOT_FOUND


@njit(B_TYPE(I_2D, I_2D), cache=True)
def all_indices_are_contained(subset_indices: np.ndarray, indices: np.ndarray) -> bool:
    """Checks if a set of indices is a subset of (or equal to) another set of indices.

    :param subset_indices: one set of multi indices.
    :param indices: another set of multi indices.
    :return: ``True`` if ``subset_indices`` is a subset or equal to ``indices``, ``False`` otherwise.

    Notes
    -----
    Exploits the lexicographical order of the indices to abort early -> not testing all indices.
    """
    nr_exp, dim = indices.shape
    nr_exp_subset, dim_subset = subset_indices.shape
    if nr_exp == 0 or nr_exp_subset == 0:
        raise ValueError("empty index set")
    if dim != dim_subset:
        raise ValueError("dimensions do not match.")
    if nr_exp < nr_exp_subset:
        return False

    # return True when all candidate indices are contained
    match_idx = -1
    for i in range(nr_exp_subset):
        candidate_index = subset_indices[i, :]
        indices2search = indices[match_idx + 1 :, :]  # start from the next one
        match_idx = get_match_idx(indices2search, candidate_index)
        if match_idx == NOT_FOUND:
            return False
    return True


@njit(void(I_2D, I_2D, I_1D), cache=True)
def fill_match_positions(larger_idx_set, smaller_idx_set, positions):
    """Finds matching positions (array indices) for multi index entries in two multi indices.

    :param larger_idx_set: the larger set of multi indices
    :param smaller_idx_set: the smaller set of multi indices
    :param positions: an array with positions (array indices) of multi index entries in `larger_idx_set`` that matches
                      each of the entry in ``smaller_idx_set``.

    """
    search_pos = -1
    nr_exp_smaller, spatial_dimension = smaller_idx_set.shape
    for i in range(nr_exp_smaller):
        idx1 = smaller_idx_set[i, :]
        while 1:
            search_pos += 1
            idx2 = larger_idx_set[search_pos, :]
            if lex_smaller_or_equal(idx1, idx2) and lex_smaller_or_equal(idx2, idx1):
                # NOTE: testing for equality directly is faster, but only in the case of equality (<- rare!)
                #   most of the times the index won't be smaller and the check can be performed with fewer comparisons
                positions[i] = search_pos
                break
