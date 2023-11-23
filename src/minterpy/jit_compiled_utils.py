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
) -> None:
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
    - This is a Numba-accelerated function.
    - The function precompute all the (chained) products required during Newton evaluation for a single query point
      with complexity of ``O(mN)``.
    - The (pre-)computation of Newton monomials is coefficient agnostic.
    - Results are stored in the placeholder arrays. The function returns None.
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
) -> None:
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
    - This is a Numba-accelerated function.
    - The memory footprint for evaluating the Newton monomials iteratively
       with a single query point at a time is smaller than evaluating all
       the Newton monomials on all query points.
       However, when multiplied with multiple coefficient sets,
       this approach will be faster.
    - Results are stored in the placeholder arrays. The function returns None.
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
            # TODO: Refactor this, this is triangular because the monomials
            #       are evaluated at the unisolvent nodes, otherwise it won't
            #       be and the results would be misleading.
            # When evaluated on unisolvent nodes, some values will be a priori 0
            n_active_polys = idx + 1
            # Only some exponents are active
            active_exponents = exponents[:n_active_polys, :]
            # IMPORTANT: initialised empty. set all others to 0!
            monomials_placeholder_single[n_active_polys:] = 0.0
            # Only modify the non-zero entries
            monomials_placeholder_single = \
                monomials_placeholder_single[:n_active_polys]

        # Evaluate the Newton monomials on a single query point
        # NOTE: Due to "view" access,
        # the whole 'monomials_placeholder' will be modified
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
def is_lex_smaller_or_equal(index_1: np.ndarray, index_2: np.ndarray) -> bool:
    """Check if an index is lexicographically smaller than or equal to another.

    Parameters
    ----------
    index_1 : :class:`numpy:numpy.ndarray`
        A given multi-index, a one-dimensional array of length ``m``.
    index_2 : :class:`numpy:numpy.ndarray`
        Another multi-index, a one dimensional array of length ``m``.

    Returns
    -------
    bool
        Return `True` if ``index_1 <= index_2`` lexicographically,
        otherwise `False`.

    Notes
    -----
    - By default, Numba disables the bound-checking for performance reason.
      Therefore, if the two input arrays are of inconsistent shapes, no
      exception will be raised and the results cannot be trusted.

    Examples
    --------
    >>> my_index_1 = np.array([1, 2, 3])  # "Reference index"
    >>> my_index_2 = np.array([1, 2, 3])  # Equal
    >>> is_lex_smaller_or_equal(my_index_1, my_index_2)
    True
    >>> my_index_3 = np.array([2, 4, 5])  # Larger
    >>> is_lex_smaller_or_equal(my_index_1, my_index_3)
    True
    >>> my_index_4 = np.array([0, 3, 2])  # Smaller
    >>> is_lex_smaller_or_equal(my_index_1, my_index_4)
    False
    """
    spatial_dimension = len(index_1)
    # lexicographic: Iterate backward from the highest dimension
    for m in range(spatial_dimension - 1, -1, -1):
        if index_1[m] > index_2[m]:
            # index_1 is lexicographically larger
            return False

        if index_1[m] < index_2[m]:
            # index_1 is Lexicographically smaller
            return True

    # index_1 is lexicographically equal
    return True


@njit(B_TYPE(I_2D), cache=True)
def is_lex_sorted(indices: np.ndarray) -> bool:
    """Check if an array of multi-indices is lexicographically sorted.

    Parameters
    ----------
    indices : :class:`numpy:numpy.ndarray`
        Array of multi-indices, a two-dimensional non-negative integer array
        of shape ``(N, m)``, where ``N`` is the number of multi-indices
        and ``m`` is the number of spatial dimensions.

    Returns
    -------
    bool
        ``True`` if the multi-indices is lexicographically sorted, and
        ``False`` otherwise

    Notes
    -----
    - If there are any duplicate entries (between rows),
      an array of multi-indices does not have a lexicographical ordering.

    Examples
    --------
    >>> my_indices = np.array([[0, 2, 0]])  # single entry
    >>> is_lex_sorted(my_indices)
    True
    >>> my_indices = np.array([[0, 0], [1, 0], [0, 2]])  # already sorted
    >>> is_lex_sorted(my_indices)
    True
    >>> my_indices = np.array([[1, 0], [0, 0], [0, 2]])  # unsorted
    >>> is_lex_sorted(my_indices)
    False
    >>> my_indices = np.array([[0, 0], [2, 0], [2, 0]])  # duplicate entries
    >>> is_lex_sorted(my_indices)
    False
    """
    nr_indices = indices.shape[0]

    # --- Single entry is always lexicographically ordered
    if nr_indices <= 1:
        return True

    # --- Loop over the multi-indices and find any unsorted entry or duplicate
    index_1 = indices[0, :]
    for n in range(1, nr_indices):
        index_2 = indices[n, :]

        if is_lex_smaller_or_equal(index_2, index_1):
            # Unsorted entry or duplicates
            return False

        index_1 = index_2

    return True


@njit(INT(I_2D, I_1D), cache=True)
def search_lex_sorted(indices: np.ndarray, index: np.ndarray) -> int:
    """Find the position of a given entry within an array of multi-indices.

    Parameters
    ----------
    indices : :class:`numpy:numpy.ndarray`
        Array of lexicographically sorted multi-indices, a two-dimensional
        non-negative integer array of shape ``(N, m)``,
        where ``N`` is the number of multi-indices and ``m`` is the number
        of spatial dimensions.
    index : :class:`numpy:numpy.ndarray`
        Multi-index entry to check in ``indices``. The element is represented
        by a one-dimensional array of length ``m``, where ``m`` is the number
        of spatial dimensions.

    Returns
    -------
    int
        If ``index`` is present in ``indices``, its position in ``indices``
        is returned (the row number). Otherwise, a global constant
        ``NOT_FOUND`` is returned instead.

    Notes
    -----
    - ``indices`` must be lexicographically sorted.
    - This function is a binary search implementation that exploits
      a lexicographically sorted array of multi-indices.
      The time complexity of the implementation is :math:`O(m\log{N})`.
    - By Minterpy convention, duplicate entries are not allowed in
      a lexicographically sorted multi-indices. However, having duplicate
      entries won't stop the search. In that case, the search returns
      the position of the first match but cannot guarantee which one is that
      from the duplicates.

    Examples
    --------
    >>> my_indices = np.array([
    ... [0, 0, 0],  # 0
    ... [1, 0, 0],  # 1
    ... [2, 0, 0],  # 2
    ... [0, 0, 1],  # 3
    ... ])
    >>> my_index_1 = np.array([2, 0, 0])  # is present in my_indices
    >>> search_lex_sorted(my_indices, my_index_1)
    2
    >>> my_index_2 = np.array([0, 1, 0])  # is not present in my_indices
    >>> search_lex_sorted(my_indices, my_index_2)
    -1
    """
    nr_indices = indices.shape[0]
    if nr_indices == 0:
        # Zero-length multi-indices has no entry
        return NOT_FOUND

    # Initialize the search
    out = NOT_FOUND
    low = 0
    high = nr_indices - 1

    # Start the binary search
    while low <= high:

        mid = (high + low) // 2

        if is_lex_smaller_or_equal(indices[mid], index):
            # NOTE: Equality must be checked here because the function
            #       `is_lex_smaller_or_equal()` cannot check just for smaller.
            if is_lex_smaller_or_equal(index, indices[mid]):
                return mid

            low = mid + 1

        else:
            high = mid - 1

    return out


@njit(B_TYPE(I_2D, I_1D), cache=True)
def is_index_contained(indices: np.ndarray, index: np.ndarray) -> bool:
    """Check if a multi-index entry is present in a set of multi-indices.

    Parameters
    ----------
    indices : :class:`numpy:numpy.ndarray`
        Array of lexicographically sorted multi-indices, a two-dimensional
        non-negative integer array of shape ``(N, m)``,
        where ``N`` is the number of multi-indices and ``m`` is the number
        of spatial dimensions.
    index : :class:`numpy:numpy.ndarray`
        Multi-index entry to check in the set. The element is represented
        by a one-dimensional array of length ``m``,
        where ``m`` is the number of spatial dimensions.

    Returns
    -------
    bool
        ``True`` if the entry ``index`` is contained in the set ``indices``
        and ``False`` otherwise.

    Notes
    -----
    - The implementation is based on the binary search and therefore
      ``indices`` must be lexicographically sorted.

    Examples
    --------
    >>> my_indices = np.array([
    ... [0, 0, 0],  # 0
    ... [1, 0, 0],  # 1
    ... [2, 0, 0],  # 2
    ... [0, 0, 1],  # 3
    ... ])
    >>> is_index_contained(my_indices, np.array([1, 0, 0]))  # is present
    True
    >>> is_index_contained(my_indices, np.array([0, 1, 2]))  # is not present
    False
    """
    return search_lex_sorted(indices, index) != NOT_FOUND


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
        match_idx = search_lex_sorted(indices2search, candidate_index)
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
            if is_lex_smaller_or_equal(idx1, idx2) and is_lex_smaller_or_equal(idx2, idx1):
                # NOTE: testing for equality directly is faster, but only in the case of equality (<- rare!)
                #   most of the times the index won't be smaller and the check can be performed with fewer comparisons
                positions[i] = search_pos
                break


if __name__ == "__main__":
    import doctest
    doctest.testmod()
