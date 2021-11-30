"""
Module containing several numba optimized functions.
"""

import numpy as np
from numba import b1, njit, void

from minterpy.global_settings import (B_TYPE, F_1D, F_2D, F_3D, FLOAT, I_1D,
                                      I_2D, INT, NOT_FOUND)


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


@njit(FLOAT(F_1D, F_1D), cache=True)  # O(N)
def single_eval(coefficients, monomial_vals):
    """Evaluation of one polynomial at a single point given the coefficients and monomial evaluations.

    - ``N`` number of monomials

    :param coefficients: numpy array of polynomial coefficients. The shape has to be ``N``.
    :param monomial_vals: numpy array of evaluated monomial values at the point. The shape has to be ``N``.
    :return: the value of polynomial evaluated at the point

    Notes
    -----
    The value of a polynomial in Newton form is the sum over all coefficients multiplied with the value of the
    corresponding Newton basis polynomial.
    """

    assert len(coefficients) == len(monomial_vals)
    return np.sum(coefficients * monomial_vals)


@njit(void(F_1D, I_2D, F_2D, I_1D, F_2D, F_1D), cache=True)  # O(Nm)
def eval_newton_polynomials(
    x,
    exponents,
    generating_points,
    max_exponents,
    prod_placeholder,
    monomial_vals_placeholder,
):
    """Precomputes the value of all given Newton basis polynomials at a point.

    Core of the fast polynomial evaluation algorithm.
    - ``m`` spatial dimension
    - ``N`` number of monomials
    - ``n`` maximum exponent in each dimension

    :param x: coordinates of the point. The shape has to be ``m``.
    :param exponents: numpy array with exponents for the polynomial. The shape has to be ``(N x m)``.
    :param generating_points: generating points used to generate the grid. The shape is ``(n x m)``.
    :param max_exponents: array with maximum exponent in each dimension. The shape has to be ``m``.
    :param prod_placeholder: a numpy array for storing the (chained) products.
    :param monomial_vals_placeholder: a numpy array of length N for storing the values of all Newton basis polynomials.

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

    m = len(x)
    for i in range(m):
        max_exp_in_dim = max_exponents[i]
        x_i = x[i]
        prod = 1.0
        for j in range(max_exp_in_dim):  # O(n)
            # TODO there are n+1 1D grid values, the last one will never be used!?
            p_ij = generating_points[j, i]
            prod *= x_i - p_ij
            # NOTE: shift index by one
            exponent = j + 1  # NOTE: otherwise the result type is float
            prod_placeholder[exponent, i] = prod

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
                newt_mon_val *= prod_placeholder[exp, i]
        monomial_vals_placeholder[j] = newt_mon_val
    # NOTE: results have been stored in the numpy arrays. no need to return anything.


# @njit(void(F_2D, I_2D, F_2D, I_1D, F_2D, F_2D, B_TYPE), cache=True)
def eval_all_newt_polys(
    x,
    exponents,
    generating_points,
    max_exponents,
    prod_placeholder,
    matrix_placeholder,
    triangular=False,
):
    """Evaluates all Newton basis polynomials (monomials) on all given points.

    - ``m`` spatial dimension
    - ``k`` number of points
    - ``N`` number of monomials
    - ``p`` number of polynomials
    - ``n`` maximum exponent in each dimension

    :param x: numpy array with coordinates of points where polynomial is to be evaluated.
              The shape has to be ``(k x m)``.
    :param exponents: numpy array with exponents for the polynomial. The shape has to be ``(N x m)``.
    :param generating_points: generating points used to generate the grid. The shape is ``(n x m)``.
    :param max_exponents: array with maximum exponent in each dimension. The shape has to be ``m``.
    :param prod_placeholder: a numpy array for storing the (chained) products.
    :param matrix_placeholder: placeholder numpy array where the results of evaluation are stored.
                               The shape has to be ``(k x p)``.
    :param triangular: whether the output will be of lower triangular form or not.
                       -> will skip the evaluation of some values
    :return: the value of each Newton polynomial at each point. The shape will be ``(k x N)``.

    Notes
    -----
    ``exponents`` are allowed to be incomplete! Results are stored in the placeholder arrays. Nothing is returned.

    """
    nr_points, dimensionality = x.shape
    active_exponents = exponents  # all per default
    for point_nr in range(nr_points):  # evaluate on all given points points
        x_single = x[point_nr, :]
        monomial_vals_placeholder = matrix_placeholder[point_nr]  # row of the matrix
        if triangular:
            # only evaluate some polynomials to create a triangular output array
            nr_active_polys = point_nr + 1
            # IMPORTANT: initialised empty. set all others to 0!
            monomial_vals_placeholder[nr_active_polys:] = 0.0
            monomial_vals_placeholder = monomial_vals_placeholder[:nr_active_polys]
            active_exponents = exponents[:nr_active_polys, :]

        eval_newton_polynomials(
            x_single,
            active_exponents,
            generating_points,
            max_exponents,
            prod_placeholder,
            monomial_vals_placeholder,
        )


@njit(void(F_2D, F_2D, I_2D, F_2D, I_1D, F_2D, F_1D, F_2D), cache=True)
def evaluate_multiple(
    x,
    coefficients,
    exponents,
    generating_points,
    max_exponents,
    prod_placeholder,
    monomial_vals_placeholder,
    results_placeholder,
):
    """Newton polynomial evaluation on several points.

    - ``m`` spatial dimension
    - ``k`` number of points
    - ``N`` number of monomials
    - ``p`` number of polynomials
    - ``n`` maximum exponent in each dimension

    :param x: numpy array with coordinates of points where polynomial is to be evaluated.
              The shape has to be ``(k x m)``.
    :param exponents: numpy array with exponents for the polynomial. The shape has to be ``(N x m)``.
    :param generating_points: generating points used to generate the grid. The shape is ``(n x m)``.
    :param max_exponents: array with maximum exponent in each dimension. The shape has to be ``m``.
    :param prod_placeholder: a numpy array for storing the (chained) products.
    :param monomial_vals_placeholder: a numpy array of length N for storing the values of all Newton basis polynomials.
    :param results_placeholder: placeholder numpy array where the results of evaluation are stored.
                                The shape has to be ``(k x p)``.

    Notes
    -----
    Results are stored in the placeholder arrays. Nothing is returned.

    """
    nr_points = x.shape[0]
    nr_polynomials = coefficients.shape[1]
    for point_nr in range(nr_points):
        x_single = x[point_nr, :]
        # NOTE: with a fixed single point x to evaluate the polynomial on,
        # the values of the Newton polynomials become fixed (coefficient agnostic)
        # -> precompute all intermediary results (=compute the value of all Newton polynomials)
        eval_newton_polynomials(
            x_single,
            exponents,
            generating_points,
            max_exponents,
            prod_placeholder,
            monomial_vals_placeholder,
        )
        for poly_nr in range(nr_polynomials):
            coeffs_single = coefficients[:, poly_nr]
            results_placeholder[point_nr, poly_nr] = single_eval(
                coeffs_single, monomial_vals_placeholder
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


@njit(void(I_1D, I_2D, I_2D), cache=True)
def insert_single_index_numba(index2insert, indices, indices_out):
    """Insert a multi index entry to a set of indices.

    :param index2insert: the multi index entry to be inserted.
    :param indices: the set of multi indices given as input.
    :param indices_out: the resulting multi indices after (lexicographical) insertion.

    """
    i = 0
    bigger_entry_exists = False
    spatial_dimension, nr_exponents = indices.shape
    for i in range(nr_exponents):
        contained_index = indices[:, i]
        if lex_smaller_or_equal(index2insert, contained_index):
            bigger_entry_exists = True
            break
        indices_out[:, i] = contained_index  # insert the already contained index
    if bigger_entry_exists:
        # the multi-index should be inserted at the THIS position
        indices_out[:, i] = index2insert
        indices_out[:, i + 1 :] = indices[
            :, i:
        ]  # fill up the indices with the remaining indices
    else:  # no smaller entry exists, simply insert at the end
        indices_out[:, -1] = index2insert


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


@njit(FLOAT(I_1D, FLOAT), cache=True)
def lp_norm_for_exponents(exp_vect, p):
    """Robust lp-norm function. Works essentially like numpy.linalg.norm, but is numerically stable for big arguments.

    :param exp_vect: a multi index entry.
    :param p: a real number
    :return: the Lp norm of ``exp_vect``.

    """
    a = np.abs(exp_vect).max()
    if a == 0:  # NOTE: avoid division by 0
        return 0.0

    if p == 1.0:
        return np.sum(exp_vect)
    elif p == np.inf:
        return np.max(exp_vect)
    else:
        return a * np.linalg.norm(exp_vect / a, p)


@njit(void(F_3D, I_2D), cache=True)
def compute_grad_c2c(grad_c2c: np.ndarray, exponents: np.ndarray):
    """Computes the gradient operator from canonical basis to canonical basis.

    The operator (=tensor) transforming the coefficients of a polynomial
    into the coefficients of its gradient (in canonical basis)

    :param grad_c2c: the empty tensor which should hold the result
    :param exponents: matrix of all exponent vectors

    Notes
    -----
    - Complexity is ``O(m^2 N^2)``.
    - For the canonical case this tensor is sparse!
    - obtaining the gradient operator for different bases by matrix multiplications with the transformation matrices is inefficient.

    """
    nr_monomials, dimensionality = exponents.shape
    for coeff_idx_from in range(nr_monomials):  # O(N)
        monomial_exponents = exponents[coeff_idx_from, :]
        for dim_idx in range(dimensionality):  # derivation in every dimension, O(m)
            monomial_exponent_dim = monomial_exponents[dim_idx]
            if monomial_exponent_dim > 0:
                mon_exponents_derived = monomial_exponents.copy()
                mon_exponents_derived[dim_idx] -= 1
                # "gradient exponential mapping":
                # determine where each coefficient gets mapped to
                # -> the idx of the derivative monomial
                coeff_idx_to = get_match_idx(exponents, mon_exponents_derived)  # O(mN)
                # also multiply with exponent
                grad_c2c[dim_idx, coeff_idx_to, coeff_idx_from] = monomial_exponent_dim


@njit(void(F_3D, I_2D, F_2D), cache=True)
def compute_grad_x2c(grad_x2c: np.ndarray, exponents: np.ndarray, x2c: np.ndarray):
    """Computes the gradient operator from an origin basis to canonical basis.

    The operator (=tensor) transforming the coefficients of a polynomial
    into the coefficients of its gradient (from a variable basis into canonical basis)

    :param grad_x2c: the empty tensor which should hold the result
    :param exponents: matrix of all exponent vectors
    :param x2c: the transformation matrix from origin into canonical basis

    Notes
    -----
    Exploits the sparsity of the canonical gradient operator.

    """
    nr_monomials, dimensionality = exponents.shape
    for coeff_idx_from in range(nr_monomials):  # deriving every monomial, O(N)
        monomial_exponents = exponents[coeff_idx_from, :]
        for dim_idx in range(dimensionality):  # derivation in every dimension, O(m)
            mon_exp_in_dim = monomial_exponents[dim_idx]
            if mon_exp_in_dim > 0:  # use sparsity and avoid unnecessary operations
                mon_exponents_derived = monomial_exponents.copy()
                mon_exponents_derived[dim_idx] -= 1
                # "gradient exponential mapping":
                # determine where each coefficient gets mapped to
                # -> the idx of the derivative monomial
                coeff_idx_to = get_match_idx(exponents, mon_exponents_derived)  # O(mN)
                # also multiply with exponent (cf. with canonical case: get_canonical_gradient_operator() )
                # general case:
                # gradient_x = gradient_canonical @ x2c
                # matrix multiplication: C = A @ B
                # -> C[i,:] += A[i,j] * B[j,:] ("from j to i")
                # NOTE: addition required!
                grad_x2c[dim_idx, coeff_idx_to, :] += (
                    mon_exp_in_dim * x2c[coeff_idx_from, :]
                )
