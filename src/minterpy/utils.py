"""
set of package wide utility functions
"""
from typing import Union

import numpy as np

from minterpy.core.verification import (
    check_dtype,
    convert_eval_output,
    rectify_eval_input,
    rectify_query_points,
)
from minterpy.global_settings import DEBUG, FLOAT_DTYPE, INT_DTYPE
from minterpy.jit_compiled_utils import eval_newton_monomials_multiple


def lp_norm(arr, p, axis=None, keepdims: bool = False):
    """Robust lp-norm function.

    Works essentially like ``numpy.linalg.norm``, but is numerically stable for big arguments.

    :param arr: Input array.
    :type arr: np.ndarray

    :param axis: If axis is an integer, it specifies the axis of x along which to compute the vector norms. If axis is a 2-tuple, it specifies the axes that hold 2-D matrices, and the matrix norms of these matrices are computed. If axis is None then either a vector norm (when x is 1-D) or a matrix norm (when x is 2-D) is returned. The default is :class:`None`.
    :type axis: {None, int, 2-tuple of int}, optional

    :param keepdims: If this is set to True, the axes which are normed over are left in the result as dimensions with size one. With this option the result will broadcast correctly against the original ``arr``.
    :type keepdims: bool, optional
    """

    a = np.abs(arr).max()
    if a == 0.0:  # NOTE: avoid division by 0
        return 0.0
    return a * np.linalg.norm(arr / a, p, axis, keepdims)


def cartesian_product(*arrays: np.ndarray) -> np.ndarray:
    """
    Build the cartesian product of any number of 1D arrays.

    :param arrays: List of 1D array_like.
    :type arrays: list

    :return: Array of all combinations of elements of the input arrays (a cartesian product).
    :rtype: np.ndarray

    Examples
    --------
    >>> x = np.array([1,2,3])
    >>> y = np.array([4,5])
    >>> cartesian_product(x,y)
    array([[1, 4],
           [1, 5],
           [2, 4],
           [2, 5],
           [3, 4],
           [3, 5]])

    >>> s= np.array([0,1])
    >>> cartesian_product(s,s,s,s)
    array([[0, 0, 0, 0],
           [0, 0, 0, 1],
           [0, 0, 1, 0],
           [0, 0, 1, 1],
           [0, 1, 0, 0],
           [0, 1, 0, 1],
           [0, 1, 1, 0],
           [0, 1, 1, 1],
           [1, 0, 0, 0],
           [1, 0, 0, 1],
           [1, 0, 1, 0],
           [1, 0, 1, 1],
           [1, 1, 0, 0],
           [1, 1, 0, 1],
           [1, 1, 1, 0],
           [1, 1, 1, 1]])

    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def lp_sum(arr: np.ndarray, p: Union[float, int]) -> Union[float, int]:
    """
    Sum of powers, i.e. lp-norm to the lp-degree.

    :param arr: 2D-array to be lp-summed
    :type arr: np.ndarray

    :param p: Power for each element in the lp-sum
    :type p: Real

    :return: lp-sum over the last axis of the input array powered by the given power
    :rtype: np.ndarray

    Notes
    -----
    - equivalent to ```lp_norm(arr,p,axis=-1)**p``` but more stable then the implementation using `np.linalg.norm` (see `numpy #5697`_  for more informations)

    .. _numpy #5697:
        https://github.com/numpy/numpy/issues/5697
    """
    return np.sum(np.power(arr, p), axis=-1)


def chebychev_2nd_order(n: int):  # 2nd order
    """Factory function of Chebychev points of the second kind.

    :param n: Degree of the point set, i.e. number of Chebychev points.
    :type n: int

    :return: Array of Chebychev points of the second kind.
    :rtype: np.ndarray

    .. todo::
        - rename this function
        - rename the parameter ``n``.
    """
    if n == 0:
        return np.zeros(1, dtype=FLOAT_DTYPE)
    if n == 1:
        return np.array([-1.0, 1.0], dtype=FLOAT_DTYPE)
    return np.cos(np.arange(n, dtype=FLOAT_DTYPE) * np.pi / (n - 1))


def gen_chebychev_2nd_order_leja_ordered(n: int):
    """Factory function of Leja ordered Chebychev points of the second kind.

    :param n: Degree of the point set, i.e. number of Chebychev points (plus one!).
    :type n: int

    :return: Array of Leja ordered Chebychev points of the second kind.
    :rtype: np.ndarray

    .. todo::
        - rename this function
        - rename the parameter ``n``.
        - refactor this function to remove all the loops.
        - make the arguments equivalent to ``chebychev_2nd_order``, i.e. if the degree ``n`` is passed, the number of points shall be ``n`` (not ``n+1``).
    """
    n = int(n)
    points1 = chebychev_2nd_order(n + 1)[::-1]
    points2 = points1  # TODO
    ord = np.arange(1, n + 1)

    lj = np.zeros([1, n + 1])
    lj[0] = 0
    m = 0

    for k in range(0, n):
        jj = 0
        for i in range(0, n - k):
            P = 1
            for j in range(k + 1):
                idx_pts = int(lj[0, j])
                P = P * (points1[idx_pts] - points1[ord[i]])
            P = np.abs(P)
            if P >= m:
                jj = i
                m = P
        m = 0
        lj[0, k + 1] = ord[jj]
        ord = np.delete(ord, jj)

    leja_points = np.zeros([n + 1, 1])
    for i in range(n + 1):
        leja_points[i, 0] = points2[int(lj[0, i])]
    return leja_points


def eval_newton_monomials(
    x: np.ndarray,
    exponents: np.ndarray,
    generating_points: np.ndarray,
    verify_input: bool = False,
    triangular: bool = False,
) -> np.ndarray:
    """Newton evaluation function.

    Compute the value of each Newton monomial on each given point. Internally it uses ``numba`` accelerated evaluation function.

    :param x: The points to evaluate the polynomials on.
    :type x: np.ndarray
    :param exponents: the multi indices "alpha" for every Newton polynomial corresponding to the exponents of this "monomial"
    :type exponents: np.ndarray
    :param generating_points: Nodes where the Newton polynomial lives on. (Or the points which generate these nodes?!)
    :type generating_points: np.ndarray
    :param verify_input: weather the data types of the input should be checked. Turned off by default for performance.
    :type verify_input: bool
    :param triangular: weather or not the output will be of lower triangular form. This will skip the evaluation of some values. Defaults to :class:`False`.
    :type triangular: bool

    :return: the value of each Newton polynomial on each point. The output shape is ``(k, N)``, where ``k`` is the number of points and ``N`` is the number of coeffitions of the Newton polyomial.
    :rtype: np.ndarray

    .. todo::
        - rename ``generation_points`` according to :class:`Grid`.
        - use instances of :class:`MultiIndex` and/or :class:`Grid` instead of the array representations of them.
        - ship this to the submodule ``newton_polynomials``.
        - Refactor the "triangular" parameter, the Newton monomials
          only becomes triangular if evaluated at the unisolvent nodes.
          So it needs a special function instead of parametrizing this function
          that can give a misleading result.

    See Also
    --------
    eval_all_newt_polys : concrete ``numba`` accelerated implementation of polynomial evaluation in Newton base.
    """
    N, m = exponents.shape
    nr_points, x = rectify_query_points(
        x, m
    )  # also working for 1D array, reshape into required shape
    if verify_input:
        check_dtype(x, FLOAT_DTYPE)
        check_dtype(exponents, INT_DTYPE)

    # NOTE: the downstream numba-accelerated function does not support kwargs,
    # so the maximum exponent per dimension must be computed here
    max_exponents = np.max(exponents, axis=0)

    # Create placeholders for the final and intermediate results
    result_placeholder = np.empty((nr_points, N), dtype=FLOAT_DTYPE)
    prod_placeholder = np.empty(
        (np.max(max_exponents) + 1, m), dtype=FLOAT_DTYPE
    )

    # Compute the Newton monomials on all the query points
    eval_newton_monomials_multiple(
        x,
        exponents,
        generating_points,
        max_exponents,
        prod_placeholder,
        result_placeholder,
        triangular,
    )

    return result_placeholder


def eval_newton_polynomials(
    xx: np.ndarray,
    coefficients: np.ndarray,
    exponents: np.ndarray,
    generating_points: np.ndarray,
    verify_input: bool = False,
    batch_size: int = None,
):
    """Evaluate the polynomial(s) in Newton form at multiple query points.

    Iterative implementation of polynomial evaluation in Newton form

    This version able to handle both:
        - list of input points x (2D input)
        - list of input coefficients (2D input)

    Here we use the notations:
        - ``n`` = polynomial degree
        - ``N`` = amount of coefficients
        - ``k`` = amount of points
        - ``p`` = amount of polynomials


    .. todo::
        - idea for improvement: make use of the sparsity of the exponent matrix and avoid iterating over the zero entries!
        - refac the explanation and documentation of this function.
        - use instances of :class:`MultiIndex` and/or :class:`Grid` instead of the array representations of them.
        - ship this to the submodule ``newton_polynomials``.

    :param xx: Arguemnt array with shape ``(m, k)`` the ``k`` points to evaluate on with dimensionality ``m``.
    :type xx: np.ndarray
    :param coefficients: The coefficients of the Newton polynomials.
        NOTE: format fixed such that 'lagrange2newton' conversion matrices can be passed
        as the Newton coefficients of all Lagrange monomials of a polynomial without prior transponation
    :type coefficients: np.ndarray, shape = (N, p)
    :param exponents: a multi index ``alpha`` for every Newton polynomial corresponding to the exponents of this ``monomial``
    :type exponents: np.ndarray, shape = (m, N)
    :param generating_points: Grid values for every dimension (e.g. Leja ordered Chebychev values).
        the values determining the locations of the hyperplanes of the interpolation grid.
        the ordering of the values determine the spacial distribution of interpolation nodes.
        (relevant for the approximation properties and the numerical stability).
    :type generating_points: np.ndarray, shape = (m, n+1)
    :param verify_input: weather the data types of the input should be checked. turned off by default for speed.
    :type verify_input: bool, optional
    :param batch_size: batch size of query points
    :type batch_size: int, optional

    :raise TypeError: If the input ``generating_points`` do not have ``dtype = float``.

    :return: (k, p) the value of each input polynomial at each point. TODO squeezed into the expected shape (1D if possible). Notice, format fixed such that the regression can use the result as transformation matrix without transponation

    Notes
    -----
    - This method is faster than the recursive implementation of ``tree.eval_lp(...)`` for a single point and a single polynomial (1 set of coeffs):
        - time complexity: :math:`O(mn+mN) = O(m(n+N)) = ...`
        - pre-computations: :math:`O(mn)`
        - evaluation: :math:`O(mN)`
        - space complexity: :math:`O(mn)` (precomputing and storing the products)
        - evaluation: :math:`O(0)`
    - advantage:
        - just operating on numpy arrays, can be just-in-time (jit) compiled
        - can evaluate multiple polynomials without recomputing all intermediary results

    See Also
    --------
    evaluate_multiple : ``numba`` accelerated implementation which is called internally by this function.
    convert_eval_output: ``numba`` accelerated implementation of the output converter.
    """

    # Rectify the inputs
    # TODO: Refactor this rectify function
    verify_input = verify_input or DEBUG
    _, coefficients, _, n_points, _, xx = \
        rectify_eval_input(xx, coefficients, exponents, verify_input)

    # Get batch size
    # TODO: Verify the batch size
    if batch_size is None or batch_size >= n_points:
        newton_monomials = eval_newton_monomials(
            xx,
            exponents,
            generating_points,
            verify_input,
            False
        )
        results = newton_monomials @ coefficients
    else:
        # Evaluate the Newton polynomials in batches
        results = eval_newton_polynomials_batch(
            xx,
            coefficients,
            exponents,
            generating_points,
            batch_size
        )

    return convert_eval_output(results)


def eval_newton_polynomials_batch(
    xx: np.ndarray,
    coefficients: np.ndarray,
    exponents: np.ndarray,
    generating_points: np.ndarray,
    batch_size: int
):
    """Evaluate the polynomial in Newton form in batches of query points.

    Notes
    -----
    - It is assumed that the inputs have all been verified and rectified.
    - It would be more expensive to evaluate smaller batch sizes
      but with a less smaller memory footprint in any given iteration.
    - If memory does not permit whole evaluation of query points,
      consider using smaller but not the smallest batch size (i.e., not 1).
    """

    # Get some important numbers
    n_points = xx.shape[0]
    n_polynomials = coefficients.shape[1]

    # Create a placeholder for the results
    results_placeholder = np.empty(
        (n_points, n_polynomials), dtype=FLOAT_DTYPE
    )

    # Batch processing: evaluate the polynomials at a batch of query points
    n_batches = n_points // batch_size + 1
    for idx in range(n_batches):
        start_idx = idx * batch_size

        if idx == n_batches - 1:
            end_idx = idx * batch_size + n_points % batch_size
        else:
            end_idx = (idx + 1) * batch_size

        # Get the current batch of query points
        xx_batch = xx[start_idx:end_idx, :]

        # Compute the Newton monomials for the batch
        newton_monomials = eval_newton_monomials(
            xx_batch,
            exponents,
            generating_points,
            False,
            False
        )

        # Compute the polynomial values for the batch
        results_placeholder[start_idx:end_idx, :] = \
            newton_monomials @ coefficients

    return results_placeholder

def make_coeffs_2d(coefficients: np.ndarray) -> np.ndarray:
    """Make coefficients array 2d.

    Parameters
    ----------
    coefficients: np.ndarray with coefficients

    Returns
    -------
    Returns a 2d array in the case of both single and multiple polynomials

    Notes
    -----
    This function is similar to np.atleast_2d, but adds the extra dimension differently.
    """

    coeff_shape = coefficients.shape
    if len(coeff_shape) == 1:  # 1D: a single polynomial
        coefficients = np.expand_dims(coefficients,-1)  # reshape to 2D

    return coefficients
