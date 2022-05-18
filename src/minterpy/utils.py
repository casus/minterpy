"""
set of package wide utility functions
"""
import itertools
from typing import Union

import numpy as np

from minterpy.core.verification import (
    check_dtype,
    convert_eval_output,
    rectify_eval_input,
    rectify_query_points,
)
from minterpy.global_settings import DEBUG, FLOAT_DTYPE, INT_DTYPE
from minterpy.jit_compiled_utils import eval_all_newt_polys, evaluate_multiple


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


def eval_newt_polys_on(
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

    result_placeholder = np.empty((nr_points, N), dtype=FLOAT_DTYPE)
    max_exponents = np.max(exponents, axis=0)
    prod_placeholder = np.empty((np.max(max_exponents) + 1, m), dtype=FLOAT_DTYPE)
    eval_all_newt_polys(
        x,
        exponents,
        generating_points,
        max_exponents,
        prod_placeholder,
        result_placeholder,
        triangular,
    )
    return result_placeholder


def newt_eval(
    x, coefficients, exponents, generating_points, verify_input: bool = False
):
    """Iterative implementation of polynomial evaluation in Newton form

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

    :param x: Arguemnt array with shape ``(m, k)`` the ``k`` points to evaluate on with dimensionality ``m``.
    :type x: np.ndarray
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
    verify_input = verify_input or DEBUG
    N, coefficients, m, nr_points, nr_polynomials, x = rectify_eval_input(
        x, coefficients, exponents, verify_input
    )
    # m_grid, nr_grid_values = generating_points.shape

    if verify_input:
        if generating_points.dtype != FLOAT_DTYPE:
            raise TypeError(
                f"grid values: expected dtype {FLOAT_DTYPE} got {generating_points.dtype}"
            )

    max_exponents = np.max(exponents, axis=0)
    # initialise arrays required for computing and storing the intermediary results:
    # will be reused in the different runs -> memory efficiency
    prod_placeholder = np.empty((np.max(max_exponents) + 1, m), dtype=FLOAT_DTYPE)
    monomial_vals_placeholder = np.empty(N, dtype=FLOAT_DTYPE)
    results_placeholder = np.empty((nr_points, nr_polynomials), dtype=FLOAT_DTYPE)
    evaluate_multiple(
        x,
        coefficients,
        exponents,
        generating_points,
        max_exponents,
        prod_placeholder,
        monomial_vals_placeholder,
        results_placeholder,
    )
    return convert_eval_output(results_placeholder)

def deriv_newt_eval(x: np.ndarray, coefficients: np.ndarray, exponents: np.ndarray,
                    generating_points: np.ndarray, derivative_order_along: np.ndarray) -> np.ndarray:
    """ Evaluate the derivative of a polynomial in the Newton form.

     m = spatial dimension
     n = polynomial degree
     N = number of coefficients
     k = number of points

    Parameters
    ----------
    x: (k, m) the k points to evaluate on with dimensionality m.
    coefficients: (N) the coefficients of the Newton polynomial.
    exponents: (m, N) a multi index "alpha" for every Newton polynomial
        corresponding to the exponents of this "monomial"
    generating_points: (m, n+1) grid values for every dimension (e.g. Leja ordered Chebychev values).
    derivative_order_along: (m) specifying the order along each dimension to compute the derivative
    eg. [2,3,1] will compute respectively 2nd order, 3rd order, and 1st order along spatial dimensions
    0, 1, and 2.

    Returns
    -------
    (k) the value of derivative of the polynomial evaluated at each point.

    Notes
    -----
    - Can compute derivative polynomials without transforming to canonical basis.
    - This derivative evaluation is done by taking derivatives of the Newton monomials.
    - JIT compilation using Numba was not used here as itertools.combinations() does not work with Numba.

    """

    N, m = exponents.shape
    nr_points = x.shape[0]
    max_exponents = np.max(exponents, axis=0)

    # Result of the derivative evaluation
    results = np.empty(nr_points, dtype=FLOAT_DTYPE)
    # Array to store individial basis monomial evaluations
    monomial_vals= np.empty(N, dtype=FLOAT_DTYPE)

    num_prods = np.max(max_exponents) + 1
    # Array to store products in basis monomial along each dimension
    products = np.empty((num_prods, m), dtype=FLOAT_DTYPE)

    # Newton monomials have to be evaluated at each input point separately
    for point_nr in range(nr_points):
        x_single = x[point_nr, :]

        # Constructing the products array
        for i in range(m):
            max_exp_in_dim = max_exponents[i]
            x_i = x_single[i]
            order = derivative_order_along[i]
            if order == 0: # no partial derivative along this dimension
                prod = 1.0
                for j in range(max_exp_in_dim):  # O(n)
                    p_ij = generating_points[j, i]
                    prod *= (x_i - p_ij)
                    # NOTE: shift index by one
                    exponent = j + 1  # NOTE: otherwise the result type is float
                    products[exponent, i] = prod
            else: # take partial derivative of 'order' along this dimension

                # derivative of first 'order' newt monomials will be 0 as their degree < order
                products[:order, i] = 0.0

                # if order of derivative larger than the degree
                if order >= num_prods:
                    continue

                # derivative of newt monomial 'order' will be just factorial of order
                fact = np.math.factorial(order)
                products[order, i] = fact

                # for all bigger monomials, use chain rule of differentiation to compute derivative of products
                for q in range(order + 1, max_exp_in_dim + 1):
                    combs = itertools.combinations(range(q), q-order)
                    res = 0.0
                    for comb in combs: # combs is a generator for combinations
                        prod = np.prod(x_i - generating_points[list(comb), i])
                        res += prod

                    res *= fact
                    products[q, i] = res

        # evaluate all Newton polynomials. O(Nm)
        for j in range(N):
            # the exponents of each monomial ("alpha")
            # are the indices of the products which need to be multiplied
            newt_mon_val = 1.0  # required as multiplicative identity
            for i in range(m):
                exp = exponents[j, i]
                # NOTE: an exponent of 0 should not cause a multiplication
                if exp > 0:
                    newt_mon_val *= products[exp, i]
                else:
                    order = derivative_order_along[i]
                    if order > 0:
                        newt_mon_val = 0.0
            monomial_vals[j] = newt_mon_val

        results[point_nr] = np.dot(coefficients, monomial_vals)

    return results
