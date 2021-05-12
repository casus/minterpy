"""
set of package wide utility functions
"""

import numpy as np

from minterpy.global_settings import DEBUG, FLOAT_DTYPE, INT_DTYPE
from minterpy.jit_compiled_utils import eval_all_newt_polys, evaluate_multiple
from minterpy.verification import (check_dtype, convert_eval_output,
                                   rectify_eval_input, rectify_query_points)


def lp_norm(arr, p, axis=None, keepdims: bool = False):
    """
    Robust lp-norm function. Works essentially like numpy.linalg.norm, but is numerically stable for big arguments.
    """

    a = np.abs(arr).max()
    if a == 0.0:  # NOTE: avoid division by 0
        return 0.0
    return a * np.linalg.norm(arr / a, p, axis, keepdims)


def chebychev_2nd_order(n: int):  # 2nd order
    if n == 0:
        return np.zeros(1, dtype=FLOAT_DTYPE)
    if n == 1:
        return np.array([-1.0, 1.0], dtype=FLOAT_DTYPE)
    return np.cos(np.arange(n, dtype=FLOAT_DTYPE) * np.pi / (n - 1))


def gen_chebychev_2nd_order_leja_ordered(n: int):
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


def report_error(errors, description=None):
    if description is not None:
        print("\n\n")
        print(description)

    print(f"mean: {np.mean(errors):.2e}")
    print(f"median: {np.median(errors):.2e}")
    print(f"variance: {np.var(errors):.2e}")
    print(f"l2-norm: {np.linalg.norm(errors):.2e}")
    # f"l_infty error (max): {np.linalg.norm(errors, ord=np.inf)}\n")
    errors = np.abs(errors)
    print(f"abs mean: {np.mean(errors):.2e}")
    print(f"abs median: {np.median(errors):.2e}")
    print(f"abs variance: {np.var(errors):.2e}")
    print(f"max abs {np.max(errors):.2e}")


def eval_newt_polys_on(
    x: np.ndarray,
    exponents: np.ndarray,
    generating_points: np.ndarray,
    verify_input: bool = False,
    triangular: bool = False,
) -> np.ndarray:
    """computes the value of each Newton polynomial on each given point

    N = amount of coefficients
    k = amount of points

    Parameters
    ----------
    x: the points to evaluate the polynomials on
    exponents: the multi indices "alpha" for every Newton polynomial
        corresponding to the exponents of this "monomial"
    generating_points:
    verify_input: weather the data types of the input should be checked. turned off by default for performance.
    triangular: weather or not the output will be of lower triangular form
        -> will skip the evaluation of some values

    Returns
    -------
    (k, N) the value of each Newton polynomial on each point
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
    """iterative implementation of polynomial evaluation in Newton form

    version able to handle both:
     - list of input points x (2D input)
     - list of input coefficients (2D input)

     NOTE: assuming equal input array shapes as the reference implementation

    TODO idea for improvement: make use of the sparsity of the exponent matrix
     and avoid iterating over the zero entries!

     n = polynomial degree
     N = amount of coefficients
     k = amount of points
     p = amount of polynomials


    faster than the recursive implementation of tree.eval_lp(...)
    for a single point and a single polynomial (1 set of coeffs):
    time complexity: O(mn+mN) = O(m(n+N)) = ...
        pre-computations: O(mn)
        evaluation: O(mN)
        NOTE: N >> n depending on l_p-degree

    space complexity: O(mn) (precomputing and storing the products)
        evaluation: O(0)

    advantage:
        - just operating on numpy arrays, can be just-in-time (jit) compiled
        - can evaluate multiple polynomials without recomputing all intermediary results


    Parameters
    ----------
    x: (m, k) the k points to evaluate on with dimensionality m.
    coefficients: (N, p) the coefficients of the Newton polynomials.
        NOTE: format fixed such that 'lagrange2newton' conversion matrices can be passed
        as the Newton coefficients of all Lagrange monomials of a polynomial without prior transponation
    exponents: (m, N) a multi index "alpha" for every Newton polynomial
        corresponding to the exponents of this "monomial"
    generating_points: (m, n+1) grid values for every dimension (e.g. Leja ordered Chebychev values).
        the values determining the locations of the hyperplanes of the interpolation grid.
        the ordering of the values determine the spacial distribution of interpolation nodes.
        (relevant for the approximation properties and the numerical stability).
    verify_input: weather the data types of the input should be checked. turned off by default for speed.


    Returns
    -------
    (k, p) the value of each input polynomial at each point. TODO squeezed into the expected shape (1D if possible)
    NOTE: format fixed such that the regression can use the result as transformation matrix without transponation
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
