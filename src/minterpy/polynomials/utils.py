"""
set of utility functions to be used in polynomials submodule
"""
import itertools
import numpy as np

from scipy.special import roots_legendre
from typing import Callable

from minterpy.core.tree import MultiIndexTree
from minterpy.dds import dds
from minterpy.global_settings import FLOAT_DTYPE
from minterpy.utils import rectify_eval_input, eval_newton_monomials


def deriv_newt_eval(x: np.ndarray, coefficients: np.ndarray, exponents: np.ndarray,
                    generating_points: np.ndarray, derivative_order_along: np.ndarray) -> np.ndarray:
    """Evaluate the derivative of a polynomial in the Newton form.

     m = spatial dimension
     n = polynomial degree
     N = number of coefficients
     p = number of polynomials
     k = number of evaluation points

    Parameters
    ----------
    x: (k, m) the k points to evaluate on with dimensionality m.
    coefficients: (N, p) the coefficients of the Newton polynomial(s).
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

    N, coefficients, m, nr_points, nr_polynomials, x = \
        rectify_eval_input(x, coefficients, exponents, False)

    max_exponents = np.max(exponents, axis=0)

    # Result of the derivative evaluation
    results = np.empty((nr_points, nr_polynomials), dtype=FLOAT_DTYPE)

    # Array to store individual basis monomial evaluations
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

        results[point_nr] = np.sum(monomial_vals[:,None] * coefficients, axis=0)

    return results


def integrate_monomials_newton(
    exponents: np.ndarray, generating_points: np.ndarray, bounds: np.ndarray
) -> np.ndarray:
    """Integrate the Newton monomials given a set of exponents.

    Parameters
    ----------
    exponents : :class:`numpy:numpy.ndarray`
        A set of exponents from a multi-index set that defines the polynomial,
        an ``(N, M)`` array, where ``N`` is the number of exponents
        (multi-indices) and ``M`` is the number of spatial dimensions.
        The number of exponents corresponds to the number of monomials.
    generating_points : :class:`numpy:numpy.ndarray`
        A set of generating points of the interpolating polynomial,
        a ``(P + 1, M)`` array, where ``P`` is the maximum degree of
        the polynomial in any dimensions and ``M`` is the number
        of spatial dimensions.
    bounds : :class:`numpy:numpy.ndarray`
        The bounds (lower and upper) of the definite integration, an ``(M, 2)``
        array, where ``M`` is the number of spatial dimensions.

    Returns
    -------
    np.ndarray
        The integrated Newton monomials, an ``(N,)`` array, where N is
        the number of monomials (exponents).
    """
    # --- Get some basic data
    num_monomials, num_dim = exponents.shape
    max_exp = np.max(exponents)
    max_exps_in_dim = np.max(exponents, axis=0)

    # --- Compute the integrals of one-dimensional bases
    one_dim_integrals = np.empty((max_exp + 1, num_dim))  # A lookup table
    for j in range(num_dim):
        max_exp_in_dim = max_exps_in_dim[j]
        exponents_1d = np.arange(max_exp_in_dim + 1)[:, np.newaxis]
        generating_points_in_dim = generating_points[:, j][:, np.newaxis]
        # NOTE: Newton monomials are polynomials, Gauss-Legendre quadrature
        #       will be exact for degree == 2*num_points - 1.
        quad_num_points = np.ceil((max_exp_in_dim + 1) / 2)

        # Compute the integrals
        one_dim_integrals[: max_exp_in_dim + 1, j] = _gauss_leg_quad(
            lambda x: eval_newton_monomials(
                x, exponents_1d, generating_points_in_dim
            ),
            num_points=quad_num_points,
            bounds=bounds[j],
        )

    # --- Compute integrals of the monomials (multi-dimensional basis)
    integrated_monomials = np.zeros(num_monomials)
    for i in range(num_monomials):
        out = 1.0
        for j in range(num_dim):
            exp = exponents[i, j]
            out *= one_dim_integrals[exp, j]
        integrated_monomials[i] = out

    return integrated_monomials


def integrate_monomials_canonical(
    exponents: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:
    """Integrate the monomials in the canonical basis.

    Parameters
    ----------
    exponents : :class:`numpy:numpy.ndarray`
        A set of exponents from a multi-index set that defines the polynomial,
        an ``(N, M)`` array, where ``N`` is the number of exponents
        (multi-indices) and ``M`` is the number of spatial dimensions.
        The number of exponents corresponds to the number of monomials.
    bounds : :class:`numpy:numpy.ndarray`
        The bounds (lower and upper) of the definite integration, an ``(M, 2)``
        array, where ``M`` is the number of spatial dimensions.

    Returns
    -------
    np.ndarray
        The integrated Canonical monomials, an ``(N,)`` array, where ``N`` is
        the number of monomials (exponents).
    """
    bounds_diff = np.diff(bounds)

    if np.allclose(bounds_diff, 2):
        # NOTE: Over the whole canonical domain [-1, 1]^M, no need to compute
        #       the odd-degree terms.
        case = np.all(np.mod(exponents, 2) == 0, axis=1)  # All even + 0
        even_terms = exponents[case]

        integrated_even_terms = bounds_diff.T / (even_terms + 1)

        integrated_monomials = np.zeros(exponents.shape)
        integrated_monomials[case] = integrated_even_terms

        return integrated_monomials.prod(axis=1)

    # NOTE: Bump the exponent by 1 (polynomial integration)
    bounds_power = np.power(bounds.T[:, None, :], (exponents + 1)[None, :, :])
    bounds_diff = bounds_power[1, :] - bounds_power[0, :]

    integrated_monomials = bounds_diff / (exponents + 1)

    return integrated_monomials.prod(axis=1)


def integrate_monomials_lagrange(
    exponents: np.ndarray,
    generating_points: np.ndarray,
    tree: MultiIndexTree,
    bounds: np.ndarray,
) -> np.ndarray:
    """Integrate the Lagrange monomials given a set of exponents.

    Parameters
    ----------
    exponents : :class:`numpy:numpy.ndarray`
        A set of exponents from a multi-index set that defines the polynomial,
        an ``(N, M)`` array, where ``N`` is the number of exponents
        (multi-indices) and ``M`` is the number of spatial dimensions.
        The number of exponents corresponds to the number of monomials.
    generating_points : :class:`numpy:numpy.ndarray`
        A set of generating points of the interpolating polynomial,
        a ``(P + 1, M)`` array, where ``P`` is the maximum degree of
        the polynomial in any dimensions and ``M`` is the number
        of spatial dimensions.
    tree : MultiIndexTree
        The MultiIndexTree to perform multi-dimensional divided-difference
        scheme (DDS) for transforming the Newton basis to Lagrange basis.
    bounds : :class:`numpy:numpy.ndarray`
        The bounds (lower and upper) of the definite integration, an ``(M, 2)``
        array, where ``M`` is the number of spatial dimensions.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The integrated Lagrange monomials, an ``(N,)`` array, where ``N``
        is the number of monomials (exponents).

    Notes
    -----
    - The Lagrange monomials are represented in the Newton basis.
      For integration, first integrate the Newton monomials and then transform
      the results back to the Lagrange basis. This is why the `MultiIndexTree`
      instance is needed.
    """
    integrated_newton_monomials = integrate_monomials_newton(
        exponents, generating_points, bounds
    )
    l2n = dds(np.eye(exponents.shape[0]), tree)

    return l2n.T @ integrated_newton_monomials


def _gauss_leg_quad(
    fun: Callable, num_points: int, bounds: np.ndarray
) -> np.ndarray:
    """Integrate a one-dimensional function using Gauss-Legendre quadrature.

    Parameters
    ----------
    fun : Callable
        The function to integrate, the output may be a vector.
    num_points : int
        The number of points used in the quadrature scheme.
    bounds : :class:`numpy:numpy.ndarray`
        The bounds of integration, an 1-by-2 array (lower and upper bounds).

    Returns
    -------
    np.ndarray
        The integral of the function over the given bounds.
    """
    quad_nodes, quad_weights = roots_legendre(num_points)

    bound_diff = np.diff(bounds)
    bound_sum = np.sum(bounds)

    fun_vals = fun(bound_diff / 2.0 * quad_nodes + bound_sum / 2)

    integrals = bound_diff / 2 * quad_weights @ fun_vals

    return integrals
