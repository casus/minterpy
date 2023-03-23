"""
set of utility functions to be used in polynomials submodule
"""
import itertools
import numpy as np
from minterpy.global_settings import FLOAT_DTYPE
from minterpy.utils import rectify_eval_input

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
