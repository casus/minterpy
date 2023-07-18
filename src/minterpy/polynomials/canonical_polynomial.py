"""
Base class for polynomials in the canonical base.

"""
import numpy as np

from copy import deepcopy
from scipy.special import factorial

from minterpy.core import MultiIndexSet
from minterpy.core.ABC import MultivariatePolynomialSingleABC
from minterpy.core.utils import find_match_between
from minterpy.core.verification import convert_eval_output, rectify_eval_input, verify_domain
from minterpy.global_settings import DEBUG, FLOAT_DTYPE, INT_DTYPE
from minterpy.jit_compiled_utils import can_eval_mult, all_indices_are_contained
from minterpy.polynomials.utils import integrate_monomials_canonical
from minterpy.utils import make_coeffs_2d


__all__ = ["CanonicalPolynomial"]


def dummy():
    """Placeholder function.

    .. warning::
      This function is not implemented yet!
    """
    raise NotImplementedError("This feature is not implemented yet.")


# Arithmetics


def _generic_canonical_add(mi1, c1, mi2, c2):
    """
    Generic add function based on exponents and coeffs (passed as arrays).

    Parameters
    ----------
    mi1 : array_like (shape = (mi1_len,dim))
        exponents of the first multi_index set.
    c1 : array_like (shape = (mi1_len,))
        coeffs related to mi1
    mi2 : array_like (shape = (mi2_len,dim))
        exponents of the second multi_index set.
    c2 : array_like (shape = (mi2_len,))
        coeffs related to mi2

    Returns
    -------
    res_mi : array_like
        Resulting exponents (essentially the union of mi1 and mi2) in lexicographical order.
    res_coeffs : array_like
        Resulting coefficients related to res_mi
        (essentially the sum of c1 and c2 where the exponents in mi1 and mi2 are the same and a concatenation of the rest.)

    Notes
    -----
    - there is no check if the shapes of the passed arrays match. This lies in the actual canonical_add function.
    """

    len1, dim1 = mi1.shape
    (
        len2,
        dim2,
    ) = (
        mi2.shape
    )  # assume m1 and m2 are same dimension -> expand_dim of the mi with smaller dim

    mi1r = mi1.reshape(len1, 1, dim1)
    mi2r = mi2.reshape(1, len2, dim1)

    add_cond = np.equal(mi1r, mi2r).all(
        axis=2, keepdims=False
    )  # where are mi1 and mi2 equal along the dim?
    add_cond1 = add_cond.any(axis=1)  # added condition for m1,c1
    not_add_cond1 = np.logical_not(add_cond1)  # not added condition for m1,c1
    add_cond2 = add_cond.any(axis=0)  # add condition for m2,c2
    not_add_cond2 = np.logical_not(add_cond2)  # not added condition for m2,c2

    # build resulting mi
    added_mi = mi1[add_cond1, :]  # shall be the same as mi2[add_cond2,:]
    not_added_mi = np.concatenate(
        (mi1[not_add_cond1, :], mi2[not_add_cond2, :]), axis=0
    )  # collect the rest
    res_mi = np.concatenate(
        (added_mi, not_added_mi), axis=0
    )  # summed mi shall be the first
    sort_args = np.lexsort(res_mi.T, axis=-1)

    # build resulting coeffs
    added_c = c1[add_cond1] + c2[add_cond2]
    not_added_c = np.concatenate((c1[not_add_cond1], c2[not_add_cond2]))
    res_c = np.concatenate((added_c, not_added_c))

    return res_mi[sort_args], res_c[sort_args]


def _match_dims(poly1, poly2, copy=None):
    """Dimensional expansion of two polynomial in order to match their spatial_dimensions.

    Parameters
    ----------
    poly1 : CanonicalPolynomial
        First polynomial in canonical basis
    poly2 : CanonicalPolynomial
        Second polynomial in canonical basis
    copy : bool
        If True, work on deepcopies of the passed polynomials (doesn't change the input).
        If False, inplace expansion of the passed polynomials

    Returns
    -------
    (poly1,poly2) : (CanonicalPolynomial,CanonicalPolynomial)
        Dimensionally matched polynomials in the same order as input.

    Notes
    -----
    - Maybe move this to the MultivariatePolynomialSingleABC since it shall be avialable for all poly bases
    """
    if copy is None:
        copy = True

    if copy:
        p1 = deepcopy(poly1)
        p2 = deepcopy(poly2)
    else:
        p1 = poly1
        p2 = poly2

    dim1 = p1.multi_index.spatial_dimension
    dim2 = p2.multi_index.spatial_dimension
    if dim1 >= dim2:
        p2.expand_dim(dim1)
    else:
        p1.expand_dim(dim2)
    return p1, p2


def _matching_internal_domain(poly1, poly2, tol=None):
    """Test if two polynomial have the same internal_domain.

    Works on polynomial with same spatial_dimension.

    Parameters
    ----------
    poly1, poly2 : CanonicalPolynomial
            Polynomials to check
    tol : float
            tolerance for matching floats in the domains. Default tol = 1e-16

    Returns
    -------
    match : bool
            True if both
    """
    if tol is None:
        tol = 1e-16
    return (
        np.less_equal(np.abs(poly1.internal_domain - poly2.internal_domain), tol)
        & np.less_equal(np.abs(poly1.user_domain - poly2.user_domain), tol)
    ).all()


def _canonical_add(poly1, poly2):
    """
    Addition of two polynomials in canonical basis.

    Parameters
    ----------
    poly1 : CanonicalPolynomial
        First summand of the sum
    poly2 : CanonicalPolynomial
        Second summand of the sum

    Returns
    -------
    summed_polynomial : CanonicalPolynomial
        The sum of the passed polynomials in canonical base

    Notes
    -----
     - This works only for the same domains!
     - This function will be called on `self,other` if `__add__` is called.
    """
    p1, p2 = _match_dims(poly1, poly2)
    # print(p1.internal_domain,p2.internal_domain) # here is the error!!!!
    if _matching_internal_domain(p1, p2):
        res_mi_arr, res_c = _generic_canonical_add(
            p1.multi_index.exponents, p1.coeffs, p2.multi_index.exponents, p2.coeffs
        )

        # warning: this is not general. we are *assuming* that the resultant lp_degree is
        # max of the two. This is done in order to avoid issue #37.
        max_lp_degree = np.max([p1.multi_index.lp_degree, p2.multi_index.lp_degree])
        res_mi = MultiIndexSet(res_mi_arr, lp_degree=max_lp_degree)
        return CanonicalPolynomial(
            res_mi,
            res_c,
            internal_domain=p1.internal_domain,
            user_domain=p1.user_domain,
        )
    else:
        raise NotImplementedError(
            "Addition is not implemented for canonical polynomials with different domains"
        )


def _canonical_sub(poly1, poly2):
    """
    Substraction of two polynomials in canonical basis.

    Parameters
    ----------
    poly1 : CanonicalPolynomial
        Minuend of the difference.
    poly2 : CanonicalPolynomial
        Subtrahend of the difference.

    Returns
    -------
    difference_polynomial : CanonicalPolynomial
        The difference of the passed polynomials in canonical base.

    Notes
    -----
    This works only for the same domains!
    """
    return _canonical_add(poly1, -poly2)


def _canonical_eval(pts: np.ndarray,exponents: np.ndarray, coeffs: np.ndarray):
    """
    Unsafe version of naive canonical evaluation

    :param pts: List of points, the polynomial must be evaluated on. Assumed shape: `(number_of_points,spatial_dimension)`.
    :type pts: np.ndarray

    :param exponents: Exponents from a multi-index set. Assumed shape:` (number_of_monomials, spatial_dimension)`.
    :type exponents: np.ndarray

    :param coeffs: List of coefficients. Assumed shape: `(number_of_monomials,)`
    :type coeffs: np.ndarray

    :return: result of the canonical evaluation.
    :rtype: np.ndarray
    
    """
    yy = np.dot(np.prod(np.power(pts[:, None, :], exponents[None, :, :]), axis=-1), coeffs)

    return convert_eval_output(yy)


def _verify_eval_input(pts, spatial_dimension):
    """
    verification of the input of the canonical evaluation. 
    """
    assert(isinstance(pts,np.ndarray))
    assert(pts.ndim==2)
    assert(pts.shape[-1]==spatial_dimension)


def canonical_eval(canonical_poly, pts: np.ndarray):
    """
    Navie canonical evaluation

    :param canonical_poly: Polynomial in canonical form to be evaluated.
    :type canonical_poly: CanonicalPolynomial

    :param pts: List of points, the polynomial must be evaluated on. Assumed shape: `(number_of_points,spatial_dimension)`.
    :type pts: np.ndarray

    :return: result of the canonical evaluation. 
    :rtype: np.ndarray

    """
    _verify_eval_input(pts,canonical_poly.spatial_dimension)
    return _canonical_eval(pts,canonical_poly.multi_index.exponents,canonical_poly.coeffs)


# TODO redundant
canonical_generate_internal_domain = verify_domain
canonical_generate_user_domain = verify_domain


def _canonical_partial_diff(poly: "CanonicalPolynomial", dim: int, order: int) -> "CanonicalPolynomial":
    """ Partial differentiation in Canonical basis.
    """
    spatial_dim = poly.multi_index.spatial_dimension
    deriv_order_along = np.zeros(spatial_dim, dtype=INT_DTYPE)
    deriv_order_along[dim] = order
    return _canonical_diff(poly, deriv_order_along)


def _canonical_diff(poly: "CanonicalPolynomial", order: np.ndarray) -> "CanonicalPolynomial":
    """ Partial differentiation in Canonical basis.
    """

    coeffs = make_coeffs_2d(poly.coeffs)
    exponents = poly.multi_index.exponents

    # Guard rails in ABC ensures that the len(order) == poly.spatial_dimension
    subtracted_exponents = exponents - order

    # compute mask for non-negative multi index entries
    diff_exp_mask = np.all(exponents >= order, axis = 1)

    # multi index entries in the differentiated polynomial
    diff_exponents = subtracted_exponents[diff_exp_mask]

    # Checking if the necessary multi index entries are present
    # Zero size check is needed here as all_indices_are_contained throws error otherwise
    if diff_exponents.size != 0 and not all_indices_are_contained(diff_exponents, exponents):
        raise ValueError(f"Cannot differentiate as some of the required multi indices are not present.")

    # coefficients of the differentiated polynomial
    diff_coeffs = coeffs[diff_exp_mask] * np.prod(factorial(exponents[diff_exp_mask]) / factorial(diff_exponents),
                                                  axis=1)[:,None]

    # The differentiated polynomial being expressed wrt multi indices of the given poly
    # NOTE: 'find_match_between' assumes 'exponents' is lexicographically ordered
    map_pos = find_match_between(diff_exponents, exponents)
    new_coeffs = np.zeros_like(coeffs)
    new_coeffs[map_pos] = diff_coeffs

    # Squeezing the last dimension to handle single polynomial
    return CanonicalPolynomial.from_poly(poly, new_coeffs.reshape(poly.coeffs.shape))


def _canonical_integrate_over(
    poly: "CanonicalPolynomial",
    bounds: np.ndarray,
) -> np.ndarray:
    """Compute the definite integral of a polynomial in the canonical basis.

    Parameters
    ----------
    poly : CanonicalPolynomial
        The polynomial of which the integration is carried out.
    bounds : :class:`numpy:numpy.ndarray`
        The bounds (lower and upper) of the definite integration, an ``(M, 2)``
        array, where ``M`` is the number of spatial dimensions.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The integral value of the polynomial over the given domain.
    """
    # --- Compute the integral of the canonical monomials (quadrature weights)
    exponents = poly.multi_index.exponents

    quad_weights = integrate_monomials_canonical(exponents, bounds)

    return quad_weights @ poly.coeffs


class CanonicalPolynomial(MultivariatePolynomialSingleABC):
    """
    Polynomial type in the canonical base.
    """

    __doc__ = """
    This is the docstring of the canonical base class.
    """

    # __doc__ += MultivariatePolynomialSingleABC.__doc_attrs__
    # Virtual Functions
    _add = staticmethod(_canonical_add)
    _sub = staticmethod(_canonical_sub)
    _mul = staticmethod(dummy)
    _div = staticmethod(dummy)
    _pow = staticmethod(dummy)
    _eval = canonical_eval

    _partial_diff = staticmethod(_canonical_partial_diff)
    _diff = staticmethod(_canonical_diff)

    _integrate_over = staticmethod(_canonical_integrate_over)

    generate_internal_domain = staticmethod(canonical_generate_internal_domain)
    generate_user_domain = staticmethod(canonical_generate_user_domain)
