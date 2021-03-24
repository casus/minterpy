"""
CanonicalPolynomial class
"""
from copy import deepcopy

import numpy as np

from minterpy.jit_compiled_utils import can_eval_mult
from minterpy.multivariate_polynomial_abstract import MultivariatePolynomialSingleABC

__all__ = ['CanonicalPolynomial']

from minterpy.global_settings import DEBUG, FLOAT_DTYPE

from minterpy.verification import verify_domain, rectify_eval_input, convert_eval_output


def dummy():
    raise NotImplementedError(f"This feature is not implemented yet.")


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
        Resulting exponents (essentially the union of mi1 and mi2).
    res_coeffs : array_like
        Resulting coefficients related to res_mi
        (essentially the sum of c1 and c2 where the exponents in mi1 and mi2 are the same and a concatenation of the rest.)

    Notes
    -----
    - there is no check if the shapes of the passed arrays match. This lies in the actual canonical_add function.
    """
    len1, dim1 = mi1.shape
    # print("len1,dim1",len1,dim1)
    len2, dim2 = mi2.shape  # assume m1 and m2 are same dimension -> expand_dim of the mi with smaller dim
    # print("len2,dim2",len2,dim2)

    mi1r = mi1.reshape(dim1, len1, 1)
    # print("mi1r",mi1r.shape)
    # print("mi1r",mi1r)
    mi2r = mi2.reshape(dim1, 1, len2)
    # print("mi2r",mi2r.shape)
    # print("mi2r",mi2r)

    add_cond = np.equal(mi1r, mi2r).all(axis=0, keepdims=False)  # where are mi1 and mi2 equal along the dim?
    # print("add_cond",add_cond)
    add_cond1 = add_cond.any(axis=-1)  # added condition for m1,c1
    # print("add_cond1",add_cond1)
    not_add_cond1 = np.logical_not(add_cond1)  # not added condition for m1,c1
    # print("not_add_cond1",not_add_cond1)
    add_cond2 = add_cond.any(axis=0)  # add condition for m2,c2
    # print("add_cond2",add_cond2)
    not_add_cond2 = np.logical_not(add_cond2)  # not added condition for m2,c2
    # print("not_add_cond2",not_add_cond2)

    # build resulting mi
    added_mi = mi1[:, add_cond1]  # shall be the same as mi2[add_cond2,:]
    # print("added_mi",added_mi)
    not_added_mi = np.concatenate((mi1[:, not_add_cond1], mi2[:, not_add_cond2]), axis=-1)  # collect the rest
    # print("not_added_mi",not_added_mi)
    res_mi = np.concatenate((added_mi, not_added_mi), axis=-1)  # summed mi shall be the first
    # print("res_mi",res_mi)

    # build resulting coeffs
    added_c = c1[add_cond1] + c2[add_cond2]
    not_added_c = np.concatenate((c1[not_add_cond1], c2[not_add_cond2]))
    res_c = np.concatenate((added_c, not_added_c))

    return res_mi, res_c


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

    dim1 = poly1.multi_index.spatial_dimension
    dim2 = poly2.multi_index.spatial_dimension
    if dim1 >= dim2:
        poly2.expand_dim(dim1)
    else:
        poly1.expand_dim(dim2)
    return poly1, poly2


def _matching_internal_domain(poly1, poly2, tol=None):
    """	Test if two polynomial have the same internal_domain.

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
    """
    p1, p2 = _match_dims(poly1, poly2)
    # print(p1.internal_domain,p2.internal_domain) # here is the error!!!!
    if _matching_internal_domain(p1, p2):
        res_mi, res_c = _generic_canonical_add(p1.multi_index.exponents, p1.coeffs, p2.multi_index.exponents, p2.coeffs)
        return CanonicalPolynomial(res_c, res_mi, internal_domain=p1.internal_domain, user_domain=p1.user_domain)
    else:
        raise NotImplementedError("Addition is not implemented for canonical polynomials with different domains")


def _canonical_sub(poly1, poly2):
    """
    Subtraction of two polynomials in canonical basis.
    """
    return _canonical_add(poly1, -poly2)


def canonical_add(self, other):
    """
    WARNING: unchanged old implementation in order to test the more comprehensive _canonical_add

    Addition of two polynomials in the canonical basis.
    """
    if self.multi_index == other.multi_index:
        res_coeffs = np.add(self.coeffs, other.coeffs)
    else:
        raise NotImplementedError(f"Addition of polynomials with different exponents not implemented")
    return CanonicalPolynomial(res_coeffs, self.multi_index, self.internal_domain, self.user_domain)


def canonical_sub(self, other):
    """
    WARNING: unchanged old implementation in order to test the more comprehensive _canonical_sum

    Subtraction of two polynomials in the canonical basis.
    """
    if self.multi_index == other.multi_index:
        res_coeffs = np.add(self.coeffs, -other.coeffs)
    else:
        raise NotImplementedError(f"Addition of polynomials with different exponents not implemented")
    return CanonicalPolynomial(res_coeffs, self.multi_index, self.internal_domain, self.user_domain)



def can_eval(x, coefficients, exponents, verify_input: bool = False):
    """ naive evaluation of the canonical polynomial

    version able to handle both:
     - list of input points x (2D input)
     - list of input coefficients (2D input)

    TODO Use Horner scheme for evaluation:
     https://github.com/MrMinimal64/multivar_horner
     package also supports naive evaluation, but not of multiple coefficient sets

    NOTE: assuming equal input array shapes as the Newton evaluation

    N = amount of coeffs
    k = amount of points
    p = amount of polynomials


    Parameters
    ----------
    x: (k, m) the k points to evaluate on with dimensionality m.
    coefficients: (N, p) the coeffs of each polynomial in canonical form (basis).
    exponents: (N, m) a multi index "alpha" corresponding to the exponents of each monomial
    verify_input: weather the data types of the input should be checked. turned off by default for speed.

    Returns
    -------
    (k, p) the value of each input polynomial at each point. TODO squeezed into the expected shape (1D if possible)
    """
    verify_input = verify_input or DEBUG
    N, coefficients, m, nr_points, nr_polynomials, x = rectify_eval_input(x, coefficients, exponents, verify_input)
    results_placeholder = np.zeros((nr_points, nr_polynomials), dtype=FLOAT_DTYPE)  # IMPORTANT: initialise to 0!
    can_eval_mult(x, coefficients, exponents, results_placeholder)
    return convert_eval_output(results_placeholder)


def canonical_eval(canonical_poly, x: np.ndarray):
    coefficients = canonical_poly.coeffs
    exponents = canonical_poly.multi_index.exponents
    return can_eval(x, coefficients, exponents)


# TODO redundant
canonical_generate_internal_domain = verify_domain
canonical_generate_user_domain = verify_domain


class CanonicalPolynomial(MultivariatePolynomialSingleABC):
    """
    Canonical polynomial class.
    """
    # Virtual Functions
    _add = staticmethod(_canonical_add)
    _sub = staticmethod(_canonical_sub)
    _mul = staticmethod(dummy)
    _div = staticmethod(dummy)
    _pow = staticmethod(dummy)
    _eval = canonical_eval

    generate_internal_domain = staticmethod(canonical_generate_internal_domain)
    generate_user_domain = staticmethod(canonical_generate_user_domain)

    def expand_dim(self, dim, extra_internal_domain=None, extra_user_domain=None):
        """
        Expands the dimension of the polynomial by adding zeros to the multi_indices
        (which is equivalent to the multiplication of ones to each monomial)

        TODO maybe move this function to MultivariatePolynomialSingleABC, but be aware of the grid points.
        """
        expand_dim = dim - self.multi_index.spatial_dimension

        self.multi_index.expand_dim(dim)  # breaks if dim<spacial_dimension, i.e. expand_dim<0
        extra_internal_domain = verify_domain(extra_internal_domain, expand_dim)
        self.internal_domain = np.concatenate((self.internal_domain, extra_internal_domain))
        extra_user_domain = verify_domain(extra_user_domain, expand_dim)
        self.user_domain = np.concatenate((self.user_domain, extra_user_domain))
