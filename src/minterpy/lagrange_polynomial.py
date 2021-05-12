"""
LagrangePolynomial class
"""
from typing import Optional

import numpy as np

import minterpy
from minterpy.canonical_polynomial import (_match_dims,
                                           _matching_internal_domain)
from minterpy.global_settings import ARRAY
from minterpy.multi_index_utils import insert_lexicographically
from minterpy.multivariate_polynomial_abstract import \
    MultivariatePolynomialSingleABC
from minterpy.utils import newt_eval
from minterpy.verification import verify_domain

__all__ = ["LagrangePolynomial"]


def dummy():
    raise NotImplementedError(f"This feature is not implemented yet.")


# TODO: part of polynomial utils?
# TODO: optimize with numba
def _union_of_exponents(exp1, exp2):
    """
    Returns the union of two multiindices along with a mapping of indices from the
    resultant set to the exp1 and exp2
    """
    res_exp = insert_lexicographically(exp1.copy(), exp2.copy())
    nr_monomials, _ = res_exp.shape

    res_map = np.zeros((nr_monomials, 2)).astype(np.int)

    # Assuming all exponents are lexicographically sorted
    pos_exp1 = 0
    pos_exp2 = 0
    for i in range(nr_monomials):
        if np.array_equal(res_exp[i, :], exp1[pos_exp1, :]):
            res_map[i, 0] = pos_exp1
            pos_exp1 += 1
        if np.array_equal(res_exp[i, :], exp2[pos_exp2, :]):
            res_map[i, 1] = pos_exp2
            pos_exp2 += 1

    return res_exp, res_map


# TODO : Generalize to handle multiple polynomials (2D coeffs)
def _generic_lagrange_add(exp1, coeff1, exp2, coeff2):
    if len(coeff2) > len(coeff1):
        return _generic_lagrange_add(exp2, coeff2, exp1, coeff1)

    res_exp, res_map = _union_of_exponents(exp1, exp2)
    nr_monomials, _ = res_exp.shape
    res_coeff = np.zeros(nr_monomials)
    for i in range(nr_monomials):
        if res_map[i, 0] != -1:
            res_coeff[i] += coeff1[res_map[i, 0]]
        if res_map[i, 1] != -1:
            res_coeff[i] += coeff2[res_map[i, 1]]

    return res_exp, res_coeff


# TODO : poly2 can be of a different basis?
def _lagrange_add(poly1, poly2):
    """Addition of two polynomials in Lagrange basis."""
    p1, p2 = _match_dims(poly1, poly2)
    if _matching_internal_domain(p1, p2):
        res_mi, res_c = _generic_lagrange_add(
            p1.multi_index.exponents, p1.coeffs, p2.multi_index.exponents, p2.coeffs
        )
        return LagrangePolynomial(
            res_c,
            res_mi,
            internal_domain=p1.internal_domain,
            user_domain=p1.user_domain,
        )
    else:
        raise NotImplementedError(
            "Addition is not implemented for lagrange polynomials with different domains"
        )


def _lagrange_sub(poly1, poly2):
    """Subtraction of two polynomials in Lagrange basis."""
    p1, p2 = _match_dims(poly1, poly2)
    if _matching_internal_domain(p1, p2):
        res_mi, res_c = _generic_lagrange_add(
            p1.multi_index.exponents, p1.coeffs, p2.multi_index.exponents, -p2.coeffs
        )
        return LagrangePolynomial(
            res_c,
            res_mi,
            internal_domain=p1.internal_domain,
            user_domain=p1.user_domain,
        )
    else:
        raise NotImplementedError(
            "Subtraction is not implemented for lagrange polynomials with different domains"
        )


def _lagrange_mul(poly1, poly2):
    """Multiplication of two polynomials in Lagrange basis."""
    p1, p2 = _match_dims(poly1, poly2)
    if _matching_internal_domain(p1, p2):
        l2n_1 = minterpy.TransformationLagrangeToNewton(p1)
        l2n_2 = minterpy.TransformationLagrangeToNewton(p2)

        degree_poly1 = p1.multi_index.poly_degree
        degree_poly2 = p2.multi_index.poly_degree
        lpdegree_poly1 = p1.multi_index.lp_degree
        lpdegree_poly2 = p2.multi_index.lp_degree

        res_degree = int(degree_poly1 + degree_poly2)
        res_lpdegree = lpdegree_poly1 + lpdegree_poly2

        res_mi = minterpy.MultiIndex.from_degree(
            p1.spatial_dimension, res_degree, res_lpdegree
        )
        res_grid = minterpy.Grid(res_mi)

        num_points_res = res_grid.unisolvent_nodes.shape[0]

        num_points_poly1 = p1.unisolvent_nodes.shape[0]
        scale_up_poly1 = np.zeros((num_points_res, num_points_poly1))
        for i in range(num_points_poly1):
            scale_up_poly1[:, i] = newt_eval(
                res_grid.unisolvent_nodes,
                l2n_1.transformation_operator.array_repr_full[:, i],
                p1.multi_index.exponents,
                p1.grid.generating_points,
            )

        num_points_poly2 = p2.unisolvent_nodes.shape[0]
        scale_up_poly2 = np.zeros((num_points_res, num_points_poly2))
        for i in range(num_points_poly2):
            scale_up_poly2[:, i] = newt_eval(
                res_grid.unisolvent_nodes,
                l2n_2.transformation_operator.array_repr_full[:, i],
                p2.multi_index.exponents,
                p2.grid.generating_points,
            )

        res_c = np.multiply(scale_up_poly1 @ p1.coeffs, scale_up_poly2 @ p2.coeffs)
        print(res_c)
        return LagrangePolynomial(
            res_c,
            res_mi,
            internal_domain=p1.internal_domain,
            user_domain=p1.user_domain,
            grid=res_grid,
        )
    else:
        raise NotImplementedError(
            "Multiplication is not implemented for Lagrange polynomials with different domains"
        )


# TODO redundant
lagrange_generate_internal_domain = verify_domain
lagrange_generate_user_domain = verify_domain


class LagrangePolynomial(MultivariatePolynomialSingleABC):
    """
    class for defining polynomials in Lagrange basis

    a polynomial in Lagrange basis is the sum of so called Lagrange polynomials (each multiplied with a coefficient)
    a SINGLE Lagrange polynomial is per definition 1 on one of the grid points and 0 on all others

    NOTE:
    a polynomial in Lagrange basis is well defined also for multi indices which are lexicographically incomplete.
    This means that the corresponding Lagrange polynomials also form a basis in such cases.
    These Lagrange polynomials however will possess their special property of being 1 on a single grid point
        and 0 on all others, with respect to the given grid!
    This allows defining very "sparse" polynomials (few multi indices -> few coefficients),
        but which still fulfill additional constraints (vanish on additional grid points).
    Practically this can be achieved by storing a "larger" grid (defined on a larger set of multi indices).
    In this case the transformation matrices become non-square, since there are fewer Lagrange polynomials
        than there are grid points (<-> only some of the Lagrange polynomials of this basis are "active").
    Conceptually this is equal to fix the "inactivate" coefficients to always be 0.
    """

    _newt_coeffs_lagr_monomials: Optional[ARRAY] = None

    # Virtual Functions
    _add = staticmethod(_lagrange_add)
    _sub = staticmethod(_lagrange_sub)
    _mul = staticmethod(_lagrange_mul)
    _div = staticmethod(dummy)
    _pow = staticmethod(dummy)
    _eval = staticmethod(dummy)

    generate_internal_domain = staticmethod(lagrange_generate_internal_domain)
    generate_user_domain = staticmethod(lagrange_generate_user_domain)

    @property
    def newt_coeffs_lagr_monomials(self) -> ARRAY:
        """the Newton coefficients of all active Lagrange monomials"""
        if self._newt_coeffs_lagr_monomials is None:  # lazy initialisation
            transformer_l2n = minterpy.TransformationLagrangeToNewton(self)
            # TODO more performant alternative?
            self._newt_coeffs_lagr_monomials = (
                transformer_l2n.transformation_operator.array_repr_sparse
            )
        return self._newt_coeffs_lagr_monomials

    def eval_lagrange_monomials_on(self, points: ARRAY) -> ARRAY:
        """computes the values of all Lagrange monomials at all k input points
        :param points: (m x k) the k points to evaluate on.
        :return: (k x N) the value of each active Lagrange monomial in Newton form at each point.
        """
        grid = self.grid
        generating_points = grid.generating_points
        # ATTENTION: ALL Newton polynomials of a basis are required to represent any single Lagrange polynomial
        # -> even for evaluating only some "active" Lagrange polynomials (corresponding to self.multi_index)
        # always ALL exponents from the basis (corresponding to grid.multi_index) are required.
        exponents = grid.multi_index.exponents
        coefficients = self.newt_coeffs_lagr_monomials
        return newt_eval(points, coefficients, exponents, generating_points)
