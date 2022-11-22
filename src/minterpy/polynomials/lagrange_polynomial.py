"""
LagrangePolynomial class
"""
from typing import Any, Optional

import numpy as np

import minterpy
from minterpy.global_settings import ARRAY

from ..core import Grid, MultiIndexSet
from ..core.ABC import MultivariatePolynomialSingleABC
from ..core.verification import verify_domain
from .canonical_polynomial import _match_dims, _matching_internal_domain

__all__ = ["LagrangePolynomial"]


def dummy(x: Optional[Any] = None) -> None:
    """Placeholder function.

    .. warning::
      This feature is not implemented yet!
    """
    raise NotImplementedError("This feature is not implemented yet.")


# TODO : poly2 can be of a different basis?
def _lagrange_add(
    poly1: MultivariatePolynomialSingleABC, poly2: MultivariatePolynomialSingleABC
) -> MultivariatePolynomialSingleABC:
    """Addition of two polynomials in Lagrange basis.


    :param poly1: First polynomial to be added.
    :type poly1: LagrangePolynomial
    :param poly2: Second polynomial to be added.
    :type poly2: LagrangePolynomial

    :return: The result of ``poly1 + poly2``
    :rtype: LagrangePolynomial

    """
    p1, p2 = _match_dims(poly1, poly2)
    if _matching_internal_domain(p1, p2):
        l2n_p1 = minterpy.transformations.LagrangeToNewton(p1)
        newt_p1 = l2n_p1()
        l2n_p2 = minterpy.transformations.LagrangeToNewton(p2)
        newt_p2 = l2n_p2()

        max_poly_degree = np.max(
            np.array([p1.multi_index.poly_degree, p2.multi_index.poly_degree])
        )
        max_lp_degree = np.max(
            np.array([p1.multi_index.lp_degree, p2.multi_index.lp_degree])
        )

        dim = p1.spatial_dimension  # must be the same for p2

        res_mi = MultiIndexSet.from_degree(dim, int(max_poly_degree), max_lp_degree)
        res_grid = Grid(res_mi)

        un = res_grid.unisolvent_nodes

        eval_p1 = newt_p1(un)
        eval_p2 = newt_p2(un)

        res_coeffs = eval_p1 + eval_p2

        return LagrangePolynomial(
            res_mi,
            res_coeffs,
            grid=res_grid,
            internal_domain=p1.internal_domain,
            user_domain=p1.user_domain,
        )
    else:
        raise NotImplementedError(
            "Addition is not implemented for lagrange polynomials with different domains"
        )


def _lagrange_sub(
    poly1: MultivariatePolynomialSingleABC, poly2: MultivariatePolynomialSingleABC
) -> MultivariatePolynomialSingleABC:
    """Subtraction of two polynomials in Lagrange basis.


    :param poly1: First polynomial from which will be substracted.
    :type poly1: LagrangePolynomial
    :param poly2: Second polynomial which is substracted.
    :type poly2: LagrangePolynomial

    :return: The result of ``poly1 - poly2``
    :rtype: LagrangePolynomial
    """
    p1, p2 = _match_dims(poly1, poly2)
    if _matching_internal_domain(p1, p2):
        l2n_p1 = minterpy.transformations.LagrangeToNewton(p1)
        newt_p1 = l2n_p1()
        l2n_p2 = minterpy.transformations.LagrangeToNewton(p2)
        newt_p2 = l2n_p2()

        max_poly_degree = np.max(
            np.array([p1.multi_index.poly_degree, p2.multi_index.poly_degree])
        )
        max_lp_degree = np.max(
            np.array([p1.multi_index.lp_degree, p2.multi_index.lp_degree])
        )

        dim = p1.spatial_dimension  # must be the same for p2

        res_mi = MultiIndexSet.from_degree(dim, int(max_poly_degree), max_lp_degree)
        res_grid = Grid(res_mi)

        un = res_grid.unisolvent_nodes

        eval_p1 = newt_p1(un)
        eval_p2 = newt_p2(un)

        res_coeffs = eval_p1 - eval_p2

        return LagrangePolynomial(
            res_mi,
            res_coeffs,
            grid=res_grid,
            internal_domain=p1.internal_domain,
            user_domain=p1.user_domain,
        )
    else:
        raise NotImplementedError(
            "Subtraction is not implemented for lagrange polynomials with different domains"
        )


def _lagrange_mul(
    poly1: MultivariatePolynomialSingleABC, poly2: MultivariatePolynomialSingleABC
) -> MultivariatePolynomialSingleABC:
    """Multiplication of two polynomials in Lagrange basis.


    :param poly1: First polynomial to be multiplied.
    :type poly1: LagrangePolynomial
    :param poly2: Second polynomial to be multiplied.
    :type poly2: LagrangePolynomial

    :return: The result of ``poly1 * poly2``
    :rtype: LagrangePolynomial
    """
    p1, p2 = _match_dims(poly1, poly2)
    if _matching_internal_domain(p1, p2):
        l2n_p1 = minterpy.transformations.LagrangeToNewton(p1)
        newt_p1 = l2n_p1()
        l2n_p2 = minterpy.transformations.LagrangeToNewton(p2)
        newt_p2 = l2n_p2()

        degree_poly1 = p1.multi_index.poly_degree
        degree_poly2 = p2.multi_index.poly_degree
        lpdegree_poly1 = p1.multi_index.lp_degree
        lpdegree_poly2 = p2.multi_index.lp_degree

        res_degree = int(degree_poly1 + degree_poly2)
        res_lpdegree = lpdegree_poly1 + lpdegree_poly2

        res_mi = MultiIndexSet.from_degree(
            p1.spatial_dimension, res_degree, res_lpdegree
        )
        res_grid = Grid(res_mi)

        un = res_grid.unisolvent_nodes

        eval_p1 = newt_p1(un)
        eval_p2 = newt_p2(un)

        res_coeffs = eval_p1 * eval_p2

        return LagrangePolynomial(
            res_mi,
            res_coeffs,
            grid=res_grid,
            internal_domain=p1.internal_domain,
            user_domain=p1.user_domain,
        )
    else:
        raise NotImplementedError(
            "Multiplication is not implemented for Lagrange polynomials with different domains"
        )


# TODO redundant
lagrange_generate_internal_domain = verify_domain
lagrange_generate_user_domain = verify_domain


class LagrangePolynomial(MultivariatePolynomialSingleABC):
    """Datatype to discribe polynomials in Lagrange base.

    A polynomial in Lagrange basis is the sum of so called Lagrange polynomials (each multiplied with a coefficient).
    A `single` Lagrange monomial is per definition 1 on one of the grid points and 0 on all others.

    Attributes
    ----------
    coeffs
    nr_active_monomials
    spatial_dimension
    unisolvent_nodes

    Notes
    -----
    A polynomial in Lagrange basis is well defined also for multi indices which are lexicographically incomplete. This means that the corresponding Lagrange polynomials also form a basis in such cases. These Lagrange polynomials however will possess their special property of being 1 on a single grid point and 0 on all others, with respect to the given grid! This allows defining very "sparse" polynomials (few multi indices -> few coefficients), but which still fulfill additional constraints (vanish on additional grid points). Practically this can be achieved by storing a "larger" grid (defined on a larger set of multi indices). In this case the transformation matrices become non-square, since there are fewer Lagrange polynomials than there are grid points (<-> only some of the Lagrange polynomials of this basis are "active"). Conceptually this is equal to fix the "inactivate" coefficients to always be 0.

    .. todo::
        - provide a short definition of this base here.
    """

    _newt_coeffs_lagr_monomials: Optional[ARRAY] = None

    # Virtual Functions
    _add = staticmethod(_lagrange_add)
    _sub = staticmethod(_lagrange_sub)
    _mul = staticmethod(_lagrange_mul)
    _div = staticmethod(dummy)  # type: ignore
    _pow = staticmethod(dummy)  # type: ignore
    _eval = staticmethod(dummy)  # type: ignore

    _partial_diff = staticmethod(dummy)
    _diff = staticmethod(dummy)

    generate_internal_domain = staticmethod(lagrange_generate_internal_domain)
    generate_user_domain = staticmethod(lagrange_generate_user_domain)
