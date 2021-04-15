"""
LagrangePolynomial class
"""
from typing import Optional

import numpy as np

import minterpy
from minterpy.global_settings import ARRAY
from minterpy.multivariate_polynomial_abstract import MultivariatePolynomialSingleABC
from minterpy.utils import newt_eval
from minterpy.verification import verify_domain

__all__ = ['LagrangePolynomial']


def dummy():
    raise NotImplementedError(f"This feature is not implemented yet.")


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

    _newt_coeffs_lagr_monomials:Optional[ARRAY] = None

    # Virtual Functions
    _add = staticmethod(dummy)
    _sub = staticmethod(dummy)
    _mul = staticmethod(dummy)
    _div = staticmethod(dummy)
    _pow = staticmethod(dummy)
    _eval = staticmethod(dummy)

    generate_internal_domain = staticmethod(lagrange_generate_internal_domain)
    generate_user_domain = staticmethod(lagrange_generate_user_domain)

    @property
    def newt_coeffs_lagr_monomials(self):
        if self._newt_coeffs_lagr_monomials is None:  # lazy initialisation
            transformer_l2n = minterpy.TransformationLagrangeToNewton(self)
            # TODO more performant alternative?
            self._newt_coeffs_lagr_monomials = transformer_l2n.transformation_operator.to_array()
        return self._newt_coeffs_lagr_monomials

    def eval_lagrange_monomials_on(self, points: np.ndarray):
        """ computes the values of all Lagrange monomials at all k input points
        :param points: (m x k) the k points to evaluate on.
        :return: (k x N) the value of each Lagrange monomial in Newton form at each point.
        """
        grid = self.grid
        generating_points = grid.generating_points
        # ATTENTION: ALL Newton polynomials of a basis are required to represent any single Lagrange polynomial
        # -> for evaluating the "active" Lagrange polynomials (corresponding to self.multi_index)
        # always ALL exponents from the basis (corresponding to the grid) are required.
        exponents = grid.multi_index.exponents
        coefficients = self.newt_coeffs_lagr_monomials
        return newt_eval(points, coefficients, exponents, generating_points)
