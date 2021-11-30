"""
Concrete implementations of the Transformation classes for the LagrangePolynomial.

Transformations from Lagrange basis to Newton and Canonical basis are provided.
"""
from minterpy import CanonicalPolynomial, NewtonPolynomial
from minterpy.core.ABC import TransformationABC
from minterpy.polynomials.lagrange_polynomial import LagrangePolynomial

from .utils import (_build_lagrange_to_canonical_operator,
                    _build_lagrange_to_newton_operator)

__all__ = ["LagrangeToNewton", "LagrangeToCanonical"]


class LagrangeToNewton(TransformationABC):
    """Transformation from LagrangePolynomial to NewtonPolynomial"""

    origin_type = LagrangePolynomial
    target_type = NewtonPolynomial
    _get_transformation_operator = _build_lagrange_to_newton_operator


class LagrangeToCanonical(TransformationABC):
    """Transformation from LagrangePolynomial to CanonicalPolynomial"""

    origin_type = LagrangePolynomial
    target_type = CanonicalPolynomial
    _get_transformation_operator = _build_lagrange_to_canonical_operator
