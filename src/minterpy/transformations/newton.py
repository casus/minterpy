"""
Concrete implementations of the Transformation classes for the NewtonPolynomial.

Transformations from Newton basis to Canonical and Lagrange basis are provided.
"""

from minterpy import CanonicalPolynomial
from minterpy import LagrangePolynomial
from minterpy.polynomials.newton_polynomial import NewtonPolynomial
from minterpy.core.ABC import TransformationABC
from .utils import (_build_newton_to_canonical_operator,
                                           _build_newton_to_lagrange_operator)

__all__ = ["NewtonToCanonical", "NewtonToLagrange"]


class NewtonToCanonical(TransformationABC):
    """Transformation from NewtonPolynomial to CanonicalPolynomial
    """

    origin_type = NewtonPolynomial
    target_type = CanonicalPolynomial
    _get_transformation_operator = _build_newton_to_canonical_operator


class NewtonToLagrange(TransformationABC):
    """Transformation from NewtonPolynomial to LagrangePolynomial
    """

    origin_type = NewtonPolynomial
    target_type = LagrangePolynomial
    _get_transformation_operator = _build_newton_to_lagrange_operator
