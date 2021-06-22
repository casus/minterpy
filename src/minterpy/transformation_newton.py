"""
Concrete implementations of the Transformation classes for the NewtonPolynomial.

Transformations from Newton basis to Canonical and Lagrange basis are provided.
"""

from minterpy.canonical_polynomial import CanonicalPolynomial
from minterpy.lagrange_polynomial import LagrangePolynomial
from minterpy.newton_polynomial import NewtonPolynomial
from minterpy.transformation_abstract import TransformationABC
from minterpy.transformation_utils import (_build_newton_to_canonical_operator,
                                           _build_newton_to_lagrange_operator)

__all__ = ["TransformationNewtonToCanonical", "TransformationNewtonToLagrange"]


class TransformationNewtonToCanonical(TransformationABC):
    """Transformation from NewtonPolynomial to CanonicalPolynomial
    """

    origin_type = NewtonPolynomial
    target_type = CanonicalPolynomial
    _get_transformation_operator = _build_newton_to_canonical_operator


class TransformationNewtonToLagrange(TransformationABC):
    """Transformation from NewtonPolynomial to LagrangePolynomial
    """

    origin_type = NewtonPolynomial
    target_type = LagrangePolynomial
    _get_transformation_operator = _build_newton_to_lagrange_operator
