"""
Concrete Transformation classes for the NewtonPolynomials
"""

from minterpy.newton_polynomial import NewtonPolynomial
from minterpy.canonical_polynomial import CanonicalPolynomial
from minterpy.lagrange_polynomial import LagrangePolynomial
from minterpy.transformation_abstract import TransformationABC
from minterpy.transformation_utils import _build_newton_to_canonical_operator, _build_newton_to_lagrange_operator

__all__ = ['TransformationNewtonToCanonical', 'TransformationNewtonToLagrange']


class TransformationNewtonToCanonical(TransformationABC):
    """
    Transformation from NewtonPolynomial to CanonicalPolynomial
    """
    _short_name = "newton2canonical"
    origin_type = NewtonPolynomial
    target_type = CanonicalPolynomial
    _get_transformation_operator = _build_newton_to_canonical_operator


class TransformationNewtonToLagrange(TransformationABC):
    """
    Transformation from NewtonPolynomial to LagrangePolynomial
    """
    _short_name = "newton2lagrange"
    origin_type = NewtonPolynomial
    target_type = LagrangePolynomial
    _get_transformation_operator = _build_newton_to_lagrange_operator
