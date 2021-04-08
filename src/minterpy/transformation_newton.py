"""
Concrete Transformation classes for the NewtonPolynomials
"""

from minterpy.newton_polynomial import NewtonPolynomial
from minterpy.canonical_polynomial import CanonicalPolynomial
from minterpy.lagrange_polynomial import LagrangePolynomial
from minterpy.transformation_abstract import TransformationABC
from minterpy.transformation_utils import _build_newton_to_canonical, _build_newton_to_lagrange

__all__ = ['TransformationNewtonToCanonical', 'TransformationNewtonToLagrange']


class TransformationNewtonToCanonical(TransformationABC):
    """
    Transformation from NewtonPolynomial to CanonicalPolynomial
    """
    _short_name = "newton2canonical"
    origin_type = NewtonPolynomial
    target_type = CanonicalPolynomial
    _get_transformation_matrix = _build_newton_to_canonical


class TransformationNewtonToLagrange(TransformationABC):
    """
    Transformation from NewtonPolynomial to LagrangePolynomial
    """
    _short_name = "newton2lagrange"
    origin_type = NewtonPolynomial
    target_type = LagrangePolynomial
    _get_transformation_matrix = _build_newton_to_lagrange
