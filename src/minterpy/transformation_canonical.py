"""
Concrete Transformation classes for the CanonicalPolynomials
"""

from minterpy import CanonicalPolynomial
from minterpy.lagrange_polynomial import LagrangePolynomial
from minterpy.newton_polynomial import NewtonPolynomial
from minterpy.transformation_abstract import TransformationABC
from minterpy.transformation_utils import _build_canonical_to_newton, _build_canonical_to_lagrange

__all__ = ['TransformationCanonicalToNewton', 'TransformationCanonicalToLagrange']


class TransformationCanonicalToNewton(TransformationABC):
    """
    Transformation from CanonicalPolynomial to NewtonPolynomial
    """
    _short_name = "canonical2newton"
    origin_type = CanonicalPolynomial
    target_type = NewtonPolynomial
    _get_transformation_matrix = _build_canonical_to_newton


class TransformationCanonicalToLagrange(TransformationABC):
    """
    Transformation from CanonicalPolynomial to LagrangePolynomial
    """
    _short_name = "canonical2lagrange"
    origin_type = CanonicalPolynomial
    target_type = LagrangePolynomial
    _get_transformation_matrix = _build_canonical_to_lagrange
