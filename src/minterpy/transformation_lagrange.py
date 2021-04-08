"""
Concrete Transformation classes for the LagrangePolynomials
"""

import minterpy
from minterpy.canonical_polynomial import CanonicalPolynomial
from minterpy.newton_polynomial import NewtonPolynomial
from minterpy.transformation_abstract import TransformationABC
from minterpy.transformation_utils import _build_lagrange_to_newton, _build_lagrange_to_canonical

__all__ = ['TransformationLagrangeToNewton', 'TransformationLagrangeToCanonical']


class TransformationLagrangeToNewton(TransformationABC):
    """
    Transformation from LagrangePolynomial to NewtonPolynomial
    """
    _short_name = "lagrange2newton"
    origin_type = minterpy.LagrangePolynomial
    target_type = NewtonPolynomial
    _get_transformation_matrix = _build_lagrange_to_newton


class TransformationLagrangeToCanonical(TransformationABC):
    """
    Transformation from LagrangePolynomial to CanonicalPolynomial
    """
    _short_name = "lagrange2canonical"
    origin_type = minterpy.LagrangePolynomial
    target_type = CanonicalPolynomial
    _get_transformation_matrix = _build_lagrange_to_canonical
