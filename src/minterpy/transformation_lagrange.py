"""
Concrete implementations of the Transformation classes for the LagrangePolynomial.

Transformations from Lagrange basis to Newton and Canonical basis are provided.
"""

import minterpy
from minterpy.canonical_polynomial import CanonicalPolynomial
from minterpy.newton_polynomial import NewtonPolynomial
from minterpy.transformation_abstract import TransformationABC
from minterpy.transformation_utils import (
    _build_lagrange_to_canonical_operator, _build_lagrange_to_newton_operator)

__all__ = ["TransformationLagrangeToNewton", "TransformationLagrangeToCanonical"]


class TransformationLagrangeToNewton(TransformationABC):
    """Transformation from LagrangePolynomial to NewtonPolynomial
    """

    origin_type = minterpy.LagrangePolynomial
    target_type = NewtonPolynomial
    _get_transformation_operator = _build_lagrange_to_newton_operator


class TransformationLagrangeToCanonical(TransformationABC):
    """Transformation from LagrangePolynomial to CanonicalPolynomial
    """

    origin_type = minterpy.LagrangePolynomial
    target_type = CanonicalPolynomial
    _get_transformation_operator = _build_lagrange_to_canonical_operator
