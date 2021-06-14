"""
Concrete implementation of Identity transformation when there is no basis change.

Notes
-----
This class is needed for completeness reasons and also to enable the high-level helper functions for polynomial basis
transformations to handle transformations between identical basis.
"""

from copy import copy

import numpy as np

from minterpy import MultivariatePolynomialABC, TransformationABC
from minterpy.transformation_operators import MatrixTransformationOperator


def _build_identity_transformation_operator(transformation):
    return MatrixTransformationOperator(
        transformation, np.identity(len(transformation.multi_index))
    )


class TransformationIdentity(TransformationABC):
    """Transformation between same basis.
    """

    origin_type = MultivariatePolynomialABC
    target_type = MultivariatePolynomialABC
    _get_transformation_operator = _build_identity_transformation_operator

    def _apply_transformation(self, origin_poly):
        return copy(origin_poly)  # no need to compute dot product
