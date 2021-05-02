from copy import copy

import numpy as np

from minterpy import TransformationABC, MultivariatePolynomialABC
from minterpy.transformation_operators import MatrixTransformationOperator


def _build_identity_transformation_operator(transformation):
    return MatrixTransformationOperator(transformation, np.identity(len(transformation.multi_index)))


class TransformationIdentity(TransformationABC):
    """
    Transformation into the identical basis
    """
    _short_name = "identity"
    origin_type = MultivariatePolynomialABC
    target_type = MultivariatePolynomialABC
    _get_transformation_operator = _build_identity_transformation_operator

    def _apply_transformation(self, origin_poly):
        return copy(origin_poly)  # no need to compute dot product
