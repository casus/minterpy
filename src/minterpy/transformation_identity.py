from copy import copy

import numpy as np

from minterpy import TransformationABC, MultivariatePolynomialABC


def _build_identity_transformation(self):
    return np.identity(len(self.multi_index))


class TransformationIdentity(TransformationABC):
    """
    Transformation into the identical basis
    """
    _short_name = "identity"
    origin_type = MultivariatePolynomialABC
    target_type = MultivariatePolynomialABC
    _get_transformation_matrix = _build_identity_transformation

    def _apply_transformation(self, origin_poly):
        return copy(origin_poly)  # no need to compute dot product