"""
Module providing high-level helper functions for polynomial basis transformations.
"""

from minterpy.multivariate_polynomial_abstract import MultivariatePolynomialSingleABC
from minterpy.transformation_abstract import TransformationABC
from minterpy.transformation_identity import TransformationIdentity

__all__ = ["get_transformation", "get_transformation_class"]


def get_transformation_class(origin_type, target_type) -> TransformationABC:
    """Finds the Transformation class to go from origin_type to target_type

    A registry of 'available_transforms' is maintained in the TransformationABC. This function performs a lookup in
    this registry to retrieve the correct Transformation class.

    :param origin_type: data type of the origin polynomial
    :param target_type: data type of the target polynomial
    :return: the Transformation class that can perform this transformation.

    """
    try:
        if origin_type == target_type:
            return TransformationIdentity
        else:
            return TransformationABC.available_transforms[(origin_type, target_type)]
    except IndexError:
        raise NotImplementedError(
            f"there is no known transformation from {origin_type} to {target_type}"
        )

def get_transformation(origin_polynomial: MultivariatePolynomialSingleABC, target_type) -> TransformationABC:
    """Finds the Transformation class that can transform the basis of the origin_polynomial to the desired target_type.

    :param origin_polynomial: an instance of the origin polynomial
    :param target_type: data type of the target polynomial
    :return: the Transformation class that can perform this transformation.

    """
    transformation_class = get_transformation_class(
        origin_polynomial.__class__, target_type
    )
    return transformation_class(origin_polynomial)
