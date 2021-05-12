#!/usr/bin/env python
""" functions for determining the required transformation class

TODO make this module obsolete by introducing a global transformer class (s. transformation_abstract.py)
"""

from minterpy import (CanonicalPolynomial, LagrangePolynomial,
                      MultivariatePolynomialABC, NewtonPolynomial,
                      TransformationCanonicalToLagrange,
                      TransformationCanonicalToNewton,
                      TransformationLagrangeToCanonical,
                      TransformationLagrangeToNewton,
                      TransformationNewtonToCanonical,
                      TransformationNewtonToLagrange)
from minterpy.transformation_identity import TransformationIdentity

__author__ = "Jannik Michelfeit"
__copyright__ = "Copyright 2021, minterpy"
__credits__ = ["Jannik Michelfeit"]
# __license__ =
# __version__ =
# __maintainer__ =
__email__ = "jannik@michelfe.it"
__status__ = "Development"

available_transformations = {
    # (origin_type, target_type): class
    (LagrangePolynomial, NewtonPolynomial): TransformationLagrangeToNewton,
    (LagrangePolynomial, CanonicalPolynomial): TransformationLagrangeToCanonical,
    (CanonicalPolynomial, NewtonPolynomial): TransformationCanonicalToNewton,
    (CanonicalPolynomial, LagrangePolynomial): TransformationCanonicalToLagrange,
    (NewtonPolynomial, LagrangePolynomial): TransformationNewtonToLagrange,
    (NewtonPolynomial, CanonicalPolynomial): TransformationNewtonToCanonical,
    (LagrangePolynomial, LagrangePolynomial): TransformationIdentity,
    (NewtonPolynomial, NewtonPolynomial): TransformationIdentity,
    (CanonicalPolynomial, CanonicalPolynomial): TransformationIdentity,
}


def get_transformation_class(origin_type, target_type):
    try:
        return available_transformations[(origin_type, target_type)]
    except IndexError:
        raise NotImplementedError(
            f"the is no known transformation from {origin_type} to {target_type}"
        )


def get_transformation(origin_polynomial: MultivariatePolynomialABC, target_type):
    transformation_class = get_transformation_class(
        origin_polynomial.__class__, target_type
    )
    return transformation_class(origin_polynomial)
