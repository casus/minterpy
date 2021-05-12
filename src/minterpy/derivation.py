#!/usr/bin/env python
""" functions for deriving multivariate polynomials in different bases
"""

from typing import Optional, Type, Union

import numpy as np

from minterpy import (
    CanonicalPolynomial,
    MultiIndex,
    MultivariatePolynomialABC,
    MultivariatePolynomialSingleABC,
)
from minterpy.global_settings import ARRAY, DEBUG, FLOAT_DTYPE
from minterpy.jit_compiled_utils import (
    compute_grad_c2c,
    compute_grad_x2c,
    get_match_idx,
)
from minterpy.joint_polynomial import JointPolynomial
from minterpy.transformation_meta import get_transformation
from minterpy.verification import check_is_square

__author__ = "Jannik Michelfeit"
__copyright__ = "Copyright 2021, minterpy"
__credits__ = ["Jannik Michelfeit"]
# __license__ =
# __version__ =
# __maintainer__ =
__email__ = "jannik@michelfe.it"
__status__ = "Development"


# TODO use barycentrical transformations!

# TODO test
def partial_derivative_canonical(
    dim_idx: int, coeffs_canonical: np.ndarray, exponents: np.ndarray
) -> np.ndarray:
    """
    :param dim_idx: the index of the dimension to derive with respect to
    :param coeffs_canonical: the coefficients of the polynomial in canonical form
    :param exponents: the respective exponent vectors of all monomials,
        ATTENTION: have to be complete
    :return: the coefficients of the partial derivative in the same ordering (and canonical form)
    """
    coeffs_canonical_deriv = np.zeros(coeffs_canonical.shape)
    for monomial_idx, coeff in enumerate(coeffs_canonical):
        monomial_exponents = exponents[monomial_idx, :]
        if monomial_exponents[dim_idx] > 0:
            mon_exponents_derived = monomial_exponents.copy()
            mon_exponents_derived[dim_idx] -= 1
            # "gradient exponential mapping"
            new_coeff_idx = get_match_idx(exponents, mon_exponents_derived)
            # multiply with exponent
            coeffs_canonical_deriv[new_coeff_idx] = (
                coeff * exponents[monomial_idx, dim_idx]
            )
    return coeffs_canonical_deriv


# TODO test
def derive_gradient_canonical(
    coeffs_canonical: np.ndarray, exponents: np.ndarray
) -> np.ndarray:
    """derives the gradient without any precomputation

    :param coeffs_canonical: the coefficients of the polynomial in canonical form
    :param exponents: the respective exponent vectors of all monomials
    :return: the gradient in canonical form
    """
    nr_of_monomials, dimensionality = exponents.shape
    nr_coefficients = len(coeffs_canonical)
    assert nr_of_monomials == nr_coefficients, (
        "coefficient and exponent shapes do not match: "
        f"{coeffs_canonical.shape}, {exponents.shape}"
    )
    gradient = np.empty((nr_of_monomials, dimensionality))
    for dim_idx in range(dimensionality):
        coeffs_canonical_deriv = partial_derivative_canonical(
            dim_idx, coeffs_canonical, exponents
        )
        gradient[:, dim_idx] = coeffs_canonical_deriv
    return gradient


# # TODO remove Transformer class
# def partial_derivative_lagrange(transformer: Transformer, dim_idx: int,
#                                 coeffs_lagrange: np.ndarray) -> np.ndarray:
#     coeffs_canonical = transformer.transform_l2c(coeffs_lagrange)
#     # exponents are equal to the multi indices of the interpolation tree
#     exponents = transformer.tree.exponents
#     coeffs_canonical_deriv = partial_derivative_canonical(dim_idx, coeffs_canonical, exponents)
#     coeffs_lagrange_deriv = transformer.transform_c2l(coeffs_canonical_deriv)
#     return coeffs_lagrange_deriv
#
#
# def gradient_lagrange(transformer: Transformer, coeffs_lagrange: np.ndarray) -> np.ndarray:
#     assert len(coeffs_lagrange) == transformer.N
#     dimensionality = transformer.m
#     coeffs_canonical = transformer.transform_l2c(coeffs_lagrange)
#     # exponents are equal to the multi indices of the interpolation tree
#     exponents = transformer.tree.exponents
#     grad_canonical = derive_gradient_canonical(coeffs_canonical, exponents)
#     grad_lagrange = np.empty(grad_canonical.shape)
#     for dim_idx in range(dimensionality):
#         grad_lagrange[dim_idx, :] = transformer.transform_c2l(grad_canonical[dim_idx, :])
#     return grad_lagrange


def tensor_right_product(tensor: np.ndarray, right_factor: np.ndarray):
    # tensor dot product a . b: sum over the last axis of a and the first axis of b in order
    # = sum over tensor axis 2 and right factor axis 0
    # (m x N x N) @ (N x k) -> (m x N x k)
    return np.tensordot(tensor, right_factor, axes=1)  # <- dot product


def get_gradient_coeffs(coefficients, gradient_operator):
    """computes the gradient using a precomputed operator tensor

    @param coefficients: the coefficients of a polynomial in basis a
    @param gradient_operator: the gradient operation tensor from basis a to b
    @return: the gradient in basis b
    """
    return tensor_right_product(gradient_operator, coefficients)


def tensor_left_product(left_factor: np.ndarray, tensor: np.ndarray):
    """
    in the case of multiplication with e.g. a conversion matrix (N x X):
    tensor should maintain shape: (N x N) @ (m x N x N) -> (m x N x N)

    NOTE: equal to
    tensor = np.swapaxes(tensor, 1, 2)
    product = tensor_right_product(tensor, left_factor.T)
    # "transpose" back:
    product = np.swapaxes(product, 1, 2)

    NOTE: np.tensordot is not supported by numba!
    """
    tensor = np.transpose(tensor, (1, 0, 2))  # (N x m x N)
    product = np.tensordot(left_factor, tensor, axes=1)  # <- dot product
    # "transpose" back:
    product = np.transpose(product, (1, 0, 2))  # (N x m x N)
    return product


def _get_gradient_operator(
    exponents: np.ndarray,
    x2c: Optional[np.ndarray] = None,
    c2x: Optional[np.ndarray] = None,
) -> np.ndarray:
    """computes the gradient operation tensor from a variable basis to a variable basis

    O(mN^3) due to matrix multiplications

    @param x2c: transformation matrix from the origin basis to canonical basis,
        or None when the origin basis is canonical
    @param c2x: transformation matrix from canonical basis to the target basis,
        or None when the target basis is canonical
    @param exponents: matrix of all exponent vectors.
        ATTENTION: has to contain also the exponent vectors of all derivatives, which can otherwise not be found
            <-> exponents/multi indices have to be "complete"!
    @return: the tensor of the m partial derivative operators from Lagrange basis to Lagrange basis
    """

    nr_of_monomials, dimensionality = exponents.shape

    # gradient operation tensor from basis x to canonical basis
    grad_x2c = np.zeros((dimensionality, nr_of_monomials, nr_of_monomials))
    if x2c is None:  # no transformation -> origin basis is canonical
        compute_grad_c2c(grad_x2c, exponents)
    else:
        compute_grad_x2c(grad_x2c, exponents, x2c)
    # back to the original basis
    if c2x is None:  # no transformation -> target basis is canonical
        return grad_x2c
    grad_x2x = tensor_left_product(c2x, grad_x2c)
    return grad_x2x


class SinglePolyDerivator:
    def __init__(
        self,
        origin_poly: MultivariatePolynomialSingleABC,
        target_type: Optional[Type[MultivariatePolynomialSingleABC]] = None,
    ):
        """

        with fixed interpolation nodes ("grid") and exponents,
            the operation transforming coefficients into a gradient
            (= (m x N x N) tensor) can be precomputed
        """
        origin_type = type(origin_poly)
        if not issubclass(origin_type, MultivariatePolynomialSingleABC):
            raise TypeError(
                f"<{origin_poly}> is of type {origin_type}, "
                f"not of the expected type {MultivariatePolynomialSingleABC}"
            )

        if (
            target_type is None
        ):  # if no specific output type is specified, the output should be of the same class
            target_type = origin_type
        if not issubclass(target_type, MultivariatePolynomialSingleABC):
            raise ValueError("the specified target type must be a polynomial class")
        self.target_type = target_type

        origin_is_canonical = origin_type == CanonicalPolynomial
        target_is_canonical = self.target_type == CanonicalPolynomial

        # if not (target_is_canonical and origin_is_canonical):
        #     # ATTENTION: derivation (except in canonical basis) requires complete index sets
        #     #  due to the DDS scheme required for the X to canonical transformation
        #     # -> the multi indices in use must be complete (= without any "holes")
        # when canonical2canonical TODO only add the indices which are required for derivation (partial derivatives)
        #   origin_poly.make_derivable()
        # TODO ATTENTION: gradient operator has to "know" that some of the monomials are not relevant
        #  -> no derivation of these monomials

        # TODO by completing a polynomial. its definition (basis) changes. allowed?
        origin_poly = origin_poly.make_complete()
        self.origin_poly = origin_poly

        if origin_is_canonical:  # canonical2canonical: no transformation required
            origin2canonical = None
        else:

            origin2canonical = get_transformation(
                origin_poly, CanonicalPolynomial
            ).transformation_operator.array_repr_full
            if DEBUG:
                check_is_square(origin2canonical, size=self.nr_of_monomials)

        if target_is_canonical:  # canonical2canonical: no transformation required
            canonical2target = None
        else:
            canonical_poly = CanonicalPolynomial(None, self.multi_index)
            canonical2target = get_transformation(
                canonical_poly, self.target_type
            ).transformation_operator.array_repr_full
            if DEBUG:
                # NOTE: not just multi index! <- indices might be separate
                self.num_monomials = len(self.origin_poly.grid.multi_index)
                check_is_square(canonical2target, size=self.nr_of_monomials)

        exponents = self.multi_index.exponents
        # TODO lazy evaluation?!
        self._gradient_op = _get_gradient_operator(
            exponents, origin2canonical, canonical2target
        )

    @property
    def multi_index(self) -> MultiIndex:
        # ATTENTION: the exponents of the grid (full basis!)
        return self.origin_poly.grid.multi_index

    @property
    def nr_of_monomials(self) -> int:
        # NOTE: the total amount of grid points, not the amount of active monomials!
        return len(self.origin_poly.grid.multi_index)

    @property
    def origin_type(self):
        return type(self.origin_poly)

    def _get_gradient_coeffs(self) -> ARRAY:
        # with fixed coefficients the gradient can be precomputed
        # (N x m): Newton coefficients of the partial derivative in each dimension
        # use output format expected for coefficients!
        origin_poly = self.origin_poly
        separate_indices = origin_poly.indices_are_separate
        if separate_indices:
            # NOTE: currently the gradient operator can only be applied to coefficient arrays of "full size"
            # TODO simplify. use correct slice of the gradient operator?
            # NOTE: not just multi index! <- indices might be separate
            coeffs = np.zeros(self.nr_of_monomials, dtype=FLOAT_DTYPE)
            active_idxs = origin_poly.active_monomials
            coeffs[active_idxs] = origin_poly.coeffs
        else:
            coeffs = origin_poly.coeffs

        grad_coeffs = get_gradient_coeffs(coeffs, self._gradient_op).T
        if separate_indices:
            # ATTENTION: also only use the coefficients of the active monomials
            grad_coeffs = grad_coeffs[active_idxs]
        return grad_coeffs

    def get_gradient_poly(self) -> MultivariatePolynomialSingleABC:
        # NOTE: the gradient is composed of m polynomials itself.
        # however numerical computations with list of polynomial instances are inefficient.
        #  -> using a single polynomial class with a "list of coefficients"
        grad_coeffs = self._get_gradient_coeffs()
        output_poly = self.target_type.from_poly(
            self.origin_poly, new_coeffs=grad_coeffs
        )
        return output_poly

    def _get_partial_deriv_coeffs(self) -> ARRAY:
        raise NotImplementedError

    def partial_derivative(self, dimension: int) -> MultivariatePolynomialSingleABC:
        raise NotImplementedError


class JointPolyDerivator:
    def __init__(
        self,
        origin_poly: JointPolynomial,
        target_type: Optional[Type[MultivariatePolynomialSingleABC]] = None,
    ):
        self._derivators = []
        for poly in origin_poly.sub_polynomials:
            self._derivators.append(SinglePolyDerivator(poly, target_type))

    def get_gradient_poly(self) -> JointPolynomial:
        """
        NOTE: the gradient of a JointPolynomial (sum of polynomials) itself is a JointPolynomial
        :return:
        """
        # TODO if all input polynomials are Lagrange polynomials. a LagrangeJointPolynomial must be constructed!
        grad_polys = [derivator.get_gradient_poly() for derivator in self._derivators]
        return JointPolynomial(grad_polys)


# TODO remove nesting!
class Derivator:
    def __init__(
        self,
        origin_poly: MultivariatePolynomialABC,
        target_type: Optional[Type[MultivariatePolynomialSingleABC]] = None,
    ):
        # pick and store the fitting derivator class internally:
        if type(origin_poly) is JointPolynomial:
            derivator = JointPolyDerivator(origin_poly, target_type)
        else:
            derivator = SinglePolyDerivator(origin_poly, target_type)
        self._derivator: Union[JointPolyDerivator, SinglePolyDerivator] = derivator

    def get_gradient_poly(self):
        return self._derivator.get_gradient_poly()
