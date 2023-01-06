"""
Utility functions for computing matrices for transformation between canonical, lagrange, and newton basis.

"""
from typing import no_type_check

import numpy as np

from minterpy.core.ABC import OperatorABC, TransformationABC
from minterpy.dds import dds
from minterpy.global_settings import ARRAY, DEBUG, FLOAT_DTYPE
from minterpy.jit_compiled_utils import compute_vandermonde_n2c
from minterpy.schemes.barycentric.precomp import (
    _build_lagrange_to_newton_bary,
    _build_newton_to_lagrange_bary,
)
from minterpy.schemes.matrix_operator import MatrixOperator
from minterpy.utils import eval_newton_monomials

# NOTE: avoid looping over a numpy array! e.g. for j in np.arange(num_monomials):
# see: # https://stackoverflow.com/questions/10698858/built-in-range-or-numpy-arange-which-is-more-efficient


# todo simplify names


def invert_triangular(triangular_matrix: np.ndarray) -> np.ndarray:
    # FIXME: triangular inversion is not working! required when using barycentric transforms?
    # i, j = triangular_matrix.shape  # square matrix
    # inverted_matrix = solve_triangular(triangular_matrix, np.identity(i))
    inverted_matrix = np.linalg.inv(triangular_matrix)
    return inverted_matrix


def _build_n2l_array(grid, multi_index=None, require_invertible: bool = False) -> ARRAY:
    # NOTE: the indices might be different from the ones used in the grid!
    # -> just some "active" Lagrange polynomials
    if multi_index is None or require_invertible:
        # NOTE: the transformation matrix needs to be square for the inversion
        # even if some of the multi indices are "inactive",
        # the unisolvent nodes (grid) need to correspond to the multi indices!
        multi_index = grid.multi_index

    exponents = multi_index.exponents
    unisolvent_nodes = grid.unisolvent_nodes
    generating_points = grid.generating_points
    # NOTE: the shape of unisolvent_nodes and exponents might be different! -> non square transformation matrix
    transformation_matrix = eval_newton_monomials(
        unisolvent_nodes,
        exponents,
        generating_points,
        verify_input=DEBUG,
        triangular=True,
    )
    return transformation_matrix


def _build_newton_to_lagrange_naive(
    transformation: TransformationABC,
) -> MatrixOperator:
    """computes the Newton to Lagrange transformation given by an array

    SPECIAL PROPERTY: the evaluation of any polynomial on unisolvent nodes yields
        the Lagrange coefficients of this polynomial
        (for the Lagrange basis defined implicitly by these unisolvent nodes)
    -> evaluating every Newton polynomial (<-> the monomials forming the Newton basis) on all interpolation points,
        naturally yields the operator transforming the Newton coefficients into Lagrange coefficients

     NOTE: it is inefficient to compute this by inversion:
         newton_to_lagrange = inv(lagrange_to_newton) # based on DDS

    special property: half of the values will always be 0 (lower triangular matrix).

    :param require_invertible: weather or not the output matrix should be square
    """
    grid = transformation.grid
    transformation_matrix = _build_n2l_array(
        grid, transformation.origin_poly.multi_index
    )
    transformation_operator = MatrixOperator(transformation, transformation_matrix)
    return transformation_operator


def build_l2n_matrix_dds(grid):
    num_monomials = len(grid.multi_index)
    lagr_coeff_matrix = np.eye(num_monomials, dtype=FLOAT_DTYPE)
    tree = grid.tree
    lagrange_to_newton = dds(lagr_coeff_matrix, tree)
    return lagrange_to_newton


def _build_lagrange_to_newton_naive(
    transformation: TransformationABC,
) -> MatrixOperator:
    """computes the Lagrange to Newton transformation given by an array

    NOTE: each column of the L2N transformation matrix
        corresponds to the Newton coefficients of the respective Lagrange polynomial.

    NOTE: computing the L2N matrix could be done with the DDS scheme, but only if the multi indices are complete.
        in this case however it is more efficient to use the barycentric transformations right away.
    for the case that the indices are not complete the matrix must be computed by inverting the N2L transformation.
    """
    newton_to_lagrange = _build_n2l_array(transformation.grid, require_invertible=True)
    transformation_matrix = invert_triangular(newton_to_lagrange)
    transformation_operator = MatrixOperator(transformation, transformation_matrix)
    return transformation_operator


def _build_c2n_array(transformation) -> ARRAY:
    multi_index = transformation.grid.multi_index
    num_monomials = len(multi_index)
    V_n2c = np.ones((num_monomials, num_monomials), dtype=FLOAT_DTYPE)
    compute_vandermonde_n2c(
        V_n2c, transformation.grid.unisolvent_nodes, multi_index.exponents
    )  # computes the result "in place"
    tree = transformation.grid.tree
    c2n = dds(V_n2c, tree)
    return c2n


def _build_n2c_array(transformation: TransformationABC) -> ARRAY:
    # TODO achieve by calling inverse(n2c_operator)
    return invert_triangular(_build_c2n_array(transformation))


# TODO own module?


@no_type_check
def _build_newton_to_lagrange_operator(
    transformation: TransformationABC,
) -> OperatorABC:
    """constructs the Newton to Lagrange transformation operator

    use the barycentric transformation if the indices are complete!
    TODO find solution for the case that the multi indices are separate from the grid indices

    :param transformation: the Transformation instance
        with the fixed polynomial defining the unisolvent nodes to perform the transformation on
    :return: the transformation operator from Newton to Lagrange basis
    """
    grid = transformation.grid
    complete_indices = grid.multi_index.is_complete
    identical_indices = not transformation.origin_poly.indices_are_separate
    if complete_indices and identical_indices:
        # use barycentric transformation
        transformation_operator = _build_newton_to_lagrange_bary(transformation)
    else:  # use "naive" matrix transformation format
        transformation_operator = _build_newton_to_lagrange_naive(transformation)

    return transformation_operator


@no_type_check
def _build_lagrange_to_newton_operator(
    transformation: TransformationABC,
) -> OperatorABC:
    """constructs the Lagrange to Newton transformation operator

    use the barycentric transformation if the indices are complete!
    TODO find solution for the case that the multi indices are separate from the grid indices

    NOTE: it is inefficient to compute this by inversion:
        newton_to_lagrange = inv(lagrange_to_newton) # based on DDS

    :param transformation: the Transformation instance
        with the fixed polynomial defining the unisolvent nodes to perform the transformation on
    :return: the transformation operator from Newton to Lagrange basis
    """
    grid = transformation.grid
    complete_indices = grid.multi_index.is_complete
    identical_indices = not transformation.origin_poly.indices_are_separate
    if complete_indices and identical_indices:
        # use barycentric transformation
        transformation_operator = _build_lagrange_to_newton_bary(transformation)
    else:  # use "naive" matrix transformation format
        transformation_operator = _build_lagrange_to_newton_naive(transformation)
    return transformation_operator


# TODO test these transformations:
def _build_canonical_to_newton_operator(
    transformation: TransformationABC,
) -> MatrixOperator:
    return MatrixOperator(transformation, _build_c2n_array(transformation))


def _build_newton_to_canonical_operator(
    transformation: TransformationABC,
) -> MatrixOperator:
    return MatrixOperator(transformation, _build_n2c_array(transformation))


def _build_lagrange_to_canonical_operator(
    transformation: TransformationABC,
) -> OperatorABC:
    lagrange_to_newton = _build_lagrange_to_newton_operator(transformation)
    newton_to_canonical = _build_newton_to_canonical_operator(transformation)
    return newton_to_canonical @ lagrange_to_newton


def _build_canonical_to_lagrange_operator(
    transformation: TransformationABC,
) -> OperatorABC:
    newton_to_lagrange = _build_newton_to_lagrange_operator(transformation)
    canonical_to_newton = _build_canonical_to_newton_operator(transformation)
    return newton_to_lagrange @ canonical_to_newton
