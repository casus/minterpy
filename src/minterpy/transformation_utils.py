#!/usr/bin/env python
""" functions for computing all required transformation matrices

TODO use barycentric transformations
"""

from warnings import warn

import numpy as np

from minterpy import TransformationABC
from minterpy.global_settings import DEBUG, FLOAT_DTYPE
from minterpy.jit_compiled_utils import compute_vandermonde_n2c
from minterpy.utils import eval_newt_polys_on


# NOTE: avoid looping over a numpy array! e.g. for j in np.arange(num_monomials):
# see: # https://stackoverflow.com/questions/10698858/built-in-range-or-numpy-arange-which-is-more-efficient


def invert_triangular(triangular_matrix: np.ndarray) -> np.ndarray:
    # FIXME: triangular inversion is not working!
    # i, j = triangular_matrix.shape  # square matrix
    # inverted_matrix = solve_triangular(triangular_matrix, np.identity(i))
    inverted_matrix = np.linalg.inv(triangular_matrix)
    return inverted_matrix


def _build_newton_to_lagrange(transformation: TransformationABC, require_invertible: bool = False) -> np.ndarray:
    """ computes the Newton to Lagrange transformation matrix

    SPECIAL PROPERTY: the evaluation of any polynomial on unisolvent nodes yields
        the Lagrange coefficients of this polynomial
        (for the Lagrange basis defined implicitly by these unisolvent nodes)
    -> evaluating every Newton polynomial (<-> the monomials forming the Newton basis) on all interpolation points,
        naturally yields the operator transforming the Newton coefficients into Lagrange coefficients

    special property: half of the values will always be 0 (lower triangular matrix).

    TODO use the barycentric transformation (only if the indices are complete!)

    NOTE: it is inefficient to compute this by inversion:
        newton_to_lagrange = inv(lagrange_to_newton) # based on DDS

    :param transformation: the Transformation instance
        with the fixed polynomial defining the unisolvent nodes to perform the transformation on
    :param require_invertible: weather or not the output matrix should be square
    :return: the transformation matrix from Newton to Lagrange basis

    """
    grid = transformation.grid
    if require_invertible:
        # NOTE: the transformation matrix needs to be square for the inversion
        # even if some of the multi indices are "inactive",
        # the unisolvent nodes (grid) need to correspond to the multi indices!
        multi_index = grid.multi_index
    else:
        # NOTE: the indices might be different from the ones used in the grid!
        # -> just some "active" Lagrange polynomials
        multi_index = transformation.multi_index

    if multi_index.is_complete:
        warn("computing a N2L transformation with complete indices by Newton evaluation. "
             "consider using barycentric transformations.")

    exponents = multi_index.exponents

    unisolvent_nodes = grid.unisolvent_nodes
    generating_points = grid.generating_points
    # NOTE: the shape of unisolvent_nodes and exponents might be different! -> non square transformation matrix
    newton_to_lagrange = eval_newt_polys_on(unisolvent_nodes, exponents, generating_points, verify_input=DEBUG,
                                            triangular=True)
    return newton_to_lagrange


def build_l2n_matrix_dds(grid):
    num_monomials = len(grid.multi_index)
    lagr_coeff_matrix = np.eye(num_monomials, dtype=FLOAT_DTYPE)
    tree = grid.tree
    lagrange_to_newton = tree.build_dds_matrix(lagr_coeff_matrix)
    return lagrange_to_newton


def _build_lagrange_to_newton(transformation: TransformationABC):
    """

    TODO use the barycentric transformation (only if the indices are complete!)

    # grid = transformation.grid
    # multi_index = transformation.multi_index
    # is_complete = multi_index.is_complete
    # if is_complete:  # the DDS scheme only works with complete multi indices ("no holes")
    #     lagrange_to_newton = _build_l2n_matrix_dds(grid)
    # else:

    :param transformation: the Transformation instance defining the unisolvent nodes to perform the transformation on
    :return: the transformation matrix from Lagrange to Newton basis.
        each column of this matrix corresponds to the Newton coefficients of the respective Lagrange polynomial
    """
    if transformation.multi_index.is_complete:
        warn("computing a L2N transformation with complete indices by Newton evaluation. "
             "consider using barycentric transformations.")

    newton_to_lagrange = _build_newton_to_lagrange(transformation, require_invertible=True)
    lagrange_to_newton = invert_triangular(newton_to_lagrange)
    if transformation.origin_poly.indices_are_separate:
        # TODO find a more performant way to compute just the required entries!
        # select only the required columns of the transformation matrix
        selection_idxs = transformation.origin_poly.index_correspondence
        lagrange_to_newton = lagrange_to_newton[:, selection_idxs]
    if DEBUG:
        # there are as many "active" Lagrange polynomials as there are index vectors
        assert lagrange_to_newton.shape[1] == len(transformation.multi_index)
    return lagrange_to_newton


def _build_canonical_to_newton(transformation):
    multi_index = transformation.grid.multi_index
    num_monomials = len(multi_index)
    V_n2c = np.ones((num_monomials, num_monomials), dtype=FLOAT_DTYPE)
    compute_vandermonde_n2c(V_n2c, transformation.grid.unisolvent_nodes,
                            multi_index.exponents)  # computes the result "in place"
    tree = transformation.grid.tree
    c2n = tree.build_dds_matrix(V_n2c)
    return c2n


def _build_newton_to_canonical(transformation: TransformationABC):
    return invert_triangular(_build_canonical_to_newton(transformation))


def _build_lagrange_to_canonical(transformation: TransformationABC):
    lagrange_to_newton = _build_lagrange_to_newton(transformation)
    newton_to_canonical = _build_newton_to_canonical(transformation)
    return np.dot(newton_to_canonical, lagrange_to_newton)


def _build_canonical_to_lagrange(transformation):
    newton_to_lagrange = _build_newton_to_lagrange(transformation)
    canonical_to_newton = _build_canonical_to_newton(transformation)
    return np.dot(newton_to_lagrange, canonical_to_newton)
