#!/usr/bin/env python
""" functions required for performing (precomputed) barycentric transformations
    and for converting different transformation formats

utilises the special properties of the transformations to compute and store them in a very compact format.
this can be done very efficiently, enabling transformations for very large (e.g. high dimensional) problems.

special property:
the full transformation matrices are of nested lower triangular form and hence sparse.
the smallest possible triangular matrix pieces are determined by the leaves of the multi index tree.
each combination of 2 leaves corresponds to such an "atomic" matrix piece (some of them are all 0).
the largest of these pieces corresponds to the first node (has size n = polynomial degree).
additionally all the atomic pieces are just multiples of each other (and with different size).
this allows a very efficient computation and compact storage of the transformations:
    - solve the largest leaf sub-problem (1D)
    - compute all factors for the leaf node combinations

in the following, this compact format is called "factorised"
a factorised transformation can be stored as just numpy arrays!

TODO additional idea for optimising the transformations:
make use of the nested factorised format
precompute intermediary results of transforming each vector slice (matrix multiplication)
during a transformation only once.
then just use multiples of these results instead of performing actual matrix multiplications
"""
__author__ = "Jannik Michelfeit"
__copyright__ = "Copyright 2021, minterpy"
__credits__ = ["Jannik Michelfeit"]
# __license__ =
# __version__ =
# __maintainer__ =
__email__ = "jannik@michelfe.it"
__status__ = "Development"

import numpy as np
from numba import njit

from minterpy.global_settings import ARRAY, TYPED_LIST, INT_DTYPE, FLOAT_DTYPE, TRAFO_DICT


# functions for performing the precomputed transformations:

@njit(cache=True)
def transform_barycentric_dict(coeffs_in: ARRAY, coeffs_out: ARRAY, trafo_dict: TRAFO_DICT,
                               leaf_positions: ARRAY) -> None:
    """ performs a "piecewise" transformation (barycentric)

    TODO
    version using a dictionary encoding the transformation (= a triangular array piece for every leaf node combination)
    NOTE: this format includes a lot of redundancies,
        because the matrix pieces are actually just multiples of each other!

    NOTE: depending on the problem size it might be more performant
        to use a different implementation of this transformation!
        (e.g. regular DDS or leaf level DDS (factorised format)

    transforms and sums up the respective parts (slices) of the coefficients
    """
    for (leaf_idx_l, leaf_idx_r), matrix_piece, in trafo_dict.items():
        start_pos_in = leaf_positions[leaf_idx_l]
        start_pos_out = leaf_positions[leaf_idx_r]

        # NOTE: the size of the required slices of the coefficient vectors
        # are implicitly encoded in the size of each transformation matrix piece!
        size_out, size_in = matrix_piece.shape

        end_pos_in = start_pos_in + size_in
        slice_in = coeffs_in[start_pos_in:end_pos_in]
        # TODO exploit property that all matrix pieces are also triangular! -> avoid unnecessary computations!
        # TODO build own function using numpy.triu_indices_from?
        coeff_slice_transformed = matrix_piece @ slice_in

        end_pos_out = start_pos_out + size_out
        slice_out = coeffs_out[start_pos_out:end_pos_out]
        slice_out[:] += coeff_slice_transformed  # sum up


#  TODO void(F_1D, F_1D, F_2D, F_2D, F_1D, F_1D),
@njit(cache=True)
def transform_barycentric_factorised(coeffs_in: ARRAY, coeffs_out_placeholder: ARRAY, first_leaf_solution: ARRAY,
                                     leaf_factors: ARRAY, leaf_positions: ARRAY, leaf_sizes: ARRAY) -> None:
    """ performs a "piecewise" barycentric transformation

    # TODO benchmark against the transformation in the piecewise storage format. memory and time requirements?

    uses an optimised format of storing the transformations:
        just based on a factor for each combination of leaf problems and a single solution 1D problem

    transform and sum up the respective parts (slices) of the coefficients

    :param leaf_factors: square array of lower triangular form containing a factor for each combination of leaf nodes.
    :param first_leaf_solution: the solution of the 1D sub-problem (leaf) of maximal size
    :param coeffs_in: the Lagrange coefficients to be transformed
    :param coeffs_out_placeholder: a placeholder for the output coefficients
        NOTE: must be initialised to all 0 and have the equal size as the input coefficients
    :return: None. the output placeholder will hold the result
    """
    nr_of_leaves = len(leaf_positions)
    for node_idx_1 in range(nr_of_leaves):
        # "lower triangular form"
        for node_idx_2 in range(node_idx_1, nr_of_leaves):
            corr_factor = leaf_factors[node_idx_2, node_idx_1]
            if corr_factor == 0.0:
                continue
            start_pos_in = leaf_positions[node_idx_1]
            start_pos_out = leaf_positions[node_idx_2]

            size_in = leaf_sizes[node_idx_1]
            size_out = leaf_sizes[node_idx_2]

            end_pos_in = start_pos_in + size_in
            slice_in = coeffs_in[start_pos_in:end_pos_in]
            # TODO exploit property that all matrix pieces are also triangular! -> avoid unnecessary computations!
            # TODO build own function using numpy.triu_indices_from ?
            # TODO np.dot() is faster on contiguous arrays. by slicing contiguity is lost.
            #  is it faster to actually store all matrix pieces?!
            #  on the other hand in this implementation all input arguments are numpy arrays
            #  -> efficient JIT compilation possible!
            matrix_piece = first_leaf_solution[:size_out, :size_in]
            coeff_slice_transformed = matrix_piece @ slice_in
            coeff_slice_transformed *= corr_factor

            end_pos_out = start_pos_out + size_out
            slice_out = coeffs_out_placeholder[start_pos_out:end_pos_out]
            slice_out[:] += coeff_slice_transformed  # sum up


@njit(cache=True)
def transform_barycentric_piecewise(coeffs_in: ARRAY, coeffs_out: ARRAY, matrix_pieces: TYPED_LIST,
                                    start_positions_in: ARRAY, start_positions_out: ARRAY) -> None:
    """ performs a "piecewise" transformation (barycentric)

    unused legacy version using the explicitly computed transformation matrix pieces
    NOTE: this format includes a lot of redundancies,
        because the matrix pieces are actually just multiples of each other!

    transform and sum up the respective parts (slices) of the coefficients
    """
    for matrix_piece, start_pos_in, start_pos_out in zip(matrix_pieces, start_positions_in, start_positions_out):
        # NOTE: the size of the required slices of the coefficient vectors
        # are implicitly encoded in the size of each transformation matrix piece!
        size_out, size_in = matrix_piece.shape

        end_pos_in = start_pos_in + size_in
        slice_in = coeffs_in[start_pos_in:end_pos_in]
        # TODO exploit property that all matrix pieces are also triangular! -> avoid unnecessary computations!
        # TODO build own function using numpy.triu_indices_from ?
        coeff_slice_transformed = matrix_piece @ slice_in

        end_pos_out = start_pos_out + size_out
        slice_out = coeffs_out[start_pos_out:end_pos_out]
        slice_out[:] += coeff_slice_transformed  # sum up


# utility functions for converting different transformation formats:

@njit(cache=True)
def merge_trafo_dict(trafo_dict, leaf_positions, leaf_sizes) -> ARRAY:
    expected_size = leaf_positions[-1] + leaf_sizes[-1]
    combined_matrix = np.zeros((expected_size, expected_size), dtype=FLOAT_DTYPE)
    for (leaf_idx_l, leaf_idx_r), matrix_piece, in trafo_dict.items():
        start_pos_in = leaf_positions[leaf_idx_l]
        start_pos_out = leaf_positions[leaf_idx_r]

        # NOTE: the size of the required slices of the coefficient vectors
        # are implicitly encoded in the size of each transformation matrix piece!
        size_out, size_in = matrix_piece.shape
        end_pos_in = start_pos_in + size_in
        end_pos_out = start_pos_out + size_out

        window = combined_matrix[start_pos_out:end_pos_out, start_pos_in:end_pos_in]
        window[:] = matrix_piece

    return combined_matrix


@njit(cache=True)
def compute_matrix_pieces(first_leaf_solution, leaf_factors, leaf_positions, leaf_sizes):
    """ computes the actual matrix pieces of a transformation in factorised format explicitly

    NOTE:  useful e.g. for merging all the pieces into a single matrix
    """
    matrix_pieces = []
    start_positions_1 = []
    start_positions_2 = []
    nr_of_leaves = len(leaf_positions)
    for node_idx_1 in range(nr_of_leaves):
        # "lower triangular form"
        for node_idx_2 in range(node_idx_1, nr_of_leaves):
            corr_factor = leaf_factors[node_idx_2, node_idx_1]
            if corr_factor == 0.0:
                continue

            size_in = leaf_sizes[node_idx_1]
            size_out = leaf_sizes[node_idx_2]

            transformation_piece = first_leaf_solution[:size_out, :size_in] * corr_factor

            matrix_pieces.append(transformation_piece)

            start_pos_in = leaf_positions[node_idx_1]
            start_pos_out = leaf_positions[node_idx_2]
            start_positions_1.append(start_pos_in)
            start_positions_2.append(start_pos_out)

    start_positions_1 = np.array(start_positions_1, dtype=INT_DTYPE)
    start_positions_2 = np.array(start_positions_2, dtype=INT_DTYPE)

    return matrix_pieces, start_positions_1, start_positions_2


@njit(cache=True)
def merge_matrix_pieces(first_leaf_solution: ARRAY, leaf_factors: ARRAY, leaf_positions: ARRAY,
                        leaf_sizes: ARRAY) -> ARRAY:
    """ creates a transformation matrix of full size from a precomputed barycentric transformation in factorised form

    used for testing the equality of the transformation matrices of both regular and barycentric computation
    TODO allow to only create a slice of the total matrix
    """
    expected_size = leaf_positions[-1] + leaf_sizes[-1]
    matrix_pieces, start_positions_in, start_positions_out = compute_matrix_pieces(first_leaf_solution, leaf_factors,
                                                                                   leaf_positions, leaf_sizes)
    combined_matrix = np.zeros((expected_size, expected_size), dtype=FLOAT_DTYPE)
    for matrix_piece, start_pos_in, start_pos_out in zip(matrix_pieces, start_positions_in, start_positions_out):
        # NOTE: the size of the required slices of the coefficient vectors
        # are implicitly encoded in the size of each transformation matrix piece!
        size_out, size_in = matrix_piece.shape
        end_pos_in = start_pos_in + size_in
        end_pos_out = start_pos_out + size_out

        window = combined_matrix[start_pos_out:end_pos_out, start_pos_in:end_pos_in]
        window[:] = matrix_piece

    return combined_matrix
