"""
This module provides the functionality required for performing (precomputed) barycentric transformations.

TODO use the most performant transformation implementation depending on
NOTE: TODO the performance of each of the different formats needs to be balanced
benchmark against each other. and select the appropriate transformation. memory and time requirements?

TODO additional idea for optimising the transformations:
make use of the nested factorised format
precompute intermediary results of transforming each vector slice (matrix multiplication)
during a transformation only once.
then just use multiples of these results instead of performing actual matrix multiplications

TODO test all different transformation formats!
"""
from typing import no_type_check

from numba import njit

from minterpy.global_settings import ARRAY, TRAFO_DICT, TYPED_LIST


@no_type_check
@njit(cache=True)
def transform_barycentric_dict(
    coeffs_in: ARRAY, coeffs_out: ARRAY, trafo_dict: TRAFO_DICT, leaf_positions: ARRAY
) -> None:
    """Transformation using a dictionary encoding (= a triangular array piece for every leaf node combination).

    :param coeffs_in: the coefficients to be transformed
    :param coeffs_out: a placeholder for the output coefficients
    :param trafo_dict: dictionary with matrix pieces
    :param leaf_positions: starting positions in the input and output coeffs arrays

    Notes
    -----
    Depending on the problem size it might be more performant
    to use a different implementation of this transformation!
    (e.g. regular DDS or leaf level DDS (factorised format)

    Transforms and sums up the respective parts (slices) of the coefficients.
    """

    # NOTE: this format includes a lot of redundancies,
    #     because the matrix pieces are actually just multiples of each other!
    for (leaf_idx_l, leaf_idx_r), matrix_piece in trafo_dict.items():
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
def transform_barycentric_factorised(
    coeffs_in: ARRAY,
    coeffs_out_placeholder: ARRAY,
    first_leaf_solution: ARRAY,
    leaf_factors: ARRAY,
    leaf_positions: ARRAY,
    leaf_sizes: ARRAY,
) -> None:
    """Transformation based on factorised format that minimises storage.

    The factorised copies of the basic 1D atomic sub-problem are assigned to each combination of leaf problems.
    By keeping the decomposition, the transformation acts on the respective parts (slices) of the coefficients.

    :param leaf_factors: square array of lower triangular form containing a factor for each combination of leaf nodes.
    :param first_leaf_solution: the solution of the 1D sub-problem (leaf) of maximal size
    :param coeffs_in: the Lagrange coefficients to be transformed
    :param coeffs_out_placeholder: a placeholder for the output coefficients
        NOTE: must be initialised to all 0 and have the equal size as the input coefficients

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
def transform_barycentric_piecewise(
    coeffs_in: ARRAY,
    coeffs_out: ARRAY,
    matrix_pieces: TYPED_LIST,
    start_positions_in: ARRAY,
    start_positions_out: ARRAY,
) -> None:
    """Transformation based on piecewise format.

    :param coeffs_in: the coefficients to be transformed.
    :param coeffs_out_placeholder: a placeholder for the output coefficients.
    :param matrix_pieces: sub transformation matrices.
    :param start_positions_in: start position for the slice in coefficients to be transformed.
    :param start_positions_out: start position for the slice in output coefficients.

    """
    for matrix_piece, start_pos_in, start_pos_out in zip(
        matrix_pieces, start_positions_in, start_positions_out
    ):
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
