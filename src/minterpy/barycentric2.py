#!/usr/bin/env python
""" TODO

TODO also update the N2L version

"""

import numpy as np
from minterpy.dds import dds_1_dimensional, get_direct_child_idxs, get_leaf_idxs
from minterpy.global_settings import ARRAY, TYPED_LIST, INT_DTYPE, FLOAT_DTYPE
from numba import njit

__author__ = "Jannik Michelfeit"
__copyright__ = "Copyright 2021, minterpy"
__credits__ = ["Jannik Michelfeit"]
# __license__ =
# __version__ =
# __maintainer__ =
__email__ = "jannik@michelfe.it"
__status__ = "Development"


@njit(cache=True)
def get_leaf_array_slice(dim_idx: int, node_idx: int, array: ARRAY, split_positions: TYPED_LIST,
                         subtree_sizes: TYPED_LIST) -> ARRAY:
    pos_from, pos_to = get_leaf_idxs(dim_idx, node_idx, split_positions, subtree_sizes)
    return array[pos_from:pos_to]


@njit(cache=True)
def update_leaves(dim_idx: int, node_idx_l: int, node_idx_r: int, exponent_l: int,
                  v_left: ARRAY, result_placeholder: ARRAY, generating_values: ARRAY, split_positions: TYPED_LIST,
                  subtree_sizes: TYPED_LIST) -> None:
    """ projects v_left onto v_right and computes the divided difference

    modified version for a constant leaf node size of 1:
    each split corresponds to certain leaf nodes, which in turn correspond to a slice of result array
    project a result slice belonging to a left subtree onto its correspondence of a right subtree
     and update the result according to the divided difference scheme
    """
    idx_offset = node_idx_r - node_idx_l
    exponent_r = exponent_l + idx_offset
    # NOTE: iterate over all splits to the right (<-> "right" nodes)

    gen_val_l = generating_values[exponent_l]
    gen_val_r = generating_values[exponent_r]
    grid_val_diff = gen_val_r - gen_val_l

    # values corresponding to the right subtree
    v_right = get_leaf_array_slice(dim_idx, node_idx_r, result_placeholder, split_positions, subtree_sizes)

    # special property: the amount of leaves determines the match -> just use the length of the right slice
    nr_leaves_r = len(v_right)
    v_left = v_left[:nr_leaves_r]

    # L_2 = (L - Q_1) / Q_H
    v_right[:] = (v_right - v_left) / grid_val_diff  # replaces all values of the view


# @njit(cache=True)  # TODO
def leaf_node_dds(result_placeholder: ARRAY, generating_points: ARRAY, split_positions: TYPED_LIST,
                  subtree_sizes: TYPED_LIST, child_amounts:TYPED_LIST) -> None:
    """ divided difference scheme for multiple dimensions

    modified version for a constant leaf node size of 1
    This is the core algorithm for the computation of the barycentric L2N transformation

    -> perform the nD DDS for this special case directly "on the leaf level"
    this simplifies to computing a factor for every leaf node match (<-> matrix piece in the transformation)

    NOTE: only use case is calling this with the identity matrix?!
    TODO optimise for this use case

    ATTENTION: what results will be "passed on" is determined by the order of recursion of the dds!
    this depends on the subtree structure (<-> in which dimensions two leaf nodes are "related")
    this makes the simplification of the "leaf node level dds" very hard without completely copying the original dds!

    tl;dr idea description:

    from a global perspective the desired solution (Lagrange coefficients) is the identity matrix
    for each leaf (local perspective) the solution is 1 on the corresponding unisolvent nodes and 0 everywhere else!
    this property enables some simplifications

    from left to right
    FIRST push desired solution (Lagrange) of this node to all siblings to the right
     ("downwards" in the transformation matrix)
    dive each by "leaf" difference Q_H
    solution_right = (solution_right - solution_left) / (val_right - val_left)
    solution_corrected = (solution_expected - correction) / Q_H
    THEN compute solution with 1D DDS

    referencing the parts of the matrix by selecting two leaf nodes:
    node 1 corresponds to the selected input (= Lagrange coefficients) <-> "left to right"
    node 2 corresponds to the selected output (= Newton coefficients) <-> "top to bottom"

    the parts ABOVE the diagonal are all 0 -> no computations required

    for the parts of the matrix ON the diagonal (node 1 = node 2):
    the desired solution (starting value) is the identity matrix I (with the size of this leaf node)
    the previous desired solutions ("to the top") are all 0
    -> the correction of the previous nodes EACH only causes a division
    solution_corrected_0 = I
    solution_corrected = solution_corrected / Q_H
    the value selection is based on the active initial exponents of the leaf nodes

    all parts BELOW the diagonal (node 1 left of node 2 in the tree)
    the size is (size node 2, size node 1)
    the desired solution (starting value) is all 0
    the previous desired solutions ("to the top") are all 0 above the diagonal
    -> these values cause no correction. the desired solutions stay all 0
    solution_corrected = (0 - 0) / Q_H = 0
    NOTE: this corresponds to results not passing "left-right splits in the tree"
    <-> independence of sub-problems to the right (= towards the "bottom" of the matrix)
    the previous desired solutions on the diagonal are equal to the identity I
    -> cause a correction and a division
    solution_corrected = (0 - I) / Q_H =  - I / Q_H
    the previous desired solutions below the diagonal are again all 0, but now each cause a division!
    solution_corrected = solution_corrected / Q_H
    """
    # intermediary results for every dimension:
    # finally the first entry will be the triangular
    triangular_solutions = []
    # TODO explain top down bottom up

    # starting from the "scalar 1 solution" <-> identity matrix
    curr_solutions = [np.ones(1, dtype=FLOAT_DTYPE)]
    triangular_solutions.insert(0,curr_solutions)

    dimensionality = len(split_positions)
    # traverse through the "tree" (mimicking recursion)
    # NOTE: in the last dimension the regular (1D DDS can be used)
    for dim_idx_par in range(dimensionality - 1, 0, -1):  # starting from the highest dimension
        prev_solutions = curr_solutions

        dim_idx_child = dim_idx_par - 1
        splits_in_dim = split_positions[dim_idx_par]
        generating_values = generating_points[dim_idx_par]
        nr_nodes_in_dim = len(splits_in_dim)

        curr_solutions = []
        triangular_solutions.insert(0,curr_solutions)

        # perform "1D" DDS! of the maximal appearing size in the current dimension!
        node_idx_par = 0 # the first node is always the largest due to the lexicographical ordering
        max_problem_size = child_amounts[dim_idx_par][node_idx_par]
        dds_solution_max = np.eye(max_problem_size, dtype=FLOAT_DTYPE)
        if max_problem_size > 1: # DDS only required if problem is larger than 1
            gen_vals = generating_points[dim_idx_par]  # ATTENTION: different in each dimension!
            # TODO optimise 1D dds for diagonal input <-> output!
            dds_1_dimensional(gen_vals, dds_solution_max)

        for node_idx_par in range(nr_nodes_in_dim):  # for all parent nodes

            par_solution = prev_solutions[node_idx_par]
            first_child_idx, last_child_idx = get_direct_child_idxs(dim_idx_par, node_idx_par, split_positions,
                                                                    subtree_sizes)
            curr_solution = dds_solution_max
            node_size = child_amounts[dim_idx_par][node_idx_par]
            # TODO precompute the amount of (direct) children
            if node_size == 1:
                # constant sub-problem -> no DDS required
            else:

            # each split corresponds to set of leaf nodes (= 1D sub-problems)
            for exponent_l, node_idx_l in enumerate(range(first_child_idx, last_child_idx)):
                # NOTE: the exponents of the sub-problems corresponds to the order of appearance
                # values corresponding to the left subtree
                v_left = get_leaf_array_slice(dim_idx_child, node_idx_l, result_placeholder, split_positions,
                                              subtree_sizes)
                # exponent_l = exponents[dim_idx_par, pos_l]  # ATTENTION: from the "dimension above"
                # iterate over all splits to the right (<-> "right" nodes)
                for node_idx_r in range(node_idx_l + 1, last_child_idx + 1):  # ATTENTION: also include the last idx!
                    update_leaves(dim_idx_child, node_idx_l, node_idx_r, exponent_l, v_left, result_placeholder,
                                  generating_values, split_positions, subtree_sizes)

    # NOTE: not performing the 1D DDS schemes (considering leaf nodes only)


@njit(cache=True)
def compute_l2n_factorised(generating_points: ARRAY, split_positions: TYPED_LIST, subtree_sizes: TYPED_LIST):
    """ computes the L2N transformation in compact barycentric format (factorised)

    "piecewise leaf level DDS":
    solve the 1D DDS for the maximal problem size (degree) once
    all of the matrix pieces will only be parts of this solution multiplied by a factor ...
    which can be determined by performing a DDS on the leaf level

    -> exploit this property for storing the transformation in an even more compressed format!
    (and with just numpy arrays!)

    :return: all the required data structures for the transformation
    """
    leaf_sizes = subtree_sizes[0]
    max_problem_size = np.max(leaf_sizes)
    dds_solution_max = np.eye(max_problem_size, dtype=FLOAT_DTYPE)
    gen_vals_leaves = generating_points[0]  # NOTE: equal for all leaves
    dds_1_dimensional(gen_vals_leaves, dds_solution_max)

    leaf_positions = split_positions[0]
    nr_of_leaves = len(leaf_positions)
    # compute the correction factors for all leaf node combinations (recursively defined by the DDS)
    # = the first value of each triangular transformation matrix piece
    # initialise a placeholder with the expected DDS result
    # NOTE: this matrix will be triangular (1/2 of the values are 0)
    # TODO again this matrix is of nested triangular form. optimise!?
    # TODO find more memory efficient format?!
    leaf_factors = np.eye(nr_of_leaves, dtype=FLOAT_DTYPE)
    leaf_node_dds(leaf_factors, generating_points, split_positions, subtree_sizes)

    return dds_solution_max, leaf_factors, leaf_positions, leaf_sizes


# general functions used for both N2L and L2N transformations:


@njit(cache=True)
def compute_matrix_pieces(first_leaf_solution, leaf_factors, leaf_positions, leaf_sizes):
    """ computes the actual matrix pieces of a transformation matrix explicitly

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
