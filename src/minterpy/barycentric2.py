#!/usr/bin/env python
""" TODO

"""

import numpy as np
from numba import njit
from numba.typed import List

from minterpy.dds import dds_1_dimensional, get_direct_child_idxs
from minterpy.global_settings import ARRAY, TYPED_LIST, FLOAT_DTYPE, TRAFO_DICT

__author__ = "Jannik Michelfeit"
__copyright__ = "Copyright 2021, minterpy"
__credits__ = ["Jannik Michelfeit"]
# __license__ =
# __version__ =
# __maintainer__ =
__email__ = "jannik@michelfe.it"
__status__ = "Development"


@njit(cache=True)
def compute_dds_solutions(generating_points: ARRAY, problem_sizes: TYPED_LIST) -> TYPED_LIST:
    """ performs all 1D DDS schemes required for the L2N transformation

    Returns
    -------
    a list of the 1D DDS solutions of maximal required size for each dimension
    """
    dimensionality = len(problem_sizes)  # TODO rename
    dds_solutions = List()  # use Numba typed list
    for dim_idx in range(dimensionality):
        # perform "1D" DDS! of the maximal appearing size in the current dimension!
        # NOTE: due to the lexicographical ordering the first node is always the largest
        # TODO is the problem size always equal to the degree n?
        max_problem_size = problem_sizes[dim_idx][0]
        dds_solution_max = np.eye(max_problem_size, dtype=FLOAT_DTYPE)
        gen_vals = generating_points[dim_idx]  # ATTENTION: different in each dimension!
        # TODO optimise 1D dds for diagonal input <-> output!
        dds_1_dimensional(gen_vals, dds_solution_max)  # O(n^2)
        dds_solutions.append(dds_solution_max)

    return dds_solutions


@njit(cache=True)
def expand_solution(prev_solutions: TRAFO_DICT, dds_solution_max: ARRAY, dim_idx_par: int, split_positions: TYPED_LIST,
                    subtree_sizes: TYPED_LIST, problem_sizes: TYPED_LIST) -> TRAFO_DICT:
    """ computes the DDS solution of the next lower dimension

    Parameters
    ----------
    prev_solutions: the previous DDS solution of the higher dimension, composite triangular matrix encoded by a dictionary
    dds_solution_max: 1D DDS solution of the lower dimension
    dim_idx_par: the index of the higher dimension

    Returns
    -------

    a composite triangular matrix representing the DDS solution of the next lower dimension encoded by a dictionary
    """
    expanded_solutions = dict()  # required Numba numba.typed.Dict
    splits_in_dim = split_positions[dim_idx_par]
    nr_nodes_in_dim = len(splits_in_dim)
    dim_idx_child = dim_idx_par - 1

    # for all COMBINATIONS of parent nodes (NOTE: different than in usual DDS!)
    for node_idx_par_l in range(nr_nodes_in_dim):
        first_child_idx, last_child_idx = get_direct_child_idxs(dim_idx_par, node_idx_par_l, split_positions,
                                                                subtree_sizes)
        child_idx_range_l = range(first_child_idx, last_child_idx + 1)  # ATTENTION: also include the last node!
        # for all nodes to the right including the selected left node!
        for node_idx_par_r in range(node_idx_par_l, nr_nodes_in_dim):
            first_child_idx_r, last_child_idx_r = get_direct_child_idxs(dim_idx_par, node_idx_par_r,
                                                                        split_positions,
                                                                        subtree_sizes)
            nr_children_r = last_child_idx_r - first_child_idx_r + 1

            # the solution of the parent node combination (= triangular matrix)
            # contains the the required factors for all child solutions
            factors = prev_solutions[node_idx_par_l, node_idx_par_r]

            # ATTENTION: relative and absolute indexing required
            for idx_l_rel, idx_l_abs in enumerate(child_idx_range_l):
                node_size_l = problem_sizes[dim_idx_child][idx_l_abs]
                # iterate over the child nodes of the right parent
                # NOTE: in each iteration on less node needs to be considered
                # (the factor would be 0 due to the lower triangular form)
                for idx_r_rel in range(idx_l_rel, nr_children_r):
                    idx_r_abs = first_child_idx_r + idx_r_rel
                    node_size_r = problem_sizes[dim_idx_child][idx_r_abs]
                    factor = factors[idx_r_rel, idx_l_rel]
                    # the solution for a combination of child nodes consists of
                    # the top left part of the DDS solution with the correct size
                    # multiplied by the factor of the parent
                    expanded_solution = dds_solution_max[:node_size_r, :node_size_l] * factor
                    expanded_solutions[idx_l_abs, idx_r_abs] = expanded_solution
    return expanded_solutions


@njit(cache=True)
def barycentric_dds(generating_points: ARRAY, split_positions: TYPED_LIST,
                    subtree_sizes: TYPED_LIST, problem_sizes: TYPED_LIST, stop_dim_idx: int = 0) -> TRAFO_DICT:
    """ barycentric divided difference scheme for multiple dimensions

    modified version using only the regular 1D DDS function ("fully barycentric")
    using a "top-down" approach starting from an initial 1D DDS solution
    expands the "nested" DDS solution into the full L2N transformation
    This is the core algorithm for the computation of the barycentric L2N transformation

    Parameters
    ----------
    generating_points
    split_positions
    subtree_sizes
    problem_sizes
    stop_dim_idx: the index of the dimension to stop expanding the solution at
        TODO use to determine the level of "compression".
        NOTE: dict then needs to be combined with the (remaining) 1D dds results!

    TODO dds_solutions: TYPED_LIST as input param.

    Returns
    -------

    the fully expanded nested triangular matrix encoded in a dictionary
        = a triangular array piece for every leaf node combination
    """

    if stop_dim_idx < 0:
        raise ValueError(f'the smallest possible dimension to stop is 0 (requested {stop_dim_idx + 1}).')
    dimensionality = len(split_positions)
    dim_idx_par = dimensionality - 1
    if stop_dim_idx > dim_idx_par:
        raise ValueError(f'the highest possible dimension to stop is {dimensionality} (requested {stop_dim_idx + 1}).')

    dds_solutions = compute_dds_solutions(generating_points, problem_sizes)
    dds_solution_max = dds_solutions[dim_idx_par]

    # IDEA: use a dictionary to represent composite triangular matrices
    # NOTE: the key consists of the two node indices in the current dimension each solution belongs to
    # TODO improve performance, lookup
    # start from the
    curr_solutions = {(0, 0): dds_solution_max}

    # traverse through the "tree" (mimicking recursion)
    for dim_idx_par in range(dimensionality - 1, stop_dim_idx, -1):  # starting from the highest dimension O(m)
        prev_solutions = curr_solutions
        dim_idx_child = dim_idx_par - 1
        dds_solution_max = dds_solutions[dim_idx_child]
        curr_solutions = expand_solution(prev_solutions, dds_solution_max, dim_idx_par, split_positions, subtree_sizes,
                                         problem_sizes)

    # return only the result of the last iteration
    return curr_solutions


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
