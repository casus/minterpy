#!/usr/bin/env python
""" functions required for precomputing the barycentric transformations

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

import numpy as np
from numba import njit
from numba.typed import List

from minterpy.dds import dds_1_dimensional, get_direct_child_idxs, get_leaf_idxs
from minterpy.global_settings import ARRAY, TYPED_LIST, FLOAT_DTYPE, TRAFO_DICT,DEBUG

__author__ = "Jannik Michelfeit"
__copyright__ = "Copyright 2021, minterpy"
__credits__ = ["Jannik Michelfeit"]
# __license__ =
# __version__ =
# __maintainer__ =
__email__ = "jannik@michelfe.it"
__status__ = "Development"

from minterpy.utils import eval_newt_polys_on


# functions for the L2N transformation:


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
        raise ValueError('the smallest possible dimension to stop is 0.')
    dimensionality = len(split_positions)
    dim_idx_par = dimensionality - 1
    if stop_dim_idx > dim_idx_par:
        raise ValueError('the highest possible dimension to stop is the dimensionality of the problem.')

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


@njit(cache=True)  # TODO
def leaf_node_dds(result_placeholder: ARRAY, generating_points: ARRAY, split_positions: TYPED_LIST,
                  subtree_sizes: TYPED_LIST) -> None:
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
    dimensionality = len(split_positions)
    # traverse through the "tree" (mimicking recursion)
    # NOTE: in the last dimension the regular (1D DDS can be used)
    for dim_idx_par in range(dimensionality - 1, 0, -1):  # starting from the highest dimension
        dim_idx_child = dim_idx_par - 1
        splits_in_dim = split_positions[dim_idx_par]
        nr_nodes_in_dim = len(splits_in_dim)
        generating_values = generating_points[dim_idx_par]
        for node_idx_par in range(nr_nodes_in_dim):  # for all parent nodes
            first_child_idx, last_child_idx = get_direct_child_idxs(dim_idx_par, node_idx_par, split_positions,
                                                                    subtree_sizes)
            if first_child_idx == last_child_idx:
                # ATTENTION: left and right node must not be equal!
                continue  # abort when there are no "neighbouring nodes"
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


# functions for the N2L transformation:

def compute_n2l_factorised(exponents: ARRAY, generating_points: ARRAY, unisolvent_nodes: ARRAY, leaf_positions: ARRAY,
                           leaf_sizes: ARRAY):
    """ computes the N2L transformation in factorised barycentric format

    NOTE: the Newton to Lagrange transformation is implicitly given by the Newton evaluation

    "piecewise L2N transformation":
    compute the L2N transformation for the maximal problem size (= polynomial degree) once.
    NOTE: the topmost entry of this first atomic triangular matrix will be 1.
    all of the other matrix pieces will only be parts of this solution multiplied by a factor.
    for computing the factors just "compute" the first entry of each sub matrix (again with Newton evaluation).

    NOTE: JIT compilation not possible, because newt eval fct. cannot be JIT compiled


    TODO also incorporate the latest improvements of the barycentric L2N transformation:
        evaluate just in one dimension then pass the results to all nodes in the tree "below"
        -> expand the result until the full transformation has been computed


    :return: all the required data structures for the transformation
    """
    max_problem_size = np.max(leaf_sizes)
    # compute the N2L transformation matrix piece for the first leaf node (<- is of biggest size!)
    # ATTENTION: only works for multi indices (exponents) which start with the 0 vector!
    leaf_points = unisolvent_nodes[:max_problem_size, :]
    leaf_exponents = exponents[:max_problem_size, :]
    # NOTE: the first solution piece is always quadratic ("same nodes as polys")
    first_n2l_piece = eval_newt_polys_on(leaf_points, leaf_exponents, generating_points, verify_input=DEBUG,
                                         triangular=True)

    # compute the correction factors for all leaf node combinations:
    # = the first value of each triangular transformation matrix piece
    leaf_exponents = exponents[leaf_positions, :]
    leaf_points = unisolvent_nodes[leaf_positions, :]
    # NOTE: this matrix will be triangular (1/2 of the values are 0)
    # TODO  find more memory efficient format?!
    leaf_factors = eval_newt_polys_on(leaf_points, leaf_exponents, generating_points, verify_input=DEBUG,
                                      triangular=True)

    return first_n2l_piece, leaf_factors, leaf_positions, leaf_sizes
