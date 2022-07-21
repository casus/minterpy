#!/usr/bin/env python
""" functions required for precomputing the barycentric trees

utilises the special properties of the trees to compute and store them in a very compact (barycentric) format.
this can be done very efficiently, enabling trees for very large (e.g. high dimensional) problems.

due to the nested (recursive) structure of the trees
there are different formats for storing (and computing) these trees
"""
from __future__ import annotations

from typing import TYPE_CHECKING, no_type_check

import numpy as np
from numba import njit
from numba.typed import List

from minterpy.dds import (
    dds_1_dimensional,
    get_direct_child_idxs,
    get_leaf_idxs,
    get_node_positions,
)
from minterpy.global_settings import (
    ARRAY,
    DEBUG,
    DICT_TRAFO_TYPE,
    FACTORISED_TRAFO_TYPE,
    FLOAT_DTYPE,
    INT_DTYPE,
    TRAFO_DICT,
    TYPED_LIST,
)
from minterpy.utils import eval_newton_monomials

from .operators import BarycentricFactorisedOperator, BarycentricOperator

if TYPE_CHECKING:
    # https://stackoverflow.com/questions/39740632/python-type-hinting-without-cyclic-imports
    from minterpy.core.ABC import TransformationABC


# functions for the L2N tree:
@njit(cache=True)
def compute_1d_dds_solutions(
    generating_points: ARRAY, problem_sizes: TYPED_LIST
) -> TYPED_LIST:
    """Performs all 1D DDS schemes of maximal required size along each dimension as required in the L2N tree.

    :param generating_points: the generating points from the Grid.
    :param problem_sizes: maximum problem size along each dimension.
    :return: a list of the 1D DDS solutions of maximal required size for each dimension

    Notes
    -----
    This is similar to the precomputation in the Newton evaluation.
    """
    dimensionality = len(problem_sizes)  # TODO rename
    dds_solutions = List()  # use Numba typed list
    for dim_idx in range(dimensionality):
        # perform "1D" DDS! of the maximal appearing size in the current dimension!
        # NOTE: due to the lexicographical ordering the first node is always the largest
        # TODO is the problem size always equal to the degree n?
        max_problem_size = problem_sizes[dim_idx][0]
        dds_solution_max = np.eye(max_problem_size, dtype=FLOAT_DTYPE)
        gen_vals = generating_points[
            :, dim_idx
        ]  # ATTENTION: different in each dimension!
        # TODO optimise 1D dds for diagonal input <-> output!
        dds_1_dimensional(gen_vals, dds_solution_max)  # O(n^2)
        dds_solutions.append(dds_solution_max)

    return dds_solutions


@no_type_check
@njit(cache=True)
def expand_solution(
    prev_solutions: TRAFO_DICT,
    dds_solution_max: ARRAY,
    dim_idx_par: int,
    split_positions: TYPED_LIST,
    subtree_sizes: TYPED_LIST,
    problem_sizes: TYPED_LIST,
) -> TRAFO_DICT:
    """Combining the precomputed 1D DDS solutions.

    :param prev_solutions: the previous DDS solution of the higher dimension
                           given as a composite triangular matrix encoded by a dictionary
    :param dds_solution_max: 1D DDS solution of the lower dimension
    :param dim_idx_par: the index of the higher dimension
    :return: a composite triangular matrix representing the DDS solution of the next lower dimension encoded by a
             dictionary

    Notes
    -----
    This is similar to the Newton evaluation using the precomputed products.
    """
    # from Jannik: computes the DDS solution of the next lower dimension

    expanded_solutions = dict()  # required Numba numba.typed.Dict
    splits_in_dim = split_positions[dim_idx_par]
    nr_nodes_in_dim = len(splits_in_dim)
    dim_idx_child = dim_idx_par - 1

    # for all COMBINATIONS of parent nodes (NOTE: different than in usual DDS!)
    for node_idx_par_l in range(nr_nodes_in_dim):
        first_child_idx, last_child_idx = get_direct_child_idxs(
            dim_idx_par, node_idx_par_l, split_positions, subtree_sizes
        )
        child_idx_range_l = range(
            first_child_idx, last_child_idx + 1
        )  # ATTENTION: also include the last node!
        # for all nodes to the right including the selected left node!
        for node_idx_par_r in range(node_idx_par_l, nr_nodes_in_dim):
            first_child_idx, last_child_idx = get_direct_child_idxs(
                dim_idx_par, node_idx_par_r, split_positions, subtree_sizes
            )
            child_idx_range_r = range(
                first_child_idx, last_child_idx + 1
            )  # ATTENTION: also include the last node!

            # nr_children_r = last_child_idx_r - first_child_idx_r + 1

            # the solution of the parent node combination (= triangular matrix)
            # contains the the required factors for all child solutions
            factors = prev_solutions[node_idx_par_l, node_idx_par_r]

            # ATTENTION: relative and absolute indexing required
            for idx_l_rel, idx_l_abs in enumerate(child_idx_range_l):
                node_size_l = problem_sizes[dim_idx_child][idx_l_abs]
                # iterate over the child nodes of the right parent
                # NOTE: in each iteration on less node needs to be considered
                # (the factor would be 0 due to the lower triangular form)
                for idx_r_rel, idx_r_abs in enumerate(child_idx_range_r):
                    node_size_r = problem_sizes[dim_idx_child][idx_r_abs]
                    factor = factors[idx_r_rel, idx_l_rel]

                    # the solution for a combination of child nodes consists of
                    # the top left part of the DDS solution with the correct size
                    # multiplied by the factor of the parent
                    expanded_solution = (
                        dds_solution_max[:node_size_r, :node_size_l] * factor
                    )
                    expanded_solutions[idx_l_abs, idx_r_abs] = expanded_solution
    return expanded_solutions


@njit(cache=True)
def compute_l2n_dict(
    generating_points: ARRAY,
    split_positions: TYPED_LIST,
    subtree_sizes: TYPED_LIST,
    problem_sizes: TYPED_LIST,
    stop_dim_idx: int = 0,
) -> DICT_TRAFO_TYPE:
    """Fully barycentric divided difference scheme for multiple dimensions.

    :param generating_points: generating points of the grid
    :param split_positions:
    :param subtree_sizes:
    :param problem_sizes:
    :param stop_dim_idx: the index of the dimension to stop expanding the solution at
    :return: the fully expanded nested triangular matrix encoded in a dictionary
              (a triangular array piece for every leaf node combination)

    Notes
    -----
    Modified version using only the regular 1D DDS function ("fully barycentric")
    using a "top-down" approach starting from an initial 1D DDS solution.
    Expands the "nested" DDS solution into the full L2N tree.
    This is the core algorithm for the computation of the barycentric L2N tree.

    """
    # TODO use to determine the level of "compression".
    #    NOTE: dict then needs to be combined with the (remaining) 1D dds results!

    # TODO dds_solutions: TYPED_LIST as input param.

    if stop_dim_idx < 0:
        raise ValueError("the smallest possible dimension to stop is 0.")
    dimensionality = len(split_positions)
    dim_idx_par = dimensionality - 1
    if stop_dim_idx > dim_idx_par:
        raise ValueError(
            "the highest possible dimension to stop is the dimensionality of the problem."
        )

    dds_solutions = compute_1d_dds_solutions(generating_points, problem_sizes)
    dds_solution_max = dds_solutions[dim_idx_par]

    # IDEA: use a dictionary to represent composite triangular matrices
    # NOTE: the key consists of the two node indices in the current dimension each solution belongs to
    # TODO improve performance, lookup
    # start from the
    curr_solutions = {(0, 0): dds_solution_max}

    # traverse through the "tree" (mimicking recursion)
    for dim_idx_par in range(
        dimensionality - 1, stop_dim_idx, -1
    ):  # starting from the highest dimension O(m)
        prev_solutions = curr_solutions
        dim_idx_child = dim_idx_par - 1
        dds_solution_max = dds_solutions[dim_idx_child]
        curr_solutions = expand_solution(
            prev_solutions,
            dds_solution_max,
            dim_idx_par,
            split_positions,
            subtree_sizes,
            problem_sizes,
        )

    # return only the result of the last iteration
    leaf_positions = split_positions[0]
    return curr_solutions, leaf_positions


@njit(cache=True)
def get_leaf_array_slice(
    dim_idx: int,
    node_idx: int,
    array: ARRAY,
    split_positions: TYPED_LIST,
    subtree_sizes: TYPED_LIST,
) -> ARRAY:
    """Get leaf slices"""
    pos_from, pos_to = get_leaf_idxs(dim_idx, node_idx, split_positions, subtree_sizes)
    return array[pos_from:pos_to]


@njit(cache=True)
def update_leaves(
    dim_idx: int,
    node_idx_left: int,
    node_idx_right: int,
    gen_val_idx_left: int,
    v_left: ARRAY,
    result_placeholder: ARRAY,
    generating_values: ARRAY,
    split_positions: TYPED_LIST,
    subtree_sizes: TYPED_LIST,
    leaf_mask: ARRAY,
) -> None:
    """Project the left split exponents onto right split exponents.

    Each split in the multi-index set corresponds to a group of leaf nodes,
    each of which corresponds to a slice of result array.
    This function projects a slice of a result array that belongs to a left
    subtree onto its correspondence in a right subtree and update the result
    according to the divided difference scheme.

    The correspondence between the left and right subtrees (of splits) are
    defined by the leaf mask.

    Parameters
    ----------
    dim_idx: int
        The current dimension index.
    node_idx_left: int
        The index of the left subtree.
    node_idx_right: int
        The index of the right subtree.
    gen_val_idx_left: int
        The index of the generating values that corresponds to the left subtree.
    v_left: ARRAY
        The values that correspond to the left subtree.
    result_placeholder: ARRAY
        The result placeholder for in-place modification.
    generating_values: ARRAY
        Generating points of the current parent dimension (`dim_idx` + 1).
    split_positions: TYPED_LIST
        All split positions.
    subtree_sizes: TYPED_LIST
        All sizes of the subtrees.
    leaf_mask: ARRAY
        The current leaf mask (i.e., correspondence between left and right
        subtrees).

    Returns
    -------
    None
        The input `result_placeholder` is modified in place to store the
        results of the projection.
    """

    # Get the number of "jumps" in grid values of the same dimension
    idx_offset = node_idx_right - node_idx_left
    gen_val_idx_right = gen_val_idx_left + idx_offset
    gen_val_left = generating_values[gen_val_idx_left]
    gen_val_right = generating_values[gen_val_idx_right]
    # Get the difference between two grid values
    grid_val_diff = gen_val_right - gen_val_left

    # Values corresponding to the right subtree
    v_right = get_leaf_array_slice(
        dim_idx, node_idx_right, result_placeholder, split_positions, subtree_sizes
    )

    # Consider only entries in the left subtree
    # that have matching correspondences with the right subtree
    v_left = v_left[leaf_mask]

    # Compute the divided difference L_2 = (L - Q_1) / Q_H
    # NOTE: Replace all values of the view
    v_right[:] = (v_right - v_left) / grid_val_diff


@njit(cache=True)
def compute_leaf_projection_mask(
    dim_idx: int,
    node_idx_left: int,
    node_idx_right: int,
    split_positions: TYPED_LIST,
    subtree_sizes: TYPED_LIST,
    exponents: ARRAY,
) -> ARRAY:
    """Compute the projection mask of the left tree w.r.t right on the splits.

    A projection mask defines the correspondence between the left and right
    subtrees in a dimension.
    A leaf projection mask does not consider all leaves (all exponents),
    but only the relevant split exponents.

    Parameters
    ----------
    dim_idx: int
        The current dimension index.
    node_idx_left: int
        The index of the left subtree.
    node_idx_right: int
        The index of the right subtree.
    split_positions: TYPED_LIST
        All split positions.
    subtree_sizes: TYPED_LIST
        All sizes of the subtrees.
    exponents: ARRAY
        The exponents array.

    Returns
    -------
    ARRAY
        The indices of the split exponent entries in the left subtree that
        correspond to all split exponent entries in the right subtree.

    Notes
    -----
    Left subtree (in terms of the number of splits) is larger or equal to
    the right. When they are equal then there is only one split node.
    """

    # Consider only the splits at the leaf layer
    leaf_layer_idx = 0

    # Right subtree (split positions and subtree size)
    split_positions_right = get_node_positions(
        dim_idx, node_idx_right, leaf_layer_idx, split_positions, subtree_sizes
    )
    subtree_size_right = len(split_positions_right)

    if subtree_size_right == 1:
        # Only a single split in the right subtree,
        # always take the first entry in the left subtree
        leaf_mask = np.zeros(1, dtype=INT_DTYPE)
        return leaf_mask

    # Left subtree (split positions and subtree size)
    split_positions_left = get_node_positions(
        dim_idx, node_idx_left, leaf_layer_idx, split_positions, subtree_sizes
    )

    # Initialize leaf mask (corresponds to the size of the right subtree)
    leaf_mask = np.empty(subtree_size_right, dtype=INT_DTYPE)

    # Initialize the split index of the right subtree
    split_idx_right = 0
    # The current split position of the right subtree in the exponents matrix
    leaf_pos_right = split_positions_right[split_idx_right]
    # Split exponent in the right subtree
    # NOTE: Make sure the current dimension is included in the array slicing
    exp_right = exponents[leaf_pos_right, : dim_idx + 1]

    # Initialize indices of the matched entries
    nr_entries_matched_right = 0
    nr_entries_matched_left = 0

    for leaf_pos_left in split_positions_left:
        # Loop over all splits in the left subtree

        # Split exponent in the left subtree
        # NOTE: Make sure the current dimension is included in the slicing
        exp_left = exponents[leaf_pos_left, : dim_idx + 1]

        if np.array_equal(exp_left, exp_right):
            # The two exponent splits match
            leaf_mask[nr_entries_matched_right] = nr_entries_matched_left
            nr_entries_matched_right += 1
            split_idx_right += 1
            if split_idx_right == subtree_size_right:
                # Found all correspondence, finish the iteration
                break
            # Update the split position and exponent of the right subtree
            leaf_pos_right = split_positions_right[split_idx_right]
            # NOTE: Make sure the current dimension is included in the slicing
            exp_right = exponents[leaf_pos_right, : dim_idx + 1]

        nr_entries_matched_left += 1

    return leaf_mask


@njit(cache=True)  # TODO
def leaf_node_dds(
    result_placeholder: ARRAY,
    generating_points: ARRAY,
    split_positions: TYPED_LIST,
    subtree_sizes: TYPED_LIST,
    exponents: ARRAY,
) -> None:
    """Compute the leaf factors by divided difference scheme on the split nodes.

    This function performs the m-dimensional divided difference scheme (DDS)
    directly "on the leaf level" and only on the split nodes (i.e., not all the
    nodes according to the multi-index set).

    The results are the factors for every matching split nodes
    between the left and right subtrees.
    These factors are used in the factorized form of the barycentric
    Lagrange-to-Newton transformation operator.

    NOTE: only use case is calling this with the identity matrix?!
    TODO: optimise for this use case

    Parameters
    ----------
    result_placeholder: ARRAY
        The result placeholder for in-place modification.
    generating_points: ARRAY
        The generating points of the grid.
    split_positions: TYPED_LIST
        All split positions.
    subtree_sizes: TYPED_LIST
        All sizes of the subtrees.
    exponents: ARRAY
        The exponents array.

    Returns
    -------
    None
        The input `result_placeholder` is modified in place to store the
        results of the divided difference scheme.

    Notes
    -----
    What results will be "passed on" is determined by the order of recursion
    of the dds!
    This depends on the subtree structure (i.e., in which dimensions that
    two leaf nodes (left and right) are "related").
    The correspondence between the left and right subtrees (nodes) are
    established via masking that is computed on-the-fly.
    """

    # Get the dimensionality of the problem
    dimensionality = len(split_positions)

    for dim_idx_par in range(dimensionality - 1, 0, -1):
        # Traverse through the multi-index tree (mimicking recursion)
        # starting from the highest dimension, exclude the first dimension
        dim_idx_child = dim_idx_par - 1
        splits_in_dim = split_positions[dim_idx_par]
        nr_nodes_in_dim = len(splits_in_dim)
        # Get the generating values (generating points in the parent dimension)
        generating_values = generating_points[:, dim_idx_par]

        for node_idx_par in range(nr_nodes_in_dim):
            # Loop over all parent nodes

            # Get the indices of the first and last child subtrees (nodes)
            first_child_idx, last_child_idx = get_direct_child_idxs(
                dim_idx_par, node_idx_par, split_positions, subtree_sizes
            )

            if first_child_idx == last_child_idx:
                # ATTENTION: left and right node must not be equal!
                # Abort when there are no "neighbouring nodes"
                continue

            # Each split corresponds to group of leaf nodes (= 1D sub-problems)
            for gen_val_idx_left, node_idx_left in enumerate(
                range(first_child_idx, last_child_idx)
            ):

                # Get the values that correspond to the left subtree (splits)
                v_left = get_leaf_array_slice(
                    dim_idx_child,
                    node_idx_left,
                    result_placeholder,
                    split_positions,
                    subtree_sizes,
                )

                for node_idx_right in range(node_idx_left + 1, last_child_idx + 1):
                    # Loop over all right subtrees (splits)
                    # ATTENTION: Also include the last subtree

                    # Get the leaf mask on the fly
                    leaf_mask = compute_leaf_projection_mask(
                        dim_idx_child,
                        node_idx_left,
                        node_idx_right,
                        split_positions,
                        subtree_sizes,
                        exponents,
                    )

                    # Update the results (project left subtree onto the right)
                    update_leaves(
                        dim_idx_child,
                        node_idx_left,
                        node_idx_right,
                        gen_val_idx_left,
                        v_left,
                        result_placeholder,
                        generating_values,
                        split_positions,
                        subtree_sizes,
                        leaf_mask,
                    )

    # NOTE: No need to perform the 1D DDS schemes (considering leaf nodes only)


@njit(cache=True)
def compute_l2n_factorised(
    generating_points: ARRAY,
    split_positions: TYPED_LIST,
    subtree_sizes: TYPED_LIST,
    exponents: ARRAY,
) -> FACTORISED_TRAFO_TYPE:
    """Compute the required data to represent factorized Lagrange-to-Newton.

    special property:
    the full tree matrices are of nested lower triangular form and hence sparse.
    the smallest possible triangular matrix pieces are determined by the leaves of the multi index tree.
    each combination of 2 leaves corresponds to such an "atomic" matrix piece (some of them are all 0).
    the largest of these pieces corresponds to the first node (has size n = polynomial degree).
    additionally all the atomic pieces are just multiples of each other (and with different size).
    this allows a very efficient computation and compact storage of the trees:
    - solve the 1D DDS for the leaf problem of maximal size once
    - compute all factors for the leaf node combinations ("leaf level DDS")

    -> exploit this property for storing the tree in an even more compressed format!
    (and with just numpy arrays!)

    in the following, this compact format is called "factorised"

    Parameters
    ----------
    generating_points: ARRAY
        The generating points of the grid.
    split_positions: ARRAY
        All split positions.
    subtree_sizes: TYPED_LIST
        All sizes of the subtrees.
    exponents: ARRAY
        The exponents array.

    Returns
    -------
    FACTORIZED_TRAFO_TYPE
        All the required data structures for the tree

    Notes
    -----
    This is considered a legacy version as it is not exploiting the full nested
    structure of the multi-dimensional DDS problem!
    """
    leaf_sizes = subtree_sizes[0]
    max_problem_size = np.max(leaf_sizes)
    dds_solution_max = np.eye(max_problem_size, dtype=FLOAT_DTYPE)
    gen_vals_leaves = generating_points[:, 0]  # NOTE: equal for all leaves
    dds_1_dimensional(gen_vals_leaves, dds_solution_max)

    leaf_positions = split_positions[0]
    nr_of_leaves = len(leaf_positions)
    # compute the correction factors for all leaf node combinations (recursively defined by the DDS)
    # = the first value of each triangular tree matrix piece
    # initialise a placeholder with the expected DDS result
    # NOTE: this matrix will be triangular (1/2 of the values are 0)
    # TODO again this matrix is of nested triangular form. optimise!?
    # TODO find more memory efficient format?!
    leaf_factors = np.eye(nr_of_leaves, dtype=FLOAT_DTYPE)
    leaf_node_dds(
        leaf_factors, generating_points, split_positions, subtree_sizes, exponents
    )

    return dds_solution_max, leaf_factors, leaf_positions, leaf_sizes


# functions for the N2L tree:


def compute_n2l_factorised(
    exponents: ARRAY,
    generating_points: ARRAY,
    unisolvent_nodes: ARRAY,
    leaf_positions: ARRAY,
    leaf_sizes: ARRAY,
) -> FACTORISED_TRAFO_TYPE:
    """computes the N2L tree in factorised barycentric format

    NOTE: the Newton to Lagrange tree is implicitly given by the Newton evaluation

    "piecewise L2N tree":
    compute the L2N tree for the maximal problem size (= polynomial degree) once.
    NOTE: the topmost entry of this first atomic triangular matrix will be 1.
    all of the other matrix pieces will only be parts of this solution multiplied by a factor.
    for computing the factors just "compute" the first entry of each sub matrix (again with Newton evaluation).

    NOTE: JIT compilation not possible, because newt eval fct. cannot be JIT compiled

    TODO also incorporate the latest improvements of the barycentric L2N tree:
        evaluate just in one dimension then pass the results to all nodes in the tree "below"
        -> expand the result until the full tree has been computed

    :return: all the required data structures for the tree
    """
    max_problem_size = np.max(leaf_sizes)
    # compute the N2L tree matrix piece for the first leaf node (<- is of biggest size!)
    # ATTENTION: only works for multi indices (exponents) which start with the 0 vector!
    leaf_points = unisolvent_nodes[:max_problem_size, :]
    leaf_exponents = exponents[:max_problem_size, :]
    # NOTE: the first solution piece is always quadratic ("same nodes as polys")
    first_n2l_piece = eval_newton_monomials(
        leaf_points,
        leaf_exponents,
        generating_points,
        verify_input=DEBUG,
        triangular=True,
    )

    # compute the correction factors for all leaf node combinations:
    # = the first value of each triangular tree matrix piece
    leaf_exponents = exponents[leaf_positions, :]
    leaf_points = unisolvent_nodes[leaf_positions, :]
    # NOTE: this matrix will be triangular (1/2 of the values are 0)
    # TODO  find more memory efficient format?!
    leaf_factors = eval_newton_monomials(
        leaf_points,
        leaf_exponents,
        generating_points,
        verify_input=DEBUG,
        triangular=True,
    )

    return first_n2l_piece, leaf_factors, leaf_positions, leaf_sizes


def _build_lagrange_to_newton_bary(
    transformation: TransformationABC,
) -> BarycentricOperator:
    """Construct the barycentric transformation operator for Lagrange-to-Newton.

    Parameters
    ----------
    transformation: TransformationABC
        An instance of one of the concrete implementation of TransformationABC

    Returns
    -------
    BarycentricOperator
        a BarycentricOperator for Lagrange to Newton transformation.
    """
    # TODO balancing: deciding on the optimal tree format to use!
    grid = transformation.grid
    tree = grid.tree
    generating_points = grid.generating_points
    split_positions = tree.split_positions
    subtree_sizes = tree.subtree_sizes

    exponents = grid.multi_index.exponents

    transformation_data = compute_l2n_factorised(
        generating_points, split_positions, subtree_sizes, exponents
    )

    transformation_operator = BarycentricFactorisedOperator(
        transformation, transformation_data
    )

    return transformation_operator


def _build_newton_to_lagrange_bary(
    transformation: TransformationABC,
) -> BarycentricFactorisedOperator:
    """Construct the barycentric transformation operator for Newton to Lagrange.

    :param transformation: an instance of one of the concrete implementation of TransformationABC
    :return: a BarycentricOperator for Newton to Lagrange transformation.

    """
    # TODO balancing: deciding on the optimal tree format to use!
    grid = transformation.grid
    multi_index = grid.multi_index
    tree = grid.tree
    generating_points = grid.generating_points
    exponents = multi_index.exponents
    unisolvent_nodes = grid.unisolvent_nodes
    leaf_positions = tree.split_positions[0]
    leaf_sizes = tree.subtree_sizes[0]
    transformation_data = compute_n2l_factorised(
        exponents, generating_points, unisolvent_nodes, leaf_positions, leaf_sizes
    )
    transformation_operator = BarycentricFactorisedOperator(
        transformation, transformation_data
    )

    return transformation_operator
