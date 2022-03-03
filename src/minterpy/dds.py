"""
Module with functions required for the divided difference scheme in multiple dimensions.

Notes
-----
The barycentric transformations provide a similar functionality (L2N transformation) but with reduced time complexity!
This module also includes the functionality for implicitly creating and traversing a "Multi Index Tree".
The tree structure is being encoded by numpy arrays for increased performance.
"""

from typing import TYPE_CHECKING, Optional, no_type_check

import numpy as np
from numba import njit
from numba.typed import List

from minterpy.global_settings import (
    ARRAY,
    ARRAY_DICT,
    INT_DTYPE,
    INT_SET,
    INT_TUPLE,
    TYPED_LIST,
)

if TYPE_CHECKING:
    from .core.tree import MultiIndexTree


def compile_splits(exponents: ARRAY) -> TYPED_LIST:
    """Identify all the split positions in a multi-index set.

    A split in a given dimension of a multi-index set is indicated
    by a change of value (a "jump") of an exponent one dimension higher.
    The split position is taken after the jump takes place.

    Notes
    -----
    - The first split position is always set to [0].
    - The last dimension contains no split and the split position
      is taken to be [0].
      In other words, one-dimensional multi-index set has no split
      with a split position [0].
    - The splits in a lower dimension includes all the splits
      in the higher dimensions.

    Parameters
    ----------
    exponents : ARRAY
        An array of exponents of a multi-index set.

    Returns
    -------
    TYPED_LIST
        A list of all the split positions for the given multi-index set.
        The length of the list is the number of spatial dimension in
        the set (or the number of columns).

    Examples
    --------
    >>> xx = np.array([
    ...      [0, 0, 0],
    ...      [1, 0, 0],
    ...      [2, 0, 0],
    ...      [0, 1, 0],
    ...      [1, 1, 0],
    ...      [0, 2, 0],
    ...      [0, 0, 1],
    ...      [1, 0, 1],
    ...      [0, 1, 1],
    ...      [1, 1, 1],
    ...      [0, 0, 2],
    ... ])
    >>> compile_splits_numpy(xx)
    ListType[array(int32, 1d, C)]([[ 0  3  5  6  8 10], [ 0  6 10], [0]])
    """

    # Initialize the list of split positions
    split_positions = List()
    # The last element has no split (split position [0])
    split_positions.append(np.array([0], dtype=INT_DTYPE))
    # NOTE: the first split position is always [0]
    prev_split = np.array([0], dtype=INT_DTYPE)

    # A split is indicated when the exponents change value over the rows
    found_splits = exponents[:-1, :] != exponents[1:, :]

    # Iterate over dimension up to the 2nd
    # and find the splits in each dimension
    for i in range(exponents.shape[1] - 1, 0, -1):
        # NOTE: Shift by one to get after the shift takes place not before.
        # np.where returns np.int64,
        # so make sure it's consistent with what Numba expects
        found_split = np.where(found_splits[:, i])[0].astype(INT_DTYPE) + 1
        # Splits in higher dimension is included in the lower dimension
        union = np.union1d(prev_split, found_split)
        prev_split = union
        split_positions.insert(0, union)

    return split_positions


def compile_subtree_sizes(nr_exponents: int, split_positions: TYPED_LIST) -> TYPED_LIST:
    """Compute the sizes of all subtrees in a multi-index tree.

    Each dimension in the multi-index tree contains one or more subtrees (or nodes).
    The size of a subtree refers to the number of the multi-index set elements
    that belongs to that subtree.

    Notes
    -----
    - The largest subtree, that contains all the dimensions, has all the elements
      of the multi-index set. This subtree is the last element of the list.
    - The term subtree and node may be used interchangeably.

    Parameters
    ----------
    nr_exponents : int
        The number of elements in the corresponding multi-index set.
    split_positions: TYPED_LIST
        The list of all the split positions in the corresponding
        multi-index set.

    Returns
    -------
    TYPED_LIST
        A list of all the subtree sizes for the given multi-index tree.
        The length of the list equals to the number of spatial dimension
        in the corresponding multi-index set (or the number of columns).

    Examples
    --------
    >>> xx = np.array([
    ...      [0, 0, 0],
    ...      [1, 0, 0],
    ...      [2, 0, 0],
    ...      [0, 1, 0],
    ...      [1, 1, 0],
    ...      [0, 2, 0],
    ...      [0, 0, 1],
    ...      [1, 0, 1],
    ...      [0, 1, 1],
    ...      [1, 1, 1],
    ...      [0, 0, 2],
    ... ])
    >>> split_positions = compile_splits_numpy(xx)
    >>> compile_subtree_sizes_numpy(len(xx), split_positions)
    ListType[array(int32, 1d, C)]([[3 2 1 2 2 1], [6 4 1], [11]])
    """

    # Initialize the list of subtree sizes
    subtree_sizes = List()

    # Iterate over split positions in each dimension, excluding the last
    for split_position in split_positions[:-1]:
        # NOTE: Include the total number of exponents
        # to get the size of the last subtree in this dimension
        subtree_size = np.diff(split_position, append=nr_exponents)
        subtree_sizes.append(subtree_size)

    # The last element of the list is the largest subtree
    # NOTE: Make sure the array has the correct data type
    subtree_sizes.append(np.array([nr_exponents], dtype=INT_DTYPE))

    return subtree_sizes


def compile_problem_sizes(subtree_sizes: TYPED_LIST) -> TYPED_LIST:
    """Computes the sub-problem sizes for each subtree in the multi-index tree.

    The sub-problem size of a given dimension corresponds to the number
    of subtrees (or nodes) that belong to a single subtree of the higher dimension.

    Notes
    -----
    - The sub-problem size in one-dimension is the same as the subtree size
      in one-dimension.

    Parameters
    ----------
    subtree_sizes: TYPED_LIST
        The list of all the subtree sizes for the given multi-index tree.

    Returns
    -------
    TYPED_LIST
        A list of the problem sizes for the given multi-index tree.
        The length of the list equals to the number of spatial dimension
        in the corresponding multi-index set (or the number of columns).

    Examples
    --------
    >>> xx = np.array([
    ...      [0, 0, 0],
    ...      [1, 0, 0],
    ...      [2, 0, 0],
    ...      [0, 1, 0],
    ...      [1, 1, 0],
    ...      [0, 2, 0],
    ...      [0, 0, 1],
    ...      [1, 0, 1],
    ...      [0, 1, 1],
    ...      [1, 1, 1],
    ...      [0, 0, 2],
    ... ])
    >>> nr_exponents = len(xx)
    >>> split_positions = compile_splits_numpy(xx)
    >>> subtree_sizes = compile_subtree_sizes_numpy(nr_exponents, split_positions)
    >>> compile_problem_sizes_numpy(subtree_sizes)
    ListType[array(int32, 1d, C)]([[3 2 1 2 2 1], [3 2 1], [3]])
    """

    # Initialize the list of problem sizes
    problem_sizes = List()

    # The first problem size is the first subtree size
    problem_sizes.append(subtree_sizes[0])

    subtree_size_child = subtree_sizes[0]
    # Iterate over parent (i.e., higher dimension) subtrees
    for subtree_size_parent in subtree_sizes[1:]:
        problem_size = np.digitize(
            np.cumsum(subtree_size_parent),
            np.cumsum(subtree_size_child)
        ).astype(INT_DTYPE)

        # NOTE: Include 0 at the beginning
        # to get the first problem size in this dimension
        problem_sizes.append(
            np.diff(problem_size, prepend=0)
        )

        subtree_size_child = subtree_size_parent

    return problem_sizes


@njit(cache=True)
def get_node_position(dim_idx: int, node_idx: int, split_positions: TYPED_LIST) -> int:
    """Returns the position of the initial exponent entry corresponding to this node.

    :param dim_idx: dimension
    :param node_idx: node index
    :param split_positions: list of all split positions
    :return: the position of the initial exponent entry corresponding to the node

    Notes
    -----
    Due to JIT compilation this might return unexpected results when the input indices are out of bounds!
    """
    return split_positions[dim_idx][node_idx]


@njit(cache=True)
def get_node_size(dim_idx: int, node_idx: int, subtree_sizes: TYPED_LIST) -> int:
    """Returns the size of the specified node.

    :param dim_idx: dimension
    :param node_idx: node index
    :param subtree_sizes: list of sizes for all subtrees
    :return: problem size of the specified node

    Notes
    -----
    This is equal to the amount of exponents belonging to the corresponding slice/split of the exponent matrix.
    This is equal to the amount of "leaf" nodes which belong to the subtree with the specified node as root node.
    Due to JIT compilation this might return unexpected results when the input indices are out of bounds!
    """
    return subtree_sizes[dim_idx][node_idx]


@njit(cache=True)
def get_positions(
    dim_idx: int, node_idx: int, split_positions: TYPED_LIST, subtree_sizes: TYPED_LIST
) -> INT_TUPLE:
    """Returns the range of indices in the coeffs array that corresponds to the given node index.

    :param dim_idx: dimension
    :param node_idx: node index
    :param split_positions: all split positions
    :param subtree_sizes: all sub tree sizes
    :return: the start and end positions in the coeffs array corresponding to this sub problem
    """
    pos_from = get_node_position(dim_idx, node_idx, split_positions)
    size = get_node_size(dim_idx, node_idx, subtree_sizes)
    pos_to = (
        pos_from + size
    )  # NOTE: this position is NOT included in the referenced "split". used for slicing
    return pos_from, pos_to


@njit(cache=True)
def get_child_idxs(
    dim_idx: int,
    parent_node_idx: int,
    layer_idx: int,
    split_positions: TYPED_LIST,
    subtree_sizes: TYPED_LIST,
) -> INT_TUPLE:
    """Computes the start and end indices of all nodes in the respective layer belonging to the given subtree.

    :param dim_idx: dimension
    :param parent_node_idx: index of the parent node
    :param layer_idx: the index of the layer (= dimension) to search in
    :param split_positions: all split positions
    :param subtree_sizes: all sub tree sizes
    :return: the indices of the first and last node included in the subtree of the "query root node"

    """
    pos_from, pos_to = get_positions(
        dim_idx, parent_node_idx, split_positions, subtree_sizes
    )
    splits2search = split_positions[layer_idx]
    first_child_idx, last_child_idx = (
        0,
        len(splits2search) - 1,
    )  # end at the last entry by default
    search_start = True
    for i, pos in enumerate(splits2search):
        if search_start and pos >= pos_from:
            first_child_idx = i
            search_start = False
        elif pos >= pos_to:  # search_end = not search_start
            last_child_idx = i - 1  # the previous entry is the end of the slice
            break

    return first_child_idx, last_child_idx


@njit(cache=True)
def get_direct_child_idxs(
    dim_idx: int,
    parent_node_idx: int,
    split_positions: TYPED_LIST,
    subtree_sizes: TYPED_LIST,
):
    """Computes the start and end indices of all direct child nodes.

    :param dim_idx: dimension
    :param parent_node_idx: index of the parent node
    :param split_positions: all split positions
    :param subtree_sizes: all sub tree sizes

    """
    child_layer = dim_idx - 1  # next "lower" layer
    return get_child_idxs(
        dim_idx, parent_node_idx, child_layer, split_positions, subtree_sizes
    )


@njit(cache=True)
def get_array_slice(
    dim_idx: int,
    node_idx: int,
    array: ARRAY,
    split_positions: TYPED_LIST,
    subtree_sizes: TYPED_LIST,
) -> ARRAY:
    """Returns the slice of the array corresponding to the given sub problem.

    :param dim_idx: dimension
    :param node_idx: node index
    :param array: the coefficients array
    :param split_positions: all split positions
    :param subtree_sizes: all sub tree sizes
    :return: the slice of the coeffs array that correspond to the given sub problem

    """
    pos_from, pos_to = get_positions(dim_idx, node_idx, split_positions, subtree_sizes)
    return array[pos_from:pos_to]


@njit(cache=True)
def get_node_positions(
    dim_idx: int,
    node_idx: int,
    layer_idx: int,
    split_positions: TYPED_LIST,
    subtree_sizes: TYPED_LIST,
):
    """Returns node positions for all the child nodes.

    :param dim_idx: dimension
    :param node_idx: node index
    :param layer_idx: layer index
    :param split_positions: all split positions
    :param subtree_sizes: all sub tree sizes
    :return: node positions of all the child nodes

    """
    start_idx, end_idx = get_child_idxs(
        dim_idx, node_idx, layer_idx, split_positions, subtree_sizes
    )
    # NOTE: in order to include the last node as expected while slicing, increment the last index by 1
    end_idx += 1
    splits_in_layer = split_positions[layer_idx]
    selected_splits = splits_in_layer[start_idx:end_idx]
    return selected_splits


@njit(cache=True)
def get_leaf_idxs(
    dim_idx: int, node_idx: int, split_positions: TYPED_LIST, subtree_sizes: TYPED_LIST
):
    """Returns node indices of the leaf nodes in the given sub tree.

    :param dim_idx: dimension
    :param node_idx: node index
    :param split_positions: all split positions
    :param subtree_sizes: all sub tree sizes
    :return: the node indices of leaf nodes in the given sub tree

    """
    leaf_layer = 0  # the "lowest" layer
    start_idx, end_idx = get_child_idxs(
        dim_idx, node_idx, leaf_layer, split_positions, subtree_sizes
    )
    # NOTE: in order to include the last node as expected while slicing, increment the last index by 1
    end_idx += 1
    return start_idx, end_idx


@njit(cache=True)
def get_nodes(
    dim_idx: int,
    node_idx: int,
    layer_idx: int,
    split_positions: TYPED_LIST,
    subtree_sizes: TYPED_LIST,
):
    """Returns the nodes and corresponding problem sizes.

    :param dim_idx: dimesion
    :param node_idx: node index
    :param layer_idx: layer index
    :param split_positions: all split positions
    :param subtree_sizes: all sub tree sizes
    :return: the node indices and their respective problem sizes

    """
    start_idx, end_idx = get_child_idxs(
        dim_idx, node_idx, layer_idx, split_positions, subtree_sizes
    )
    # NOTE: in order to include the last node as expected while slicing, increment the last index by 1
    end_idx += 1
    splits_in_layer = split_positions[layer_idx]
    sizes_in_layer = subtree_sizes[layer_idx]
    selected_splits = splits_in_layer[start_idx:end_idx]
    selected_sizes = sizes_in_layer[start_idx:end_idx]
    return selected_splits, selected_sizes


@njit(cache=True)
def get_leaves(
    dim_idx: int, node_idx: int, split_positions: TYPED_LIST, subtree_sizes: TYPED_LIST
):
    """Returns all the nodes and corresponding problem sizes.

    :param dim_idx: dimesion
    :param node_idx: node index
    :param split_positions: all split positions
    :param subtree_sizes: all sub tree sizes
    :return: all node indices and their respective problem sizes

    """
    leaf_layer = 0  # the "lowest" layer
    selected_splits, selected_sizes = get_nodes(
        dim_idx, node_idx, leaf_layer, split_positions, subtree_sizes
    )
    return selected_splits, selected_sizes


@njit(cache=True)
def compute_projection_mask(
    dim_idx: int,
    node_idx_left: int,
    node_idx_right: int,
    split_positions: TYPED_LIST,
    subtree_sizes: TYPED_LIST,
    exponents: ARRAY,
) -> Optional[ARRAY]:
    """Find the correspondences between exponent entries in the left and the right node.

    :param dim_idx: dimension
    :param node_idx_left: the index of the left node to be considered
    :param node_idx_right: the index of the right node to be considered
    :param split_positions: all split positions
    :param subtree_sizes: all sub tree sizes
    :param exponents: the exponents array
    :return: the indices of the entries in the left subtree split
             corresponding to all entries in the right subtree split
    """

    # NOTE: the left subtree is often much larger than the right subtrees
    #     -> avoid creating a very sparse boolean mask
    # instead find the indices of the left leaves (segmented) corresponding to all right leaves (continuous)
    # (store only the matching idxs)
    #
    # NOTE: a "left" subtree is always bigger than any subtrees to its right
    #
    # this implementation exploits another special property of complete index sets:
    #     the corresponding leaves in the left tree will be completely contained in the right subtree
    #     and due to the lexicographical ordering the first entries of the leaves will match!
    #
    # NOTE: there might be more left leaves as right leaves! -> not a 1:1 correspondence between the leaf nodes!
    #     -> correspondence check via the exponents required!

    tree_size_r = get_node_size(dim_idx, node_idx_right, subtree_sizes)
    if (
        tree_size_r == 1
    ):  # only a single leaf entry (= all 0) also always corresponds to the first entry!
        mask = np.zeros(1, dtype=INT_DTYPE)
        return mask

    tree_size_l = get_node_size(dim_idx, node_idx_left, subtree_sizes)
    # when the two problems have the same size, the correspondence is trivial (1:1) -> no modification needed
    if tree_size_l == tree_size_r:
        mask = np.zeros(0, dtype=INT_DTYPE)  # default
        return mask

    mask = np.empty(tree_size_r, dtype=INT_DTYPE)

    leaf_positions_l, leaf_sizes_l = get_leaves(
        dim_idx, node_idx_left, split_positions, subtree_sizes
    )
    leaf_positions_r, leaf_sizes_r = get_leaves(
        dim_idx, node_idx_right, split_positions, subtree_sizes
    )
    nr_of_leaves_r = len(leaf_sizes_r)
    nr_entries_matched_r = 0
    nr_entries_matched_l = 0
    leaf_nr_r = 0
    dim_idx = (
        dim_idx + 1
    )  # required for slicing the exponent vectors for comparison correctly
    leaf_pos_r = leaf_positions_r[leaf_nr_r]  # index of the first entry of this leaf
    exp_vect_r = exponents[leaf_pos_r, :dim_idx]
    for leaf_pos_l, leaf_size_l in zip(leaf_positions_l, leaf_sizes_l):
        # NOTE: some left leaf nodes do not have a correspondence in the right subtree!
        # -> check for equality of the first entry (only in the lower dimensions)
        exp_vect_l = exponents[leaf_pos_l, :dim_idx]
        if np.array_equal(exp_vect_r, exp_vect_l):  # -> these two leaf nodes match
            # link all positions in the left leaf corresponding to existing positions in the right leaf:
            leaf_size_r = leaf_sizes_r[leaf_nr_r]
            for i in range(leaf_size_r):
                mask[nr_entries_matched_r] = nr_entries_matched_l + i
                nr_entries_matched_r += 1
            leaf_nr_r += 1
            if leaf_nr_r == nr_of_leaves_r:  # found all relevant correspondences. abort
                break
            leaf_pos_r = leaf_positions_r[leaf_nr_r]
            exp_vect_r = exponents[leaf_pos_r, :dim_idx]

        nr_entries_matched_l += leaf_size_l

    return mask


@njit(cache=True)
def precompute_masks(
    split_positions: TYPED_LIST, subtree_sizes: TYPED_LIST, exponents: ARRAY
) -> ARRAY_DICT:
    """Computes and stores all required correspondences between nodes in the left and the right of the tree
    based on the given splitting.

    :param split_positions: all split positions
    :param subtree_sizes: all sub tree sizes
    :param exponents: exponents array
    :return: a dictionary of all mappings from left to right
    """
    masks = dict()  # required Numba numba.typed.Dict
    dimensionality = len(split_positions)
    # recurse through the tree
    # NOTE: no masks are required for the last dimension (1D DDS used)
    for dim_idx_par in range(
        dimensionality - 1, 0, -1
    ):  # starting from the highest dimension
        dim_idx_child = dim_idx_par - 1
        splits_in_dim = split_positions[dim_idx_par]
        nr_nodes_in_dim = len(splits_in_dim)
        for node_idx_par in range(nr_nodes_in_dim):  # all parent nodes
            first_child_idx, last_child_idx = get_direct_child_idxs(
                dim_idx_par, node_idx_par, split_positions, subtree_sizes
            )
            if first_child_idx == last_child_idx:
                # ATTENTION: left and right node must not be equal!
                continue
            for node_idx_l in range(first_child_idx, last_child_idx):
                for node_idx_r in range(
                    node_idx_l + 1, last_child_idx + 1
                ):  # ATTENTION: also include the last idx!
                    mask = compute_projection_mask(
                        dim_idx_child,
                        node_idx_l,
                        node_idx_r,
                        split_positions,
                        subtree_sizes,
                        exponents,
                    )
                    masks[(dim_idx_child, node_idx_l, node_idx_r)] = mask

    return masks


@njit(cache=True)
def dds_1_dimensional(grid_values: ARRAY, result_placeholder: ARRAY) -> None:
    """One dimensional divided difference scheme

    :param grid_values: the positions of the interpolation nodes respectively
    :param result_placeholder: 2D array initially containing the function values
    :return: the Newton coefficients corresponding to the interpolating 1D polynomial(s)

    Notes
    -----
    Works on a result placeholder. Nothing is returned.
    Also works with multiple sets of given input coefficients.

    """

    # based on:
    #     https://en.wikipedia.org/wiki/Divided_differences
    #     https://stackoverflow.com/questions/14823891/newton-s-interpolating-polynomial-python
    #
    # TODO often used with triangular input -> triangular output
    #     optimise for this use case

    c = result_placeholder  # alias
    n = len(c)
    # NOTE: the function values get split up. the grid values however do not!
    v = grid_values[:n]  # use only the first n grid values
    for i in range(1, n):
        # IMPORTANT: assign/replace values of array view (slice)!
        i_prev = i - 1
        coeff_slice = c[i:]
        val_slice = v[i:]
        val_diff = val_slice - v[i_prev]
        val_diff = np.expand_dims(val_diff, -1)
        coeff_diff = coeff_slice - c[i_prev]
        coeff_slice[:] = coeff_diff / val_diff


@no_type_check
@njit(cache=True)
def project_n_update(
    dim_idx: int,
    node_idx_l: int,
    node_idx_r: int,
    exponent_l: int,
    v_left: ARRAY,
    result_placeholder: ARRAY,
    generating_values: ARRAY,
    split_positions: TYPED_LIST,
    subtree_sizes: TYPED_LIST,
    masks: ARRAY_DICT,
) -> None:
    """Projects v_left onto v_right and computes the divided difference.

    Each split corresponds to a slice of the function values (v_left and v_right).
    Project a value vector slice belonging to a left subtree onto its correspondence of a right subtree.

    :param dim_idx: dimension
    :param node_idx_l: the index of the left node
    :param node_idx_r: the index of the right node
    :param exponent_l: exponents in the left node
    :param v_left: slice of the coeffs array on the left
    :param result_placeholder: placeholder for divided differences
    :param generating_values: generating values used to create interpolation nodes
    :param split_positions: all split positions
    :param subtree_sizes: all subtree sizes
    :param masks: a dictionary of all precomputed required correspondences between left and right nodes

    """
    idx_offset = node_idx_r - node_idx_l
    exponent_r = exponent_l + idx_offset
    # NOTE: iterate over all splits to the right (<-> "right" nodes)

    gen_val_l = generating_values[exponent_l]
    gen_val_r = generating_values[exponent_r]
    grid_val_diff = gen_val_r - gen_val_l

    mask = masks[
        (dim_idx, node_idx_l, node_idx_r)
    ]  # look up the mapping between left and right
    if (
        len(mask) > 0
    ):  # only if the mask contains entries (no mapping required otherwise)
        v_left = v_left[mask]  # project v_left onto each v_right

    # function value corresponding to the right subtree
    v_right = get_array_slice(
        dim_idx, node_idx_r, result_placeholder, split_positions, subtree_sizes
    )
    # L_2 = (L - Q_1) / Q_H
    v_right[:] = (v_right - v_left) / grid_val_diff  # replaces all values of the view


@njit(cache=True)
def jit_dds(
    result_placeholder: ARRAY,
    generating_points: ARRAY,
    split_positions: TYPED_LIST,
    subtree_sizes: TYPED_LIST,
    masks: ARRAY_DICT,
    exponents: ARRAY,
) -> None:
    """Divided difference scheme for multiple dimensions

    :param result_placeholder: 2D array where the results (=Newton coefficients) should be stored.
        Initially this array must contain the function values on the corresponding unisolvent nodes
        (= Lagrange coefficients).
        This array may contain multiple sets of coefficients (=columns of the input matrix!) at once.
    :param generating_points: generating points for the grid
    :param split_positions: all the split positions
    :param subtree_sizes: all sub tree sizes
    :param masks: a dictionary of all precomputed required correspondences between left and right nodes
    :param exponents: the exponents array

    Notes
    -----
    Iterative scheme. Works on a result placeholder. Nothing is returned.
    Finally the ```result_placeholder``` contains the Newton coefficients of the polynomial
    which has the input function values at the respective unisolvent nodes.
    which has the input function values at the respective unisolvent nodes.
    """

    # ATTENTION: the ordering of the recursive (sub-) function calls is relevant!
    #     this is because the algorithms work on the same result placeholder (<- the views of a single array)!

    dimensionality = len(split_positions)
    # traverse through the "tree" (mimicking recursion)
    # NOTE: in the last dimension the regular (1D DDS can be used)
    for dim_idx_par in range(
        dimensionality - 1, 0, -1
    ):  # starting from the highest dimension
        dim_idx_child = dim_idx_par - 1
        splits_in_dim = split_positions[dim_idx_par]
        nr_nodes_in_dim = len(splits_in_dim)
        generating_values = generating_points[:, dim_idx_par]
        for node_idx_par in range(nr_nodes_in_dim):  # for all parent nodes
            first_child_idx, last_child_idx = get_direct_child_idxs(
                dim_idx_par, node_idx_par, split_positions, subtree_sizes
            )
            if first_child_idx == last_child_idx:
                # ATTENTION: left and right node must not be equal!
                continue  # abort when there are no "neighbouring nodes"
            # each split corresponds to a slice of the function values (v_left and v_right)
            for node_idx_l in range(first_child_idx, last_child_idx):
                # function value corresponding to the left subtree
                v_left = get_array_slice(
                    dim_idx_child,
                    node_idx_l,
                    result_placeholder,
                    split_positions,
                    subtree_sizes,
                )
                pos_l = get_node_position(dim_idx_child, node_idx_l, split_positions)
                # look up the "exponent of this sub problem"
                exponent_l = exponents[
                    pos_l, dim_idx_par
                ]  # ATTENTION: from the "dimension above"
                # iterate over all splits to the right (<-> "right" nodes)
                for node_idx_r in range(
                    node_idx_l + 1, last_child_idx + 1
                ):  # ATTENTION: also include the last idx!
                    project_n_update(
                        dim_idx_child,
                        node_idx_l,
                        node_idx_r,
                        exponent_l,
                        v_left,
                        result_placeholder,
                        generating_values,
                        split_positions,
                        subtree_sizes,
                        masks,
                    )

    # compute the usual 1D DDS for ALL leaf nodes!
    dim_idx_child = 0
    splits_in_dim = split_positions[dim_idx_child]
    nr_nodes_in_dim = len(splits_in_dim)
    generating_values = generating_points[:, dim_idx_child]
    for node_idx_par in range(nr_nodes_in_dim):
        v_leaf = get_array_slice(
            dim_idx_child,
            node_idx_par,
            result_placeholder,
            split_positions,
            subtree_sizes,
        )
        dds_1_dimensional(generating_values, v_leaf)


def dds(fct_values: ARRAY, tree: "MultiIndexTree") -> ARRAY:
    """Computes the newton coefficients for the multi dimensional polynomial using divided differences.

    :param fct_values: the function values on the unisolvent nodes
    :param tree: the MultiIndex tree instance
    :return: the newton coefficients of the polynomial

    """
    # TODO type checking?!
    # check_type_n_values(fct_values)
    # check_shape(fct_values, shape=[len(tree.multi_index)])

    # NOTE: for more memory efficiency computes the results "in place"
    # initialise the placeholder with the function values (= Lagrange coefficients
    # NOTE: the DDS function expects a 2D array as input
    result_placeholder = fct_values.copy()
    if fct_values.ndim == 1:
        # ATTENTION: the DDS operates on the first dimension! -> second dimension must be 1
        result_placeholder = result_placeholder.reshape(-1, 1)
    generating_points = tree.grid.generating_points
    split_positions = tree.split_positions
    subtree_sizes = tree.subtree_sizes
    masks = tree.stored_masks
    exponents = tree.multi_index.exponents
    jit_dds(
        result_placeholder,
        generating_points,
        split_positions,
        subtree_sizes,
        masks,
        exponents,
    )
    return result_placeholder  # = Newton coefficients
