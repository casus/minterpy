__all__ = ['MultiIndexTree']

from typing import Optional

import numpy as np

from minterpy.barycentric_transformation_fcts import transform_barycentric_factorised, transform_barycentric_dict
from minterpy.barycentric_precomp import compute_l2n_dict, compute_n2l_factorised
from minterpy.dds import jit_dds, compile_splits, compile_subtree_sizes, precompute_masks, \
    compile_problem_sizes
from minterpy.global_settings import ARRAY, FLOAT_DTYPE, ARRAY_DICT
from minterpy.verification import check_type_n_values, check_shape


class MultiIndexTree:
    #  TODO prevent dynamic attribute assignment (-> safe memory)
    # __slots__ = ["multi_index", "split_positions", "subtree_sizes", "stored_masks", "generating_points"]

    def __init__(self, grid: 'Grid'):
        multi_index = grid.multi_index
        if not multi_index.is_complete:
            raise ValueError("trying to use the divided difference scheme (multi index tree) "
                             "with incomplete multi indices, "
                             "but DDS only works for complete multi indices (without 'holes').")

        self.grid = grid

        exponents = multi_index.exponents
        nr_exponents, spatial_dimension = exponents.shape
        # NOTE: the tree structure ("splitting") depends on the exponents
        # in each dimension of the sorted multi index array
        # pre-compute and store where the splits appear in the exponent array
        # this implicitly defines the "nodes" of the tree
        # TODO compute on demand? NOTE: tree is being constructed only on demand (DDS)
        # TODO reverse the dim order of all
        #  (NOTE: then the "dim_idx" will then be counter intuitive: 0 for highest dimension...)
        self.split_positions = compile_splits(exponents)
        # also store the size of all nodes = how many exponent entries belong to this split
        # in combination with the positions of all appearing splits
        # the sizes fully determine the structure of the multi index tree
        # (position and amount of children etc.)
        self.subtree_sizes = compile_subtree_sizes(nr_exponents, self.split_positions)

        self.problem_sizes = compile_problem_sizes(nr_exponents, self.split_positions, self.subtree_sizes)

        # TODO improvement: also "pre-compute" more of the recursion through the tree,
        #  avoid computing the node indices each time
        self._stored_masks: Optional[ARRAY_DICT] = None

    @property
    def multi_index(self) -> 'MultiIndex':
        return self.grid.multi_index

    @property
    def stored_masks(self) -> ARRAY_DICT:  # the intermediary results required for DDS
        # TODO remove when regular DDS functionality is no longer required (together with the dds module)
        if self._stored_masks is None:  # lazy evaluation
            # based on the splittings one can compute all required correspondences
            # between nodes in the left and the right of the tree
            # this mapping can then be used to compute the nD DDS efficiently
            exponents = self.multi_index.exponents
            self._stored_masks = precompute_masks(self.split_positions, self.subtree_sizes, exponents)
        return self._stored_masks
