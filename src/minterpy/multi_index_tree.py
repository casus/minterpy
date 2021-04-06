__all__ = ['MultiIndexTree']

from typing import Optional

import numpy as np

from minterpy.barycentric import transform_barycentric_factorised, compute_l2n_factorised, compute_n2l_factorised
from minterpy.barycentric2 import barycentric_dds
from minterpy.dds import dds_n_dimensional, compile_splits, compile_subtree_sizes, precompute_masks, \
    compile_child_amounts
from minterpy.global_settings import ARRAY, FLOAT_DTYPE, ARRAY_DICT
from minterpy.verification import check_type_n_values, check_shape


class MultiIndexTree:
    # prevent dynamic attribute assignment (-> safe memory) TODO
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
        # TODO reverse order of all (NOTE: then the "dim_idx" will be counter intuitive: 0 for highest dimension...)
        self.split_positions = compile_splits(exponents)
        # also store the size of all nodes = how many exponent entries belong to this split
        # in combination with the positions of all appearing splits
        # the sizes fully determine the structure of the multi index tree
        # (position and amount of children etc.)
        self.subtree_sizes = compile_subtree_sizes(nr_exponents, self.split_positions)

        child_amounts = compile_child_amounts(nr_exponents, self.split_positions, self.subtree_sizes)

        x = barycentric_dds(grid.generating_points, self.split_positions, self.subtree_sizes, child_amounts)
        # TODO improvement: also "pre-compute" more of the recursion through the tree,
        #  avoid computing the node indices each time
        self._stored_masks: Optional[ARRAY_DICT] = None
        # TODO support min_dim, min_size options
        self._leaf_matches = None
        self._n2l_trafo = None
        self._l2n_trafo = None

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

    @property
    def n2l_trafo(self):
        """ the precomputed barycentric N2l transformation

        :return: all the required data structures for the transformation
        """
        if self._n2l_trafo is None:  # lazy evaluation
            exponents = self.multi_index.exponents
            generating_points = self.grid.generating_points
            unisolvent_nodes = self.grid.unisolvent_nodes
            leaf_positions = self.split_positions[0]
            leaf_sizes = self.subtree_sizes[0]
            self._n2l_trafo = compute_n2l_factorised(exponents, generating_points, unisolvent_nodes, leaf_positions,
                                                     leaf_sizes)

        return self._n2l_trafo

    @property
    def l2n_trafo(self):
        """ the precomputed barycentric L2N transformation

        :return: all the required data structures for the transformation
        """
        if self._l2n_trafo is None:  # lazy evaluation
            generating_points = self.grid.generating_points
            split_positions = self.split_positions
            subtree_sizes = self.subtree_sizes
            self._l2n_trafo = compute_l2n_factorised(generating_points, split_positions, subtree_sizes)

        return self._l2n_trafo

    def newton2lagrange(self, coeffs_newt: ARRAY) -> ARRAY:  # barycentric transformation
        check_type_n_values(coeffs_newt)
        check_shape(coeffs_newt, shape=[len(self.multi_index)])

        # use an output placeholder (for an increases compatibility with Numba JIT compilation)
        nr_coeffs = len(coeffs_newt)
        # initialise the placeholder with 0
        coeffs_lagr_placeholder = np.zeros(nr_coeffs, dtype=FLOAT_DTYPE)
        transform_barycentric_factorised(coeffs_newt, coeffs_lagr_placeholder, *self.n2l_trafo)
        return coeffs_lagr_placeholder

    def lagrange2newton(self, coeffs_lagr: ARRAY) -> ARRAY:  # barycentric transformation
        # TODO support 2D input?
        check_type_n_values(coeffs_lagr)
        check_shape(coeffs_lagr, shape=[len(self.multi_index)])

        # use an output placeholder (for an increases compatibility with Numba JIT compilation)
        nr_coeffs = len(coeffs_lagr)
        # initialise the placeholder with 0
        coeffs_newt_placeholder = np.zeros(nr_coeffs, dtype=FLOAT_DTYPE)
        transform_barycentric_factorised(coeffs_lagr, coeffs_newt_placeholder, *self.l2n_trafo)
        return coeffs_newt_placeholder

    def dds(self, coeffs_lagrange: ARRAY) -> ARRAY:
        check_type_n_values(coeffs_lagrange)
        check_shape(coeffs_lagrange, shape=[len(self.multi_index)])

        # NOTE: for more memory efficiency computes the results "in place"
        # initialise the placeholder with the function values
        # NOTE: the DDS function expects a 2D array as input
        # ATTENTION: the DDS operates on the first dimension! -> second dimension must be 1
        result_placeholder = coeffs_lagrange.copy().reshape(-1, 1)
        generating_points = self.grid.generating_points
        split_positions = self.split_positions
        subtree_sizes = self.subtree_sizes
        masks = self.stored_masks
        exponents = self.multi_index.exponents
        dds_n_dimensional(result_placeholder, generating_points, split_positions, subtree_sizes,
                          masks, exponents)
        return result_placeholder  # = Newton coefficients

    def build_dds_matrix(self, lagr_coeff_matrix: ARRAY) -> ARRAY:
        check_type_n_values(lagr_coeff_matrix)
        num_monomials = len(self.multi_index)
        check_shape(lagr_coeff_matrix, shape=[num_monomials, num_monomials])
        generating_points = self.grid.generating_points
        split_positions = self.split_positions
        subtree_sizes = self.subtree_sizes
        masks = self.stored_masks
        exponents = self.multi_index.exponents
        # NOTE: for more memory efficiency computes the results "in place"
        # initialise the output coefficient matrix
        coeff_newt_matrix = lagr_coeff_matrix.copy()  # dds matrix placeholder
        # NOTE: the DDS implementation is able to handle nD array input!
        dds_n_dimensional(coeff_newt_matrix, generating_points, split_positions, subtree_sizes,
                          masks, exponents)
        return coeff_newt_matrix
