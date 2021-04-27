"""
Module for the generating points (which provides the unisolvent nodes)
"""
from copy import deepcopy
from typing import Optional, Callable

import numpy as np

from minterpy.global_settings import INT_DTYPE, ARRAY
from minterpy.multi_index import MultiIndex
from minterpy.multi_index_tree import MultiIndexTree
from minterpy.multi_index_utils import sort_lexicographically
from minterpy.utils import gen_chebychev_2nd_order_leja_ordered
from minterpy.verification import check_domain_fit, check_type_n_values, check_values, check_shape

__all__ = ['Grid']


def _gen_unisolvent_nodes(multi_index, generating_points):
    return np.take_along_axis(generating_points, multi_index.exponents, axis=0)


def get_points_from_values(spatial_dimension: int, generating_values: np.ndarray):
    generating_points = np.tile(generating_values, (1, spatial_dimension))
    generating_points[:, ::2] *= -1
    return generating_points


DEFAULT_GRID_VAL_GEN_FCT = gen_chebychev_2nd_order_leja_ordered


def remap_indices(gen_pts_from: ARRAY, gen_pts_to: ARRAY, exponents: ARRAY) -> ARRAY:
    """ replaces the exponents such that they point to new given generating points (values)

    # TODO test
    """
    exponents_remapped = exponents.copy()  # create an independent copy!
    nr_grid_vals_n, m = gen_pts_from.shape
    for i in range(m):  # NOTE: the generating points are independent in each dimension!
        cheby_values_2n_dim = gen_pts_to[:, i]
        cheby_values_n_dim = gen_pts_from[:, i]
        indices_dim = exponents[:, i]
        for idx_old, cheby_val in enumerate(cheby_values_n_dim):
            mask = np.where(indices_dim == idx_old)
            abs_diff = np.abs(cheby_values_2n_dim - cheby_val)
            # TODO use a mutable tolerance
            zero_entries = np.isclose(abs_diff, 0.0)
            nr_zero_entries = np.sum(zero_entries)
            if nr_zero_entries == 0:
                raise ValueError(f'the value {cheby_val} does not appear in the previous generating values. '
                                 'remapping the indices not possible.')
            if nr_zero_entries > 1:
                raise ValueError(f'the given generating values are not unique. remapping the indices not possible.')
            idx_new = np.argmin(abs_diff)
            exponents_remapped[mask,i] = idx_new

    # by changing the indices the lexicographical sorting might get destroyed -> restore
    exponents_remapped = sort_lexicographically(exponents_remapped)
    return exponents_remapped


# TODO implement comparison operations based on multi index comparison operations and the generating values used
class Grid(object):
    # TODO make all attributes read only!

    _unisolvent_nodes: Optional[np.ndarray] = None

    def __init__(self, multi_index: MultiIndex,
                 generating_points: Optional[np.ndarray] = None,
                 generating_values: Optional[np.ndarray] = None):
        if not isinstance(multi_index, MultiIndex):
            raise TypeError(f'the indices must be given as {MultiIndex} class instance')
        # NOTE: the multi indices of a grid must be NOT be 'lexicographically complete in order to form a basis!
        # HOWEVER: building a MultiIndexTree requires complete indices
        self.multi_index: MultiIndex = multi_index

        if generating_points is None:
            if generating_values is None:
                generating_values = DEFAULT_GRID_VAL_GEN_FCT(multi_index.poly_degree)
            spatial_dimension = multi_index.spatial_dimension
            generating_points = get_points_from_values(spatial_dimension, generating_values)

        self.generating_values = generating_values  # perform type checks and assign degree

        check_type_n_values(generating_points)
        check_shape(generating_points, dimensionality=2)
        check_domain_fit(generating_points)
        self.generating_points: np.ndarray = generating_points
        # TODO check if values and points fit together
        # TODO redundant information.

        self._tree: Optional[MultiIndexTree] = None

    # TODO rename: name is misleading. a generator is something different in python:
    #   cf. https://wiki.python.org/moin/Generators
    @classmethod
    def from_generator(cls, multi_index: MultiIndex,
                       generating_function: Callable = DEFAULT_GRID_VAL_GEN_FCT):
        generating_values = generating_function(multi_index.poly_degree)
        return cls.from_value_set(multi_index, generating_values)

    @classmethod
    def from_value_set(cls, multi_index: MultiIndex, generating_values: np.ndarray):
        spatial_dimension = multi_index.spatial_dimension
        generating_points = get_points_from_values(spatial_dimension, generating_values)
        return cls(multi_index, generating_points, generating_values)

    @property
    def unisolvent_nodes(self):
        if self._unisolvent_nodes is None:  # lazy evaluation
            self._unisolvent_nodes = _gen_unisolvent_nodes(self.multi_index, self.generating_points)
        return self._unisolvent_nodes

    @property
    def spatial_dimension(self):
        return self.multi_index.spatial_dimension

    @property
    def generating_values(self):
        return self._generating_values

    @generating_values.setter
    def generating_values(self, values: np.ndarray):
        check_type_n_values(values)
        check_domain_fit(values.reshape(-1, 1))  # 2D
        values = values.reshape(-1)  # 1D
        nr_gen_vals = len(values)
        if nr_gen_vals == 0:
            raise ValueError('at least one generating value must be given')
        self.poly_degree = nr_gen_vals - 1
        # check if multi index and generating values fit together
        if self.multi_index.poly_degree > self.poly_degree:
            raise ValueError(f'a grid of degree {self.poly_degree} '
                             f'cannot consist of indices with degree {self.multi_index.poly_degree}')
        self._unisolvent_nodes = None  # reset the unisolvent nodes
        self._generating_values: np.ndarray = values

    @property
    def tree(self):
        if self._tree is None:  # lazy evaluation
            self._tree = MultiIndexTree(self)
        return self._tree

    def enlarge(self):  # TODO: find more meaningful name
        if self.poly_degree == 0:
            double_degree = 1
        else:
            # special property: the grid of degree n is contained in the grid of degree 2n (when using chebychev values)
            double_degree = self.poly_degree * 2
        generating_values_2n = DEFAULT_GRID_VAL_GEN_FCT(double_degree)
        generating_points_2n = get_points_from_values(self.spatial_dimension, generating_values_2n)
        # ATTENTION: the ordering of the grid values might change! remap
        generating_points_n = self.generating_points
        exponents_remapped = remap_indices(generating_points_n, generating_points_2n, self.multi_index.exponents)
        # create a new multi index instance
        multi_index_remapped = MultiIndex(exponents_remapped)
        # construct a new instance
        new_instance = self.__class__(multi_index_remapped, generating_points_2n, generating_values_2n)
        # NOTE: the unisolvent nodes stay equal!
        # NOTE: might not be initialised yet!
        # new_instance._unisolvent_nodes = self.unisolvent_nodes
        new_instance._unisolvent_nodes = self._unisolvent_nodes
        return new_instance

    def apply_func(self, func, out=None):
        # apply func to unisolvent nodes and return the func values, or store them alternatively in out
        raise NotImplementedError

    def add_points(self, exponents: np.ndarray) -> 'Grid':
        exponents = np.require(exponents, dtype=INT_DTYPE)
        check_values(exponents)
        if np.max(exponents) > self.poly_degree:
            # TODO 'enlarge' the grid, increase the degree, ATTENTION:
            raise ValueError(f'trying to add point with exponent {np.max(exponents)} '
                             f'but the grid is only of degree {self.poly_degree}')

        multi_indices_old = self.multi_index
        multi_indices_new = multi_indices_old.add_exponents(exponents)
        # ATTENTION: TODO the indices must be complete in order to build a multi index tree (e.g. for the DDS scheme)!
        # multi_indices_new = multi_indices_new.make_complete()
        if multi_indices_new is multi_indices_old:  # no changes
            return self

        # construct new:
        return self.__class__(multi_indices_new, self.generating_points, self.generating_values)

    # copying
    def __copy__(self):
        return self.__class__(self.multi_index, self.generating_points, self.generating_values)

    def __deepcopy__(self, mem):
        return self.__class__(deepcopy(self.multi_index), deepcopy(self.generating_points),
                              deepcopy(self.generating_values))
