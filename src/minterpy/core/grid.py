"""
Module for the generating points (which provides the unisolvent nodes)
"""
from copy import deepcopy
from typing import Callable, Optional

import numpy as np

from minterpy.global_settings import ARRAY, INT_DTYPE
from minterpy.utils import gen_chebychev_2nd_order_leja_ordered

from .multi_index import MultiIndexSet
from .tree import MultiIndexTree
from .verification import (
    check_dimensionality,
    check_domain_fit,
    check_type_n_values,
)

__all__ = ["Grid"]


def _gen_unisolvent_nodes(multi_index, generating_points):
    """
    .. todo::
        - document this function but ship it to utils first.
    """
    return np.take_along_axis(generating_points, multi_index.exponents, axis=0)


def get_points_from_values(spatial_dimension: int, generating_values: ARRAY):
    """
    .. todo::
        - document this function but ship it to utils first.
    """
    generating_points = np.tile(generating_values, (1, spatial_dimension))
    generating_points[:, ::2] *= -1
    return generating_points


DEFAULT_GRID_VAL_GEN_FCT = gen_chebychev_2nd_order_leja_ordered


# TODO implement comparison operations based on multi index comparison operations and the generating values used
class Grid:
    """Datatype for the nodes some polynomial bases are defined on.

    For a definition of these nodes (refered to as unisolvent nodes), see the mathematical introduction.

    .. todo::
        - insert a small introduction to the purpose of :class:`Grid` here.
        - refactor the exposed attributes (each needs at least a getter)
        - naming issues for ``generating_points`` and ``generating_values``
    """

    # TODO make all attributes read only!

    _unisolvent_nodes: Optional[ARRAY] = None

    def __init__(
        self,
        multi_index: MultiIndexSet,
        generating_points: Optional[ARRAY] = None,
        generating_values: Optional[ARRAY] = None,
    ):
        if not isinstance(multi_index, MultiIndexSet):
            raise TypeError(
                f"the indices must be given as {MultiIndexSet} class instance"
            )
        if len(multi_index) == 0:
            raise ValueError("MultiIndexSet must not be empty!")
        # NOTE: the multi indices of a grid must be NOT be 'lexicographically complete in order to form a basis!
        # HOWEVER: building a MultiIndexTree requires complete indices
        self.multi_index: MultiIndexSet = multi_index

        if generating_points is None:
            if generating_values is None:
                generating_values = DEFAULT_GRID_VAL_GEN_FCT(multi_index.poly_degree)
            spatial_dimension = multi_index.spatial_dimension
            generating_points = get_points_from_values(
                spatial_dimension, generating_values
            )

        self.generating_values = (
            generating_values  # perform type checks and assign degree
        )

        check_type_n_values(generating_points)
        check_dimensionality(generating_points, dimensionality=2)
        check_domain_fit(generating_points)
        self.generating_points: ARRAY = generating_points
        # TODO check if values and points fit together
        # TODO redundant information.

        self._tree: Optional[MultiIndexTree] = None

    # TODO rename: name is misleading. a generator is something different in python:
    #   cf. https://wiki.python.org/moin/Generators
    @classmethod
    def from_generator(
        cls,
        multi_index: MultiIndexSet,
        generating_function: Callable = DEFAULT_GRID_VAL_GEN_FCT,
    ):
        """
        Constructor from a factory method for the ``generating_values``.

        :param multi_index: The :class:`MultiIndexSet` this ``grid`` is based on.
        :type multi_index: MultiIndexSet

        :param generating_function: Factory method for the ``generating_values``. This functions gets a polynomial degree and returns a set of generating values of this degree.
        :type generating_function: callable

        :return: Instance of :class:`Grid` for the given input.
        :rtype: Grid


        """
        generating_values = generating_function(multi_index.poly_degree)
        return cls.from_value_set(multi_index, generating_values)

    @classmethod
    def from_value_set(cls, multi_index: MultiIndexSet, generating_values: ARRAY):
        """
        Constructor from given ``generating_values``.

        :param multi_index: The :class:`MultiIndexSet` this ``grid`` is based on.
        :type multi_index: MultiIndexSet

        :param generating_values: Generating values the :class:`Grid` instance shall be based on. The input shape needs to be one-dimensional.
        :type generating_function: np.ndarray

        :return: Instance of :class:`Grid` for the given input.
        :rtype: Grid


        """
        spatial_dimension = multi_index.spatial_dimension
        generating_points = get_points_from_values(spatial_dimension, generating_values)
        return cls(multi_index, generating_points, generating_values)

    @property
    def unisolvent_nodes(self):
        """Array of unidolvent nodes.

        For a definition of unisolvent nodes, see the mathematical introduction.

        :return: Array of the unisolvent nodes. If None were given, the output is lazily build from ``multi_index`` and ``generation_points``.
        :rtype: np.ndarray


        """
        if self._unisolvent_nodes is None:  # lazy evaluation
            self._unisolvent_nodes = _gen_unisolvent_nodes(
                self.multi_index, self.generating_points
            )
        return self._unisolvent_nodes

    @property
    def spatial_dimension(self):
        """Dimension of the domain space.

        This attribute is propagated from ``multi_index``.

        :return: The dimension of the domain space, where the polynomial will live on.
        :rtype: int

        """
        return self.multi_index.spatial_dimension

    @property
    def generating_values(self):
        """Generating values.

        A one-dimensitonal set of points, the ``generating_points`` will be build on.

        :return: Array of generating values.
        :rtype: np.ndarray
        :raise ValueError: If the given set of generating values is empty.
        :raise ValueError: If the number of points is not consistent with the polynomial degree given through ``multi_index``.

        """
        return self._generating_values

    @generating_values.setter
    def generating_values(self, values: ARRAY):
        check_type_n_values(values)
        check_domain_fit(values.reshape(-1, 1))  # 2D
        values = values.reshape(-1)  # 1D
        nr_gen_vals = len(values)
        if nr_gen_vals == 0:
            raise ValueError("at least one generating value must be given")
        self.poly_degree = nr_gen_vals - 1
        # check if multi index and generating values fit together
        if self.multi_index.poly_degree > self.poly_degree:
            raise ValueError(
                f"a grid of degree {self.poly_degree} "
                f"cannot consist of indices with degree {self.multi_index.poly_degree}"
            )
        self._unisolvent_nodes = None  # reset the unisolvent nodes
        self._generating_values: ARRAY = values

    @property
    def tree(self):
        """The used :class:`MultiIndexTree`.

        :return: The :class:`MultiIndexTree` which is connected to this :class:`Grid` instance.
        :rtype: MultiIndexTree

        .. todo::
            - is this really necessary?

        """
        if self._tree is None:  # lazy evaluation
            self._tree = MultiIndexTree(self)
        return self._tree

    def apply_func(self, func, out=None):
        """This function is not implemented yet and will raise a :class:`NotImplementedError` if called.

        Apply a given (universal) function on this :class:`Grid` instance.

        :param func: The function, which will be evaluated on the grid points.
        :type func: callable
        :raise NotImplementedError: if called, since it is not implemented yet.

        :param out: The array, where the result of the function evaluation will be stored. If given, the ``out`` array will be changed inplace, otherwise the a new one will be initialised.
        :type out: np.ndarray, optional


        .. todo::
            - implement an evaluation function for :class:`Grid` instances.
            - think about using the numpy interface for universal funcions.

        """
        # apply func to unisolvent nodes and return the func values, or store them alternatively in out
        raise NotImplementedError

    def _new_instance_if_necessary(self, multi_indices_new: MultiIndexSet) -> "Grid":
        """Constructs new grid instance only if the multi indices have changed

        :param new_indices: :class:`MultiIndexSet` instance for the ``grid``, needs to be a subset of the current ``multi_index``.
        :type new_indices: MultiIndexSet

        :return: Same :class:`Grid` instance if ``multi_index`` stays the same, otherwise new polynomial instance with the new ``multi_index``.
        :rtype: Grid
        """
        multi_indices_old = self.multi_index
        # TODO: Following MR !69, the MultiIndexSet will always be a new
        # instance, revise this for consistency.
        if multi_indices_new is multi_indices_old:
            return self
        # construct new:
        return self.__class__(
            multi_indices_new, self.generating_points, self.generating_values
        )

    def make_complete(self) -> "Grid":
        """completes the multi index within this :class:`Grid` instance.

        :return: completed :class:`Grid` instance
        :rtype: Grid

        Notes
        -----
        - This is required e.g. for building a multi index tree (DDS scheme)!


        """
        multi_indices_new = self.multi_index.make_complete(inplace=False)
        return self._new_instance_if_necessary(multi_indices_new)

    def add_points(self, exponents: ARRAY) -> "Grid":
        """Extend ``grid`` and ``multi_index``

        Adds points ``grid`` and exponents to ``multi_index`` related to a given set of additional exponents.

        :param exponents: Array of exponents added.
        :type exponents: np.ndarray

        :return: New ``grid`` with the added exponents.
        :rtype: Grid

        .. todo::
            - this is boilerplate, since similar code appears in :class:`MultivariatePolynomialSingleABC`.
        """
        exponents = np.require(exponents, dtype=INT_DTYPE)
        if np.max(exponents) > self.poly_degree:
            # TODO 'enlarge' the grid, increase the degree, ATTENTION:
            raise ValueError(
                f"trying to add point with exponent {np.max(exponents)} "
                f"but the grid is only of degree {self.poly_degree}"
            )

        multi_indices_new = self.multi_index.add_exponents(exponents)
        return self._new_instance_if_necessary(multi_indices_new)

    # copying
    def __copy__(self):
        """Creates of a shallow copy.

        This function is called, if one uses the top-level function ``copy()`` on an instance of this class.

        :return: The copy of the current instance.
        :rtype: Grid

        See Also
        --------
        copy.copy
            copy operator form the python standard library.
        """
        return self.__class__(
            self.multi_index, self.generating_points, self.generating_values
        )

    def __deepcopy__(self, mem):
        """Creates of a deepcopy.

        This function is called, if one uses the top-level function ``deepcopy()`` on an instance of this class.

        :return: The deepcopy of the current instance.
        :rtype: Grid

        See Also
        --------
        copy.deepcopy
            copy operator form the python standard library.

        """
        return self.__class__(
            deepcopy(self.multi_index),
            deepcopy(self.generating_points),
            deepcopy(self.generating_values),
        )
