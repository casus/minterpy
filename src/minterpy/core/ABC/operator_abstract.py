"""
Abstract base class for defining transformation operators from one polynomial basis to another.

This module provides the abstract base class for polynomial basis transformation operators from which all concrete
implentations are derived.

.. todo::
    "inverse" property useful?
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from minterpy.global_settings import ARRAY

if TYPE_CHECKING:
    # https://stackoverflow.com/questions/39740632/python-type-hinting-without-cyclic-imports
    from .transformation_abstract import TransformationABC


__all__ = ["OperatorABC"]


class OperatorABC(ABC):
    """Abstract base class for transformation operators.

    All transformation operators must be derived from this base class.
    """

    # ATTENTION: this approach is only valid for single polynomials
    transformation_data = None
    _array_repr_full: ARRAY | None = None

    # TODO useful to store the attached bases (Polynomial classes)...
    def __init__(self, transformation: TransformationABC, transformation_data):
        self.transformation: TransformationABC = transformation
        self.transformation_data = transformation_data

    @abstractmethod
    def __matmul__(self, other: ARRAY | OperatorABC):
        """Applies the transformation operator on the input.

        This is a placeholder of the ABC, which is overwritten by the concrete implementation.
        """
        # NOTE: input may be another transformation for chaining multiple transformation operators together
        # TODO input type checking here?
        # TODO discuss: own function for this?
        pass

    @abstractmethod
    def _get_array_repr(self) -> ARRAY:
        """Function computing the full array performing the transformation ("transformation matrix").

        Notes
        -----
        The output should transform the whole basis (not only some active monomials)!
        """
        pass

    @property
    def array_repr_full(self) -> ARRAY:
        """Array representation of the global transformation matrix.

        :return: the matrix representation of the transformation.

        Notes
        -----
        The output transforms the whole basis (not only the active monomials)!
        """
        if self._array_repr_full is None:  # initialise
            self._array_repr_full = self._get_array_repr()
        return self._array_repr_full

    @property
    def array_repr_sparse(self) -> ARRAY:
        """Array representation of the sub-transformation matrix transforming only the active monomials.

        :return: the transformation matrix for the active monomials.

        Notes
        -----
        This is an experimental feature which is part of ongoing work.

        """
        array_repr = self.array_repr_full
        origin_poly = self.transformation.origin_poly
        if origin_poly.indices_are_separate:
            # select only the required columns of the transformation matrix
            # TODO: find a way to make the computation more efficient, avoid computing the "whole" transformation
            active_idxs = origin_poly.active_monomials
            array_repr = array_repr[:, active_idxs]
        return array_repr
