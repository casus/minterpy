#!/usr/bin/env python

from abc import ABC, abstractmethod

__all__ = ["TransformationOperatorABC"]

from typing import Optional, Union

from minterpy.global_settings import ARRAY


class TransformationOperatorABC(ABC):
    """

    TODO "inverse" property useful?
    TODO ATTENTION: this approach is only valid for single polynomials
    """

    transformation_data = None
    _array_repr_full: Optional[ARRAY] = None

    def __init__(self, transformation: "TransformationABC", transformation_data):
        # TODO useful to store the attached bases (Polynomial classes)...
        self.transformation: "TransformationABC" = transformation
        self.transformation_data = transformation_data

    @abstractmethod
    def __matmul__(self, other: Union[ARRAY, "TransformationOperatorABC"]):
        # NOTE: input may be another transformation for chaining multiple transformation operators together
        # TODO input type checking here?
        # TODO discuss: own function for this?
        pass

    @abstractmethod
    def _get_array_repr(self) -> ARRAY:
        """function computing the full array performing the transformation ("transformation matrix")

        NOTE: the output should transform the whole basis (not only some active monomials)!
        """
        pass

    @property
    def array_repr_full(self) -> ARRAY:
        """array representation of the full transformation ("transformation matrix")

        NOTE: the output transforms the whole basis (not only the active monomials)!
        """
        if self._array_repr_full is None:  # initialise
            self._array_repr_full = self._get_array_repr()
        return self._array_repr_full

    @property
    def array_repr_sparse(self) -> ARRAY:
        """array representation of the transformation only for the active monomials"""
        array_repr = self.array_repr_full
        origin_poly = self.transformation.origin_poly
        if origin_poly.indices_are_separate:
            # select only the required columns of the transformation matrix
            # TODO: find a way to make the computation more efficient, avoid computing the "whole" transformation
            active_idxs = origin_poly.active_monomials
            array_repr = array_repr[:, active_idxs]
        return array_repr
