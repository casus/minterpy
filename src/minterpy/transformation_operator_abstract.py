#!/usr/bin/env python

from abc import ABC, abstractmethod

__all__ = ['TransformationOperatorABC']

from typing import Union

from minterpy.global_settings import ARRAY


class TransformationOperatorABC(ABC):
    transformation_data = None

    # TODO remove generating points as input. not required! stored in polynomial!
    def __init__(self, transformation_data):
        self.transformation_data = transformation_data

    # TODO "inverse" property useful?
    # TODO useful to store the attached bases (Polynomial classes)?

    # TODO input type checking here?
    # NOTE: input may be another transformation for chaining multiple transformation operators together
    # TODO discuss: own function for this?
    @abstractmethod
    def __matmul__(self, other: Union[ARRAY, 'TransformationOperatorABC']):
        pass

    @abstractmethod
    def to_array(self) -> ARRAY:
        pass
