from abc import abstractmethod
from typing import Optional, Union

import numpy as np
from _warnings import warn

from minterpy.barycentric_conversion import (
    merge_trafo_dict,
    merge_trafo_factorised,
    merge_trafo_piecewise,
)
from minterpy.barycentric_transformation_fcts import (
    transform_barycentric_dict,
    transform_barycentric_factorised,
    transform_barycentric_piecewise,
)
from minterpy.global_settings import ARRAY, FLOAT_DTYPE
from minterpy.transformation_operator_abstract import TransformationOperatorABC

__all__ = ["MatrixTransformationOperator"]


class MatrixTransformationOperator(TransformationOperatorABC):
    def __matmul__(self, other):
        if isinstance(other, TransformationOperatorABC):
            # the input is another transformation
            # instead of an array return another Matrix transformation operator constructed from the matrix product
            # TODO which transformation object should be passed?
            return MatrixTransformationOperator(
                self.transformation, self.array_repr_full @ other.array_repr_full
            )

        return self.array_repr_sparse @ other

    def _get_array_repr(self) -> ARRAY:
        return self.transformation_data


class BarycentricOperatorABC(TransformationOperatorABC):
    _array_representation: Optional[ARRAY] = None

    @property
    def array_representation(self) -> ARRAY:
        if self._array_representation is None:
            warn(
                "building a full transformation matrix from a barycentric transformation. this is inefficient."
            )
            # NOTE: 'self' arg must not be passed to the the merging fcts (@staticmethod)
            full_array = self.__class__.merging_fct(*self.transformation_data)
            self._array_representation = full_array
        return self._array_representation

    @staticmethod
    @abstractmethod
    def transformation_fct(coeffs_in, coeffs_out_placeholder, *args):
        pass

    @staticmethod
    @abstractmethod
    def merging_fct(*args):
        pass

    def __matmul__(
        self, other: Union[TransformationOperatorABC, ARRAY]
    ) -> Union[MatrixTransformationOperator, ARRAY]:
        if isinstance(other, TransformationOperatorABC):
            # the input is another transformation
            # workaround: return matrix operator constructed from the matrix product
            # TODO support this natively
            # TODO which transformation object should be passed?
            return MatrixTransformationOperator(
                self.transformation, self.array_repr_full @ other.array_repr_full
            )

        # TODO support "separate multi index" transformations

        # assuming the input are coefficients which should be transformed
        coeffs_in = other  # alias
        # use an output placeholder (for an increases compatibility with Numba JIT compilation)
        # initialise the placeholder with 0
        coeffs_out_placeholder = np.zeros(coeffs_in.shape, dtype=FLOAT_DTYPE)
        # trafo_dict, leaf_positions = self.data_container

        # NOTE: 'self' arg must not be passed to the the transformation fcts (@staticmethod)
        self.__class__.transformation_fct(
            coeffs_in, coeffs_out_placeholder, *self.transformation_data
        )
        return coeffs_out_placeholder

    def _get_array_repr(self):
        return self.array_representation


class BarycentricDictOperator(BarycentricOperatorABC):
    transformation_fct = transform_barycentric_dict
    merging_fct = merge_trafo_dict


class BarycentricFactorisedOperator(BarycentricOperatorABC):
    transformation_fct = transform_barycentric_factorised
    merging_fct = merge_trafo_factorised


class BarycentricPiecewiseOperator(BarycentricOperatorABC):
    transformation_fct = transform_barycentric_piecewise
    merging_fct = merge_trafo_piecewise
