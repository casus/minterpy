"""
Concrete implementations for Operator classes.

The following implementations are provided:

- MatrixOperator
- BarycentricDictOperator
- BarycentricFactorisedOperator
- BarycentricPiecewiseOperator

"""

from abc import abstractmethod
from typing import Optional, Union

import numpy as np
from _warnings import warn

from minterpy.core.ABC import OperatorABC
from minterpy.global_settings import ARRAY, FLOAT_DTYPE

from ..matrix_operator import MatrixOperator
from .conversion import merge_trafo_dict, merge_trafo_factorised, merge_trafo_piecewise
from .transformation_fcts import (
    transform_barycentric_dict,
    transform_barycentric_factorised,
    transform_barycentric_piecewise,
)


class BarycentricOperator(OperatorABC):
    """Base class for the barycentric transformation operators.
       Specialized versions (dict, factorised, and piecewise) are derived from this base class.
    """

    _array_representation: Optional[ARRAY] = None

    @property
    def array_representation(self) -> ARRAY:
        """Reconstructs the global transformation matrix."""
        if self._array_representation is None:
            warn(
                "building a full transformation matrix from a barycentric transformation. this is inefficient."
            )
            # NOTE: 'self' arg must not be passed to the the merging fcts (@staticmethod)
            full_array = self.__class__.merging_fct(*self.transformation_data)  # type: ignore
            self._array_representation = full_array
        return self._array_representation

    @staticmethod
    @abstractmethod
    def transformation_fct(coeffs_in, coeffs_out_placeholder, *args):
        """Abstract method for executing the potentially decomposed linear transformation."""
        pass

    @staticmethod
    @abstractmethod
    def merging_fct(*args):
        """Abstract method for reconstructing the global matrix from the decomposition."""
        pass

    def __matmul__(
        self, other: Union[OperatorABC, ARRAY]
    ) -> Union[MatrixOperator, ARRAY]:
        """Applies the transformation operator on the input."""
        if isinstance(other, OperatorABC):
            # the input is another transformation
            # workaround: return matrix operator constructed from the matrix product
            # TODO support this natively
            # TODO which transformation object should be passed?
            return MatrixOperator(
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
        )  # type: ignore
        return coeffs_out_placeholder

    def _get_array_repr(self):
        """Reconstructs the global transformation matrix."""
        return self.array_representation


class BarycentricDictOperator(BarycentricOperator):
    """Concrete implementation of the BarycentricOperator given by the edge case given by decomposition to the 1D
    atomic sub-problems.
    """

    transformation_fct = transform_barycentric_dict
    merging_fct = merge_trafo_dict


class BarycentricFactorisedOperator(BarycentricOperator):
    """Concrete implementation of the BarycentricOperator given by the edge case given by realizing the factorised
    copied of the basic 1D atomic sub-problem.
    """

    transformation_fct = transform_barycentric_factorised
    merging_fct = merge_trafo_factorised


class BarycentricPiecewiseOperator(BarycentricOperator):
    """Concrete implementation of the BarycentricOperator used to make comparisons with the global matrix possible by
    zooming into corresponding submatrix.
    """

    transformation_fct = transform_barycentric_piecewise
    merging_fct = merge_trafo_piecewise
