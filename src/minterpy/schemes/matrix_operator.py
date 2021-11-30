"""
Matrix operator class.
"""
from minterpy.core.ABC import OperatorABC
from minterpy.global_settings import ARRAY

__all__ = ["MatrixOperator"]


class MatrixOperator(OperatorABC):
    """Concrete implementation of a Operator constructed as a matrix."""

    def __matmul__(self, other):
        if isinstance(other, OperatorABC):
            # the input is another transformation
            # instead of an array return another Matrix transformation operator constructed from the matrix product
            # TODO which transformation object should be passed?
            return MatrixOperator(
                self.transformation, self.array_repr_full @ other.array_repr_full
            )

        return self.array_repr_sparse @ other

    def _get_array_repr(self) -> ARRAY:
        """Returns the matrix representation of the transformation."""
        return self.transformation_data
