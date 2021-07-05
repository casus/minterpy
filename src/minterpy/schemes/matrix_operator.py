"""
Matrix operator class.
"""
from minterpy.global_settings import ARRAY
from minterpy.core.ABC import TransformationOperatorABC

__all__ = ["MatrixTransformationOperator"]

class MatrixTransformationOperator(TransformationOperatorABC):
    """Concrete implementation of a TransformationOperator constructed as a matrix.
    """

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
        """Returns the matrix representation of the transformation.
        """
        return self.transformation_data
