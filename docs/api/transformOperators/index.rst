###############################
Transformation Operator classes
###############################

Polynomial basis transformations are representable by linear matrix transformations, which are realized as
MatrixOperator or BarycentricOperator. While the MatrixOperator is given by the global matrix,
the BarycentricOperator explores its specific recursive triangular sparse structure. This allows to decompose
the global matrix into its elementary parts enabling to execute the transformations
much faster with less storage requirement.

The OperatorABC abstraction layer makes the interface uniform for all cases, by overloading the matrix
multiplication ``@`` operator. The BarycentricOperator provides a further abstraction for barycentric
transformations.

The concrete implementations for (global) MatrixOperators and (decomposed) BarycentricOperators
are provided.

.. toctree::
   :maxdepth: 2

   barycentricOp
   matrixTransform
   barycentricDictTransform
   barycentricFactorisedTransform
   barycentricPiecewiseTransform
