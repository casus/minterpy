###############################
Transformation Operator classes
###############################

Polynomial basis transformations are representable by linear matrix transformations, which are realized as
MatrixTransformationOperator or BarycentricOperator. While the MatrixTransformationOperator is given by the global matrix,
the BarycentricOperator explores its specific recursive triangular spare structure. This allows to decompose
the global matrix into its elementary parts enabling to execute the transformations
much faster with less storage requirement.

The TransormationOperatorABC abstraction layer makes the interface uniform for all cases, by overloading the matrix
multiplication ``@`` operator. The concrete implementations for (global) MatrixOperators and (decomposed) BarycentricOperators
are provided.

.. toctree::
   :maxdepth: 2

   transformOpABC
   transformationOperators