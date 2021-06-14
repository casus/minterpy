###############################
Transformation Operator classes
###############################

In the simplest case, polynomial basis transformations can be thought of as a multiplication of the coefficients
with a transformation matrix. In general, however, the transformation need not be a matrix.
The TransormationOperatorABC abstraction layer makes the interface uniform for all cases, by overloading the matrix
multiplication ``@`` operator. Additionally, concrete implementations for MatrixOperators and BarycentricOperators
are also provided.

.. toctree::
   :maxdepth: 2

   transformOpABC
