#################################
Polynomial Transformation classes
#################################

An abstract base class is provided as the blueprint on which every implementation of a polynomial basis transformation
class must be derived from. Concrete implementations for basis transformations between the built-in
polynomial basis (Canonical, Lagrange, and Newton) are also provided.

For most common use cases, the high-level :doc:`interface` provides a convenient way to do basis
transformations.

.. toctree::
   :maxdepth: 2

   transformationCanonical
   transformationNewton
   transformationLagrange
   transformationIdentity
   interface
