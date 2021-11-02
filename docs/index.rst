##########################
``minterpy`` Documentation
##########################

Welcome!
This is the documentation for ``minterpy`` |version|, last updated on |today|.

What is ``minterpy``?
#####################

.. todo::

   This section should provide answers to the following questions:

   - What is ``minterpy``?
   - What does it do? Which problems does it solve?
   - Who is it for? Who is the intended audience?
   - Why should you use it? Include some common use cases.

The Python package ``minterpy`` is based on an optimised re-implementation of
the multivariate interpolation prototype algorithm (*MIP*) by Hecht et al.\ :footcite:`hecht2020`
and thereby provides software solutions that lift the curse of dimensionality from interpolation tasks.
While interpolation occurs as the bottleneck of most computational challenges,
``minterpy`` aims to free empirical sciences from their computational limitations.

``minterpy`` is continuously extended and improved
by adding further functionality and modules that provide novel digital solutions
to a broad field of computational challenges, including but not limited to:

 - multivariate interpolation
 - non-linear polynomial regression
 - numerical integration
 - global (black-box) optimization
 - surface level-set methods
 - non-periodic spectral partial differential equations (PDE) solvers on
   flat and complex geometries
 - machine learning regularization
 - data reconstruction
 - computational solutions in algebraic geometry

``minterpy`` is an open-source Python package that makes it easily accessible
and allows for further development and improvement by the Python community.

How the documentation is organized
==================================

.. todo::

   This section should provide cross-references to the whole documentation
   structure of ``minterpy`` each with a brief description.

How to cite
===========

.. todo::

   Provide an encouragement that if the package is used, people should provide
   an attribution, most probably to the main reference paper.
   Because there is no article in JOSS yet, then select another paper.

.. toctree::
   :maxdepth: 2

   Getting Started <getting-started/index>
   How-to <howto/index>
   Fundamentals <fundamentals/index>
   API reference <api/index>
   Developers <developers/index>
   Glossary <glossary>
   Todo list <TODO>


References
##########

.. footbibliography::