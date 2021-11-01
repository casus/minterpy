####################################
reStructuredText-based documentation
####################################

reStructuredText (reST) is the default markup language used by Sphinx,
the documentation generator for the ``minterpy`` documentation.
Most of the ``minterpy`` documentation is written in reST.

This section documents the particularities of working
with the reST-based documentation in the ``minterpy`` documentation.
It covers various features of Sphinx documentation generator used
in the documentation and how they should be used in the documentation.

.. seealso::

   - Code-heavy documentation such as the ones found in :doc:`Getting Started </getting-started/index>`
     and :doc:`Howto </howto/index>` guides are written as `Jupyter notebooks`_.
     The particularities of creating such documentation for the ``minterpy``
     documentation can be found in its :doc:`own section </developers/contrib-doc/contrib-doc-guidelines/ipynb-based-doc>`.
   - API (in-code) documentation, while also uses reST markup,
     have its own particularities and styling.
     Refer to the :doc:`Writing API documentation </developers/contrib-doc/contrib-doc-guidelines/api-doc>`
     section for details.

**Contents**

.. toctree::
   :maxdepth: 1

   character-width-rule
   admonitions
   bibliography
   code-examples

.. _Jupyter notebooks: https://jupyter.org/