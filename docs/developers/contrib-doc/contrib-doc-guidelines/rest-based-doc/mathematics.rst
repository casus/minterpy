###########
Mathematics
###########

In the ``minterpy`` documentation, Sphinx is set to display mathematical
notations using `MathJax`_.
The MathJax library provides extensive support for LaTeX, *the* markup
language for mathematics.

Inline mathematics
##################

Inline mathematics can be written using the ``:math:`` role.
For example:

.. code-block:: rest

   :math:`A_{m,n,p} = \left\{\boldsymbol{\alpha} \in \mathbb{N}^m | \|\boldsymbol{\alpha}\|_p \leq n, m,n \in \mathbb{N}, p \geq 1 \right\}` is the multi-index set.

will be rendered as:

    :math:`A_{m,n,p} = \left\{\boldsymbol{\alpha} \in \mathbb{N}^m | \|\boldsymbol{\alpha}\|_p \leq n, m,n \in \mathbb{N}, p \geq 1 \right\}` is the multi-index set.

Mathematics blocks
##################

Mathematics blocks can be written using the ``.. math::`` directive.
For example:

.. code-block:: rest

   .. math::

      N_{\boldsymbol{\alpha}}(\boldsymbol{x}) = \prod_{i=1}^{M} \prod_{j=0}^{\alpha_i - 1} (x_i - p_{j,i}), \; \boldsymbol{\alpha} \in A


will be rendered as:

    .. math::

       N_{\boldsymbol{\alpha}}(\boldsymbol{x}) = \prod_{i=1}^{M} \prod_{j=0}^{\alpha_i - 1} (x_i - p_{j,i}), \; \boldsymbol{\alpha} \in A

Numbering and cross-referencing
###############################

A math block in a page may be numbered if they are labelled using the ``:label:`` option
within the ``.. math::`` directive.
For example:

.. code-block:: rest

    .. math::
       :label: eq:newton_polynomial_basis

        N_{\boldsymbol{\alpha}}(\boldsymbol{x}) = \prod_{i=1}^{M} \prod_{j=0}^{\alpha_i - 1} (x_i - p_{j,i}), \; \boldsymbol{\alpha} \in A


will be rendered in the page as:

    .. math::
       :label: eq:newton_polynomial_basis

        N_{\boldsymbol{\alpha}}(\boldsymbol{x}) = \prod_{i=1}^{M} \prod_{j=0}^{\alpha_i - 1} (x_i - p_{j,i}), \; \boldsymbol{\alpha} \in A

The equation can then be cross-referenced *within the same page* using
the ``:eq:`` role followed by the equation name previously assigned.

For example:

.. code-block:: rest

   The multivariate Newton polynomial is defined in :eq:`eq:newton_polynomial_basis`.

The rendered page will display the equation number as a hyperlink:

    The multivariate Newton polynomial is defined in :eq:`eq:newton_polynomial_basis`.

.. note::

   Equations are numbered consecutively within the same page.
   The equation numbering will be reset to 1 in another page as ``minterpy``
   documentation doesn't use numbered table of contents.
   Therefore, it is not straightforward to cross-reference an equation defined
   in another page.
   Use instead the nearest or the most relevant heading to the equation
   as an anchor.

Best-practice recommendations
#############################

- Use the following syntax to label an equation:

  .. code-block:: rest

     :label: `eq:equation_name`

  and replace the ``equation_name`` part with the actual name of the equation.

  .. important::

     The ``equation_name`` for the label must be unique across the document.
     Make sure there's no "duplicate warning" when building the documentation.

     If such warnings arise, use common sense to rename the equation.

- Avoid cross-referencing an equation in one page from another.
  Use, instead, the nearest or the most relevant heading to the equation
  as an anchor.
  See the guidelines of
  :ref:`section heading cross-references <developers/contrib-doc/contrib-doc-guidelines/rest-based-doc/cross-references:Section headings cross-references>`
  for details.


.. _MathJax: https://www.mathjax.org/
