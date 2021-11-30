#############################
Writing reStructuredText Docs
#############################

reStructuredText (reST) is the default markup language used by Sphinx,
the documentation generator for the ``minterpy`` docs.
Most content of the ``minterpy`` docs is written in reST.

This section docs the particularities of working
with the reST-based docs in the ``minterpy`` documentation.
It covers various features of the Sphinx documentation generator used
in the docs and how they should be used.

.. seealso::

   - Code-heavy documentation such as the ones found in :doc:`Getting Started </getting-started/index>`
     and :doc:`/how-to/index` are written as `Jupyter notebooks`_.
     The particularities of creating such documentation for the ``minterpy``
     documentation can be found in :doc:`/contributors/contrib-docs/guidelines/ipynb-based-docs`.
   - This section does not cover every single aspect and syntax of reST,
     see `reStructuredText Primer`_ if you don't find what you need here.

..
    Page structure
    ##############
  - API (in-code) documentation, while also uses reST markup,
     have its own particularities and styling.
     Refer to the :doc:`/contributors/contrib-docs/guidelines/py-based-docs`
     section for details.

80-Character width rule
#######################

This rule might seem archaic as computer displays have no problem displaying
more than 80 characters in a single line.
However, long lines of text tend to be harder to read
and limiting a line to 80 characters does improve readability.
Besides, a lot of people prefer to work with half-screen windows.

Consider the following best-practice recommendations:

.. tip::

   - Try to maintain this rule when adding content to the documentation.
     Set a vertical ruler line at the 80-character width in your text editor of choice;
     this setting is almost universally available in any text editors.
   - Like any rules, there are exceptions. For instance, code blocks, URLs, and
     Sphinx roles and directives may extend beyond 80 characters
     (otherwise, Sphinx can't parse them).
     When in doubt, use common sense.

.. seealso::

   For additional rational on the 80-character width as well as
   where to break a line in the documentation source, see:

   - `Is the 80 character line limit still relevant`_ by Richard Dingwall
   - `Semantic Linefeeds`_ by Brandon Rhodes

Admonitions
###########

Sphinx provides support for a whole class of built-in `admonitions`_
as a set of directives to render text inside a highlighted box.

Available admonitions
=====================

There are several types of admonitions that may be used in the ``minterpy``
documentation:

.. note::

    Add an additional information that the reader may need to be aware of.

    Use the ``.. note::`` directive to create a note block.

.. important::

   Use an important block to make sure that the reader is aware of some key steps
   and what might go wrong if the reader doesn't have the provided information.

   Use the ``.. important::`` directive to create an important block.

.. warning::

   Add a warning to indicate irreversible (possibly detrimental) actions and
   known longstanding issues or limitations.

   Use the ``.. warning::`` directive to create a warning block.

.. tip::

   Use a tip block to offer best-practice or alternative workflow
   with respect to he current instructions.

   Use the ``..tip::`` directive to create a tip-block.

.. seealso::

   Use a see-also block to provide a list of cross-references
   (internal or external) if you think these cross-references must be listed
   separately to attract more attention.

   Use the ``.. seealso::`` directive to create a see-also block.

Consider the following best-practice recommendation:

.. tip::

   - Use admonitions sparingly and judiciously in the ``minterpy`` docs
     as they tend to obstruct the reading flow.
     Besides, if used too often, readers may become immune to notes and warnings
     and would simply ignore them.

Bibliographic citations
#######################

A bibliographic citation is a special case of :ref:`cross-referencing <contributors/contrib-docs/guidelines/rest-based-docs:Cross-references>`.
Specifically, it cross-references external scientific works
such as articles, books, or reports.
You need to include any relevant scientific works in the ``minterpy`` docs
if and when applicable.
This is typically the case when writing the :ref:`Fundamentals <fundamentals/index:Fundamentals of \`\`minterpy\`\`>` guide.

Bibliography file
=================

The bibliographic entries are located in the bibliography file, a `BibTeX`_ file
named ``refs.bib`` in the root ``docs`` directory.
An entry in the file is written in the standard BibTeX format.

For example, an article entry is written as follows:

.. code-block:: bibtex

   @article{Dyn2014,
        title={Multivariate polynomial interpolation on lower sets},
        author={Dyn, Nira and Floater, Michael S.},
        journal={Journal of Approximation Theory},
        volume={177},
        pages={34--42},
        year={2014},
        doi={10.1016/j.jat.2013.09.008}
    }

Citations
=========

To cite an entry in a page, use ``:footcite:`` role followed by the entry key.
For example:

.. code-block::

   Earlier versions of this statement were limited to the case
   where :math:`P_A` is given by a (sparse) tensorial grid\ :footcite:`Dyn2014`.

.. note::

   Notice that the backslash that precedes the space
   before ``:footcite:`` directive; it suppresses the space when rendered.

will be rendered as:

   Earlier versions of this statement were limited to the case
   where :math:`P_A` is given by a (sparse) tensorial grid\ :footcite:`Dyn2014`.

Multiple citation keys can be specified in the ``:footcite:`` role.
For example:

.. code-block::

   Spline-type interpolation is based on works of by Carl de Boor et al.\ :footcite:`DeBoor1972, DeBoor1977, DeBoor1978, DeBoor2010`.

will be rendered as:

   Spline-type interpolation is based on works of by Carl de Boor et al.\ :footcite:`DeBoor1972, DeBoor1977, DeBoor1978, DeBoor2010`.

Displaying a list of references
===============================

In the ``minterpy`` documentation, a list of references is displayed
for each page that contains bibliographic citations
(as opposed to having a single page that lists everything).
If a page contain bibliographic citations, the list of references
should be displayed at the end of document
using the ``.. footbibliography::`` directive.
Use ``References`` as the first-level heading.

For example:

.. code-block:: rest

   ...

   References
   ##########

   .. footbibliography::


which will be rendered as (``References`` heading is intentionally not displayed):

   .. footbibliography::

Consider the following best-practice recommendations:

.. tip::

   - When possible, always include the digital object identifier (`DOI`_) for each
     entry in the bibliography file.
   - Don't forget the backslash that precedes the space before ``:footcite:`` role;
     It will suppress the space when rendered.
   - Display the list of references at the very end of each page that contains
     bibliographic citations.
   - Use ``References`` as the heading title of the list of references.

Notes
=====

- Bibliographic citations in the ``minterpy`` documentation uses the `bibtex extension`_ for Sphinx.

- The `bibtex extension documentation`_ recommends using ``footcite`` and
  ``footbibliography`` to create a *local* bibliography.
  The ``minterpy`` documentation follows this recommendation.

  .. important::

     Doing this saves us a lot of trouble customizing the ``bibtex`` extension
     to avoid duplication issues.

Code examples
#############

Use code examples to illustrate how ``minterpy`` programming elements might be
used to achieve a certain goal. Depending on the length they might fall into
different categories:

- Simple one-liner, in-line with the text (in-line code examples)
- Short to long, self-contained examples used to illustrate a point or two
  (code example blocks).

In-line code examples
=====================

Use the ``:code:`` role to put a code examples.
For example:

.. code-block:: rest

   Load ``minterpy`` using :code:`import minterpy as mp`

will be rendered as:

    Load ``minterpy`` using :code:`import minterpy as mp`

Code example blocks
===================

Code example blocks are written using the ``.. code-block::`` directive.
For example:

.. code-block:: rest

   .. code-block::

       import minterpy as mp

       mi = mp.MultiIndexSet.from_degree(3, 2, 1)

will be rendered as:

    .. code-block::

       import minterpy as mp

       mi = mp.MultiIndexSet.from_degree(3, 2, 1)

Sphinx also supports syntax highlighting for various programming languages.
Specify the language after the ``.. code-block::`` directive.
Use the proper syntax highlighting when it is appropriate.
Python code in the ``minterpy`` docs should be syntax-highlighted.

For example, the same code above should be written:

.. code-block:: python

   import minterpy as mp

   mi = mp.MultiIndexSet.from_degree(3, 2, 1)

Code examples involving interactive Python session should be written
using the ``pycon`` (python console) language specification.

For example:

.. code-block:: rest

    .. code-block:: pycon

        >>> import minterpy as mp
        >>> mi = mp.MultiIndexSet.from_degree(3, 2, 1)
        >>> mi
        MultiIndexSet
        [[0 0 0]
         [1 0 0]
         [2 0 0]
         [0 1 0]
         [1 1 0]
         [0 2 0]
         [0 0 1]
         [1 0 1]
         [0 1 1]
         [0 0 2]]

will be rendered as:

    .. code-block:: pycon

        >>> import minterpy as mp
        >>> mi = mp.MultiIndexSet.from_degree(3, 2, 1)
        >>> mi
        MultiIndexSet
        [[0 0 0]
         [1 0 0]
         [2 0 0]
         [0 1 0]
         [1 1 0]
         [0 2 0]
         [0 0 1]
         [1 0 1]
         [0 1 1]
         [0 0 2]]

Cross-referencing code blocks
=============================

Cross-referencing a code example block may be done via custom anchor (label).
For instance, create an anchor for a code example to be cross-referenced later:

.. code-block:: rest

   .. _code-example:

   .. code-block:: python

      fx = lambda x: np.sin(x)
      fx_interpolator = mp.interpolate(fx, 1, 3)

this will be rendered as:

   .. _code-example:

   .. code-block:: python

      fx = lambda x: np.sin(x)
      fx_interpolator = mp.interpolate(fx, 1, 3)

and can be cross-referenced using the ``:ref:`` directive.
For example:

.. code-block:: rest

   See the code example :ref:`code example <code-example>`.

which will be rendered as:

   See the :ref:`code example <code-example>`.

.. important::

   Cross-referencing a code example block always requires a custom title.

Consider the following best-practice recommendations:

.. tip::

   - While double backticks and ``:code:`` role render the texts inside using
     a fixed-width font, always use ``:code:`` role for displaying
     inline code example for clarity.
   - When available, always specify the language in the code example block for
     syntax highlighting. Python code example in the ``minterpy`` docs
     should be syntax highlighted.
   - If you need to cross-reference a code example block, a custom label must be
     defined and the label must be unique across the docs.
     Always check for "duplicate labels" warning when building the docs.
   - Assume people will copy and paste code blocks you write, perhaps with some
     modifications, for their own use. Try to put code examples that make sense.
   - Use common sense when it comes to the length of a code block.
     A code block that is too long and doesn't have a narrative is hard to read
     in the docs.

Cross-references
################

The ``minterpy`` docs uses various types of cross-references (linking),
including: external and internal cross-references, bibliographic citations, etc.

.. seealso::

   There are various types of internal cross-references used in the ``minterpy``
   documentation specific to documentation elements
   (pages, section headings, images, equations, API elements, etc.).
   This guideline covers pages, section headings, and API elements
   cross-references;
   other types of internal cross-referencing may be found in its own guideline.

External cross-references
=========================

External cross-references provide links to external resources,
primarily to other pages on the web.

The ``minterpy`` docs uses the `link-target`_ approach
to cross-reference external resources.
Using this approach, the link text that appears on a page is separated from
the target that it points to.
This allows for a cleaner documentation page source
and target reuse (at least, within the same page).

As an example:

.. code-block:: rest

   The problem is well explained in this `Wikipedia article`_
   and also in a `DeepAI article`_.

   .. _Wikipedia article: https://en.wikipedia.org/wiki/Curse_of_dimensionality
   .. _DeepAI article: https://deepai.org/machine-learning-glossary-and-terms/curse-of-dimensionality

which will be rendered as:

    The problem is well explained in this `Wikipedia article`_
    and also in a `DeepAI article`_.

Page cross-references
=====================

A whole documentation page (a single reST file) may be cross-referenced using
the ``:doc:`` role.
The default syntax is:

.. code-block:: rest

   :doc:`<target>`

For example, to cross-reference the main page of the Developers guide, type:

.. code-block:: rest

   See the :doc:`/contributors/index` for details.

which will be rendered as:

    See the :doc:`/contributors/index` for details.

.. important::

    Don't include the ``.rst`` extension when specifying the target in
    the ``:doc:`` role.

By default, the displayed link title is the title of the page.
You can replace the default title using the following syntax:

.. code-block:: rest

   :doc:`custom_link_title <target>`

Replace ``custom_link_title`` accordingly.
For example:

.. code-block:: rest

   For details, see the Developers guide :doc:`here </contributors/index>`.

which will be rendered as:

    For details, see the Developers guide :doc:`here </contributors/index>`.

The target specification may be written in two different ways:

- a document relative to the current document. For example.
  ``:doc:ipynb-based-docs`` refers to the
  :doc:`ipynb-based-docs` section of the docs contribution guidelines.
- full path (relative to the root ``docs`` directory).
  The example above is specified as a full path.

.. important::

    Don't forget to include the backslash in front of the directory name
    if it's specified in full path (relative to the root ``docs`` directory).

Section headings cross-references
=================================

Section headings within a page may be cross-referenced using the `:ref:` role.
The ``minterpy`` documentation uses the `autosectionlabel`_ extension for Sphinx;
this means that you don't need to explicitly label a heading before you can cross-reference it.
Furthermore, all section heading labels are ensured to be unique.

The syntax to cross-reference a section heading is:

.. code-block:: rest

   :ref:`path/to/document:Heading title`

By default, the heading title in the page will be rendered.
To display a custom title, use:

.. code-block:: rest

   :ref:`custom_link_title <path/to/document:Heading title>`

For example, to cross-reference the math blocks section
of the documentation contribution guidelines, type:

.. code-block:: rest

   To write math blocks in the ``minterpy`` documentation,
   refer to :ref:`contributors/contrib-docs/guidelines/rest-based-docs:Mathematics blocks`.

which will be rendered as:

   To write math blocks in the ``minterpy`` documentation,
   refer to :ref:`contributors/contrib-docs/guidelines/rest-based-docs:Mathematics blocks`.

To replace the default title, type:

.. code-block:: rest

   To write math blocks in the ``minterpy`` documentation,
   refer to the :ref:`relevant section <contributors/contrib-docs/guidelines/rest-based-docs:Mathematics blocks>`
   in the docs contribution guidelines.

which will be rendered as:

   To write math blocks in the ``minterpy`` documentation,
   refer to the :ref:`relevant section <contributors/contrib-docs/guidelines/rest-based-docs:Mathematics blocks>`
   in the docs contribution guidelines.

.. important::

    Don't *include* the backslash in front of the directory name for target
    specified using ``:ref:`` role. The path is always relative
    to the root ``docs`` directory.

``minterpy`` API elements
=========================

Elements of the documented ``minterpy`` API (including modules, functions, classes,
methods, attributes or properties) may be cross-referenced in the docs.
The `Python domain`_ allows for cross-referencing most documented objects.
Before you can cross-reference an API element,
its documentation must be available in the :doc:`/api/index`.

Refer to the to the table below for some usages and examples.

=========  ==================  =========================================  =====================================
Element    Role                Example                                    Rendered as
=========  ==================  =========================================  =====================================
Module     :code:`:py:mod:`    ``:py:mod:`.transformations.lagrange```    :py:mod:`.transformations.lagrange`
Function   :code:`:py:func:`   ``:py:func:`.interpolate```                :py:func:`.interpolate`
Class      :code:`:py:class:`  ``:py:class:`.core.grid.Grid```            :py:class:`.core.grid.Grid`
Method     :code:`:py:meth:`   ``:py:meth:`.MultiIndexSet.from_degree```  :py:meth:`.MultiIndexSet.from_degree`
Attribute  :code:`py:attr:`    ``:py:attr:`.MultiIndexSet.exponents```    :py:attr:`.MultiIndexSet.exponents`
=========  ==================  =========================================  =====================================

.. important::

    Precede the object identifier with a dot indicating that it is relative
    to the ``minterpy`` package.

Other projects' documentation cross-references
==============================================

Documentation from other projects (say, ``NumPy``, ``Scipy``, or ``Matplolib``)
may be cross-referenced in the ``minterpy`` documentation.

To cross-reference a part or an API element from another project's docs,
use the following syntax:

.. code-block:: rest

   :py:<type>:`<mapping_key>.<ref>`

replace ``<type>`` with one of the types listed in the table above,
``<mapping_key>`` with the key listed in the ``intersphinx_mapping`` variable
inside the ``conf.py`` file, and ``ref`` with the actual documentation element.

For example, to refer to the docs for ``ndarray`` in the ``NumPy`` docs, write:

.. code-block:: rest

   :class:`numpy:numpy.ndarray`

which will be rendered as:

   :class:`numpy:numpy.ndarray`

This functionality is provided by the `intersphinx`_ extension for Sphinx.

.. note::

   Check the variable ``intersphinx_mapping`` inside the ``conf.py`` file
   of the Sphinx documentation for updated list of mappings.

Consider the following best-practice recommendation:

.. tip::

   - For external cross-references, use the `link-target`_ approach to define
     an external cross-reference and put the list of targets at the very bottom
     of a page source. See the source of this page for example.
   - Try to be descriptive with what being cross-referenced; use custom link title
     if necessary.

..
   - When you cross-reference a ``minterpy``  API element anywhere
     in the documentation, try to provide a context on why the element
     is being cross-referenced.

     For example, instead of writing:

       Finally, we call the monomials :math:`x^\alpha = \prod_{i=1}^m x^{\alpha_i}_{i}`, :math:`\alpha \in A` the
       *canonical basis* (see :py:class:`.CanonicalPolynomial`) of :math:`\Pi_{A}`.

     use a :

       Finally, we call the monomials :math:`x^\alpha = \prod_{i=1}^m x^{\alpha_i}_{i}`, :math:`\alpha \in A` the
       *canonical basis* of :math:`\Pi_{A}`.

       .. SEEALSO::

          In ``minterpy``, the canonical polynomial basis is implemented as :py:class:`.CanonicalPolynomial` class.

Glossary
########

The ``minterpy`` docs contains many specific terminologies coming
from either mathematics, computer science, and the Python ecosystem.
Moreover, ``minterpy`` also defines its own small set of terminologies related
to its user interface and implementation.

Glossary page
=============

The :doc:`/glossary` is a page within the ``minterpy`` docs that
*briefly* defines all the terms that might be useful for users to know.
Expanded definition, if any, may be cross-referenced in the definition.
The Glossary page is accessible from all the pages of the ``minterpy``
documentation.

The ``minterpy`` docs uses the Sphinx `built-in Glossary`_ to create a glossary.
A reST file named ``glossary.rst`` located in the root ``docs`` directory
contains all the Glossary entries along with their brief definition.

Glossary terms
==============

Add a new glossary entry in the ``glossary.rst``.

For example, to add a glossary entry named *my term*, type:

.. code-block:: rest

    .. Glossary::
       :sorted:

       my term

           Define the term briefly.
           Create a cross-reference either internally or externally, if necessary.

Cross-referencing glossary terms
================================

To cross-reference a glossary term used in the text, use the ``:term:`` role
with the following syntax:

.. code-block:: rest

   :term:`a defined term`

Replace ``a defined term``  with a term already defined in the Glossary.

For example the entry *Chebyshev nodes* is already defined in the Glossary.
To cross-reference this term in text, type:

.. code-block:: rest

   For :math:`\Omega = [-1, 1]`, a classic choice of sub-optimal nodes are the
   :term:`Chebyshev nodes`.

which will be rendered as:

   For :math:`\Omega = [-1, 1]`, a classic choice of sub-optimal nodes are the
   :term:`Chebyshev nodes`.

By default, the text displayed is the term itself as defined in the Glossary.

.. important::

   The entry used in the cross-reference must match exactly with the one
   in the Glossary.

To replace the default title, use the following syntax:

.. code-block::

   :term:`custom term title <a defined term>`

For example:

.. code-block:: rest

   For :math:`\Omega = [-1, 1]`, a classic choice of sub-optimal nodes are the
   :term:`roots of Chebyshev polynomials <Chebyshev nodes>`.

will be rendered as:

   For :math:`\Omega = [-1, 1]`, a classic choice of sub-optimal nodes are the
   :term:`roots of Chebyshev polynomials <Chebyshev nodes>`.

Consider the following best-practice recommendations:

.. tip::

   Consider creating an entry in the :doc:`/glossary` page if a term you are using is:

   - Specialized terms used in the general multivariate polynomial interpolation problems.
   - Specialized terms used in the approach of ``minterpy``
     (especially if they are related to its conventions).
   - Other more general terms that would be useful to define so as to make
     the docs more self-contained.

   The following are the best-practice of writing an entry in the :doc:`/glossary`:

   - One, two, or three sentences summary of what the entry is; don't be circular
   - Use consistent capitalization
   - Cross-reference other part of ``minterpy`` docs, when applicable,
     to either:

     - the :doc:`/getting-started/index` or :doc:`/how-to/index`
       guides for usage examples of the term.
     - the :doc:`Fundamentals </fundamentals/index>` guide for an expanded
       definition or more theoretical explanation
     - External resources (say, `Wikipedia`_) for a more general term
       that is included for the sake of completeness.
       Note that external resources may use different conventions
       that without further explanation might lead to confusion.

   - Put a new entry in the alphabetical order with the previous entries.
     Though they all will be sorted when rendered,
     it makes the documentation source code cleaner.

Mathematics
###########

In the ``minterpy`` docs,
Sphinx is set to display mathematical notations using `MathJax`_.
The MathJax library provides extensive support for LaTeX,
*the* markup language for mathematics.

Inline mathematics
==================

Inline mathematics can be written using the ``:math:`` role.
For example:

.. code-block:: rest

   :math:`A_{m,n,p} = \left\{\boldsymbol{\alpha} \in \mathbb{N}^m | \|\boldsymbol{\alpha}\|_p \leq n, m,n \in \mathbb{N}, p \geq 1 \right\}` is the multi-index set.

will be rendered as:

    :math:`A_{m,n,p} = \left\{\boldsymbol{\alpha} \in \mathbb{N}^m | \|\boldsymbol{\alpha}\|_p \leq n, m,n \in \mathbb{N}, p \geq 1 \right\}` is the multi-index set.

Mathematics blocks
==================

Mathematics blocks can be written using the ``.. math::`` directive.
For example:

.. code-block:: rest

   .. math::

      N_{\boldsymbol{\alpha}}(\boldsymbol{x}) = \prod_{i=1}^{M} \prod_{j=0}^{\alpha_i - 1} (x_i - p_{j,i}), \; \boldsymbol{\alpha} \in A


will be rendered as:

    .. math::

       N_{\boldsymbol{\alpha}}(\boldsymbol{x}) = \prod_{i=1}^{M} \prod_{j=0}^{\alpha_i - 1} (x_i - p_{j,i}), \; \boldsymbol{\alpha} \in A

Numbering and cross-referencing
===============================

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
   docs doesn't use numbered table of contents.
   Therefore, it is not straightforward to cross-reference an equation defined
   in another page.
   Use instead the nearest or the most relevant heading to the equation
   as an anchor.

Consider the following best-practice recommendations:

.. tip::

   - Use the following syntax to label an equation:

     .. code-block:: rest

        :label: `eq:equation_name`

     and replace the ``equation_name`` part with the actual name of the equation
     but keep the preceding ``eq:``.


   - Avoid cross-referencing an equation in one page from another.
     Use, instead, the nearest or the most relevant heading to the equation
     as an anchor.
     See the guidelines of
     :ref:`section heading cross-references <contributors/contrib-docs/guidelines/rest-based-docs:Section headings cross-references>`
     for details.

.. important::

   The ``equation_name`` for the label must be unique across the documentation.
   Make sure there's no "duplicate warning" when building the docs.

   If such warnings arise, use common sense to rename the equation.

.. _Jupyter notebooks: https://jupyter-notebook.readthedocs.io/en/stable/
.. _reStructuredText Primer: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _Is the 80 character line limit still relevant: https://www.richarddingwall.name/2008/05/31/is-the-80-character-line-limit-still-relevant
.. _Semantic Linefeeds: https://rhodesmill.org/brandon/2012/one-sentence-per-line
.. _bibtex extension: https://sphinxcontrib-bibtex.readthedocs.io/en/latest/index.html
.. _BibTeX: http://www.bibtex.org
.. _DOI: https://en.wikipedia.org/wiki/Digital_object_identifier
.. _bibtex extension documentation: https://sphinxcontrib-bibtex.readthedocs.io/en/latest/usage.html#local-bibliographies
.. _link-target: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#hyperlinks
.. _Wikipedia article: https://en.wikipedia.org/wiki/Curse_of_dimensionality
.. _DeepAI article: https://deepai.org/machine-learning-glossary-and-terms/curse-of-dimensionality
.. _autosectionlabel: https://www.sphinx-doc.org/en/master/usage/extensions/autosectionlabel.html
.. _Python domain: https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#cross-referencing-python-objects
.. _intersphinx: https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
.. _built-in Glossary: https://www.sphinx-doc.org/en/master/glossary.html
.. _Wikipedia: https://www.wikipedia.org
.. _MathJax: https://www.mathjax.org/
