#######################
Bibliographic citations
#######################

A bibliographic citation is a special case of :doc:`cross-referencing <cross-ref>`.
Specifically, it cross-references external scientific works
such as articles, books, or reports.
You need to include any relevant scientific works in the ``minterpy`` documentation
if and when applicable.
This is typically the case when writing the :ref:`Fundamentals <fundamentals/index:Fundamentals of \`\`minterpy\`\`>` guide.

Bibliography file
#################

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
#########

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
###############################

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

Best-practice recommendations
#############################

- When possible, always include the digital object identifier (`DOI`_) for each
  entry in the bibliography file.
- Don't forget the backslash that precedes the space before ``:footcite:`` role;
  It will suppress the space when rendered.
- Display the list of references at the very end of each page that contains
  bibliographic citations.
- Use ``References`` as the heading title of the list of references.

Notes
#####

- Bibliographic citations in the ``minterpy`` documentation uses the `bibtex extension`_ for Sphinx.

- The `bibtex extension documentation`_ recommends using ``footcite`` and
  ``footbibliography`` to create a *local* bibliography.
  The ``minterpy`` documentation follows this recommendation.

  .. important::

     Doing this saves you a lot of trouble customizing the ``bibtex`` extension
     to avoid duplication issues.

.. _bibtex extension: https://sphinxcontrib-bibtex.readthedocs.io/en/latest/index.html
.. _BibTeX: http://www.bibtex.org
.. _DOI: https://en.wikipedia.org/wiki/Digital_object_identifier
.. _bibtex extension documentation: https://sphinxcontrib-bibtex.readthedocs.io/en/latest/usage.html#local-bibliographies
