########
Glossary
########

The ``minterpy`` documentation contains many specific terminologies coming
from either mathematics, computer science, and the Python ecosystem.
Moreover, ``minterpy`` also defines its own small set of terminologies related
to its user interface and implementation.

Glossary page
#############

The :doc:`/glossary` is a page within the ``minterpy`` documentation that
*briefly* defines all the terms that might be useful for users to know.
Expanded definition, if any, may be cross-referenced in the definition.
The Glossary page is accessible from all the pages of the ``minterpy``
documentation.

The ``minterpy`` documentation uses the Sphinx `built-in Glossary`_
to create a glossary.
A reST file named ``glossary.rst`` located in the root ``docs`` directory
contains all the Glossary entries along with their brief definition.

Glossary terms
##############

Add a new glossary entry in the ``glossary.rst``.

For example, to add a glossary entry named *my term*, type:

.. code-block:: rest

    .. Glossary::
       :sorted:

       my term

           Define the term briefly.
           Create a cross-reference either internally or externally, if necessary.

Cross-referencing
#################

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

Best-practice recommendations
#############################

Consider creating an entry in the :doc:`/glossary` page if a term you are using is:

- Specialized terms used in the general multivariate polynomial interpolation problems.
- Specialized terms used in the approach of ``minterpy``
  (especially if they are related to its conventions).
- Other more general terms that would be useful to define so as to make
  the documentation more self-contained.

The following are the best-practice of writing an entry in the :doc:`/glossary`:

- One, two, or three sentences summary of what the entry is; don't be circular
- Use consistent capitalization
- Cross-reference other part of ``minterpy`` documentation, when applicable,
  to either:

  - the :doc:`Getting started </getting-started/index>` or :doc:`How-to </howto/index>`
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


.. _built-in Glossary: https://www.sphinx-doc.org/en/master/glossary.html
.. _Wikipedia: https://www.wikipedia.org
