################
Cross-references
################

The ``minterpy`` documentation uses three different types of cross-references
(linking): external, internal, and bibliographic citations.

.. seealso::

   There are various types of internal cross-references used in the ``minterpy``
   documentation specific to documentation elements
   (pages, section headings, images, equations, API elements, etc.).
   This guideline only external cross references and covers pages
   and sections internal cross-referencing;
   other types of internal cross-referencing may be found in its own guideline.

External cross-references
#########################

External cross-references provide links to external resources,
primarily to other pages on the web.

The ``minterpy`` documentation uses the `link-target`_ approach
to cross-reference external resources.
Using this approach, the link text that appears on a page is separated from
the target that it points to.
This allows for a cleaner documentation page source
and target reuse (at least, within the same page).

As an example:

.. code-block:: rest

   The problem is well explained in this `Wikipedia article`_
   and also in a `DeepAI article`_.

   .. _`Wikipedia article`: https://en.wikipedia.org/wiki/Curse_of_dimensionality
   .. _`DeepAI article`: https://deepai.org/machine-learning-glossary-and-terms/curse-of-dimensionality

which will be rendered as:

    The problem is well explained in this `Wikipedia article`_
    and also in a `DeepAI article`_.

Page cross-references
#####################

A whole documentation page (a single reST file) may be cross-referenced using
the ``:doc:`` role.
The default syntax is:

.. code-block:: rest

   :doc:`<target>`

For example, to cross-reference the main page of the Developers guide, type:

.. code-block:: rest

   See the :doc:`/developers/index` for details.

which will be rendered as:

    See the :doc:`/developers/index` for details.

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

   For details, see the Developers guide :doc:`here </developers/index>`.

which will be rendered as:

    For details, see the Developers guide :doc:`here </developers/index>`.

The target specification may be written in two different ways:

- relative to the current document. For example.
  ``:doc:bibliography`` refers to the
  :doc:`bibliography` section of the documentation contribution guidelines.
- full path (relative to the root ``docs`` directory). The example above is
  specified as a full path.

.. important::

    Don't forget to include the backslash in front of the directory name
    if it's specified in full path (relative to the root ``docs`` directory).

Section headings cross-references
#################################

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
   refer to :ref:`developers/contrib-doc/contrib-doc-guidelines/rest-based-doc/mathematics:Mathematics blocks`.

which will be rendered as:

   To write math blocks in the ``minterpy`` documentation,
   refer to :ref:`developers/contrib-doc/contrib-doc-guidelines/rest-based-doc/mathematics:Mathematics blocks`.

To replace the default title, type:

.. code-block:: rest

   To write math blocks in the ``minterpy`` documentation,
   refer to the :ref:`relevant section <developers/contrib-doc/contrib-doc-guidelines/rest-based-doc/mathematics:Mathematics blocks>`
   in the documentation contribution guidelines.

which will be rendered as:

   To write math blocks in the ``minterpy`` documentation,
   refer to the :ref:`relevant section <developers/contrib-doc/contrib-doc-guidelines/rest-based-doc/mathematics:Mathematics blocks>`
   in the documentation contribution guidelines.

.. important::

    Don't *include* the backslash in front of the directory name for target
    specified using ``:ref:`` role. The path is always relative
    to the root ``docs`` directory.

Best-practice recommendations
#############################

- For external cross-references, use the `link-target`_ approach to define
  an external cross-reference and put the list of targets at the very bottom
  of a page source. See the source of this page for example.
- Try to be descriptive with what being cross-referenced; use custom link title
  if necessary.

.. _link-target: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#hyperlinks
.. _`Wikipedia article`: https://en.wikipedia.org/wiki/Curse_of_dimensionality
.. _`DeepAI article`: https://deepai.org/machine-learning-glossary-and-terms/curse-of-dimensionality
.. _autosectionlabel: https://www.sphinx-doc.org/en/master/usage/extensions/autosectionlabel.html