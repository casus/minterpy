###########
Admonitions
###########

Sphinx provides support for a whole class of built-in `admonitions`_
as a set of directives to render text inside a highlighted box.

Available admonitions
#####################

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

Best-practice recommendations
#############################

- Use admonitions sparingly and judiciously in the ``minterpy`` documentation
  as they tend to obstruct the reading flow.
  Besides, if used too often, readers may become immune to notes and warnings
  and would simply ignore them.
