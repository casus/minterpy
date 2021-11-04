#############
Code examples
#############

Use code examples to illustrate how ``minterpy`` programming elements might be
used to achieve a certain goal. Depending on the length they might fall into
different categories:

- Simple one-liner, in-line with the text (in-line code examples)
- Short to long, self-contained examples used to illustrate a point or two
  (code example blocks).

In-line code examples
#####################

Use the ``:code:`` role to put a code examples.
For example:

.. code-block:: rest

   Load ``minterpy`` using :code:`import minterpy as mp`

will be rendered as:

    Load ``minterpy`` using :code:`import minterpy as mp`

Code example blocks
####################

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
Python code in the ``minterpy`` documentation should be syntax-highlighted.

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

Cross-referencing
#################

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

Best-practice recommendations
#############################

- While double backticks and ``:code:`` role render the texts inside using
  a fixed-width font, always use ``:code:`` role for displaying
  inline code example for clarity.
- When available, always specify the language in the code example block for
  syntax highlighting. Python code example in the ``minterpy`` documentation
  should be syntax highlighted.
- If you need to cross-reference a code example block, a custom label must be
  defined and the label must be unique across documentation.
  Always check for "duplicate labels" warning when building the documentation.
- Assume people will copy and paste code blocks you write, perhaps with some
  modifications, for their own use. Try to put code examples that make sense.
- Use common sense when it comes to the length of a code block.
  A code block that is too long and doesn't have a narrative is hard to read
  in the documentation.
