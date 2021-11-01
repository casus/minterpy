#######################
80-Character width rule
#######################

This rule might seem archaic as computer displays have no problem displaying
more than 80 characters in a single line.
However, long lines of text tend to be harder to read
and limiting a line to 80 characters does improve readability.
Besides, a lot of people prefer to work with half-screen windows.

Best-practice recommendations
#############################

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


.. _Is the 80 character line limit still relevant: https://www.richarddingwall.name/2008/05/31/is-the-80-character-line-limit-still-relevant
.. _Semantic Linefeeds: https://rhodesmill.org/brandon/2012/one-sentence-per-line
