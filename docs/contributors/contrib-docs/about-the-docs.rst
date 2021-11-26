##############
About the Docs
##############

We think documentation is essential to the continuous development of any software.
Good documentation helps users and contributors, including ourselves,
climb the steep curve of using and contributing to ``minterpy``.
That's why we spend a great deal of time developing and improving the docs;
if you'd like, you can also :doc:`join and help us <how-to-contrib>`
maintain and improve it.

On this page, you'll learn more about the underlying design of the ``minterpy``
docs as well as the tooling behind its creation.

Design
######

There isn't *one* documentation, but *four* of them,
each with its distinct purpose.
We follow this documentation design principle from the `Documentation System by Divio`_
to structure the ``minterpy`` docs.
The four different *categories* of docs are:

- Tutorials
- How-to
- Explanation (fundamentals, theory, etc.)
- (API) reference

The distinction between these categories is summarized in the table below.

===============  ===============  =============== ===================== ===========
Category         Orientation      Main content    Most useful when...   Think of...
===============  ===============  =============== ===================== ===========
Tutorials        Learning         Practical steps learning the code     Lessons
How-to           Problem-solving  Practical steps working with the code Recipes
Explanation      Understanding    Exposition      learning the code     Textbooks
(API) reference  Information      Exposition      working with the code Dictionary
===============  ===============  =============== ===================== ===========

Or if you're more visually-oriented...

.. figure:: /assets/images/contributors/documentation-system.png
  :align: center

  Four distinct categories of docs (adapted from the `Documentation System`_).

.. seealso::

   We encourage you to read through the design as outlined in
   the `Documentation System`_ and watch a talk by the author Daniele Procida
   in the `Pycon 2017`_.

The :ref:`main sections <contributors/contrib-docs/how-to-contrib:Docs source structure>`
of the ``minterpy`` docs reflect all these four categories.
Additionally, we include a special section in the docs for contributors called
the :doc:`/contributors/index`.
In principle, these are mainly how-to guides written for contributors instead of users.
Each addition to the docs should be clearly defined
to which single category of docs it belongs.

Tools and frameworks
####################

.. figure:: ./images/minterpy-documentation-tooling.png
  :align: center

  The components and process to build and, eventually, deploy the ``minterpy``
  documentation online.


.. _Documentation System by Divio : https://documentation.divio.com/
.. _Documentation System: https://documentation.divio.com/
.. _Pycon 2017: https://www.youtube.com/watch?v=azf6yzuJt54
