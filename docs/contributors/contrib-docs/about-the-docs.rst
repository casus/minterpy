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

- :ref:`contributors/contrib-docs/about-the-docs:Tutorials`
- :ref:`contributors/contrib-docs/about-the-docs:How-to guides`
- :ref:`contributors/contrib-docs/about-the-docs:Explanation (fundamentals, theory)`
- :ref:`(API) reference <contributors/contrib-docs/about-the-docs:API reference>`

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

Tutorials
=========

*Tutorials* are *learning-oriented* docs; think of them as *lessons*
for ``minterpy`` users.
The tasks and outcomes in the tutorials should be meaningful and immediate
while at the same time sparing users from unnecessary explanations, jargon,
and technical details (at least at the beginning).
Furthermore, they should have a minimal, but non-trivial, context.
Writing tutorials is hard; you need to select the problem (easy yet non-trivial)
and to judiciously and progressively disclose important and more advanced topics
as the tutorial progresses.

Well-curated tutorials are an important (if not the most important) element
of the docs for onboarding both users and contributors.
They provide a glimpse of what ``minterpy`` can do
and how to do it according to the intended design.

We organize all tutorials inside the :doc:`/getting-started/index`
and group them according to the difficulty (beginner to advanced)
and major ``minterpy`` features.

How-to guides
=============

*How-to Guides* are *problem-solving-oriented* docs;
think of them as *recipes* to do a particular task with ``minterpy``.
Each guide contains step-by-step instructions on how to solve a clearly defined problem.
How-to guides are largely free of context;
we assume that users, who have gained some proficiency with ``minterpy``,
are looking to solve some common and particular problems using ``minterpy``.

Writing how-to guides is a bit easier than writing tutorials
as the problems are more well defined, context-free,
and the users' assumed proficiency.

We organize all how-to guides inside the :doc:`/how-to/index`
and group them according to common tasks in numerical computing using functions
(e.g., interpolation, regression, differentiation, integration).

Explanation (fundamentals, theory)
==================================

*Explanation or fundamentals* are *understanding-oriented* docs;
think of them as the *theoretical expositions* (as in textbooks)
of the mathematics that underlies ``minterpy``.
Fundamentals provides the context and background of ``minterpy``,
the different layers of abstraction starting from the top,
and why things work the way they are.
They avoid instructions and minimize the descriptions
related to the implementation in the code.

We write the fundamentals section of the docs
to help users and contributors *understand* `` minterpy``
and the concepts behind it better.
Fundamentals is an important element of the docs for advancing users and contributors.
A user may become a contributor, and a contributor a better one,
after they use ``minterpy`` and understand the theory behind it better.

Writing fundamentals is tricky because the topics are more open-ended;
you need to decide what topic you want to explain,
the depth of your explanation, and where to end it.

We organize all the theoretical topics of ``minterpy`` inside
the :doc:`/fundamentals/index`.

API reference
=============

*API reference* is *information-oriented* docs;
think of them as a *dictionary* or an *encyclopedia* [#]_
that describes all the exposed components and machinery of `minterpy`.
API reference avoids explaining basic concepts
or providing thorough usage examples;
its main task is *to describe*.
This docs is important, particularly for advanced users and contributors.

This kind of docs tends to be terse and have a well-defined and consistent structure.
Furthermore, it has an almost one-to-one correspondence with the codebase itself.
If you're a developer, you'd most probably be familiar
and already comfortable with creating docs for API reference
(docs for modules, classes, functions, etc.).

We organize all the references for the exposed ``minterpy`` components
in the :doc:`/api/index`.

Contributors guides
===================

*Contributors guides* are docs written for, well, contributors.
They are mainly how-to guides for contributing to the ``minterpy`` project,
either to its development (codebase, dev) or to its documentation (docs).
It also contains some (meta) information regarding the project organization,
the history, and the people behind it.

Tools and frameworks
####################


.. figure:: ./images/minterpy-documentation-tooling.png
  :align: center

  The components and process to build and, eventually, deploy the ``minterpy``
  documentation online.

.. rubric:: Footnotes
.. [#] If you don't know what that is, here is `a Wikipedia article`_ about it.

.. _Documentation System by Divio : https://documentation.divio.com/
.. _Documentation System: https://documentation.divio.com/
.. _Pycon 2017: https://www.youtube.com/watch?v=azf6yzuJt54
.. _a Wikipedia article: https://en.wikipedia.org/wiki/Encyclopedia

