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
that describes all the exposed components and machinery of ``minterpy``.
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

Frameworks and tools
####################

In realizing the ``minterpy`` docs according to the above design principle,
we treat the docs *as code*.
This following the `docs-like-code framework`_ means:

- We write the docs source files in plain text using a markup language.
- We store the docs source files in a version control system,
  currently, in the `same repository`_ as the ``minterpy`` package codebase.
- We set up the contribution-to-docs workflow similar to the
  contribution-to-dev workflow, through issues tracker, commits, merge requests,
  and merge reviews. Your contributions to the docs are counted and valued
  the same as your contribution to the development.
- We build the docs artifacts automatically and publish them online
  (although you can have it local as well).

We use the `Sphinx documentation generator`_ to build the docs as HTML document [#]_
The HTML document may then be opened locally or later deployed (published)
as a static website using a hosting service such as `Read the Docs`_ [#]_.

Sphinx, by default, uses `reStructuredText (reST)`_ as the markup language.
Therefore, we write the bulk of the ``minterpy`` docs in reST, saved in
files with an ``.rst`` extension.

reST is also used to write the `docstrings`_ directly in the ``minterpy``
Python codebase (files with a ``.py`` extension)
as suggested by `PEP 287`_.
This way, most of the :doc:`/api/index` is automatically generated by processing
the docstrings in the Python source code.

There is, however, a notable exception.
Tutorials and how-to guides are typically full of code examples and the outputs
(texts or plots) of their execution.
So instead of writing them in reST, we write them as `Jupyter notebooks`_
(files with an ``.ipynb`` extension) in which executable codes, their outputs,
and narrative markup texts surrounding the codes and outputs may be combined
in a single document.
The Sphinx build process will automatically execute the notebooks,
capture the output, convert them all into HTML documents to be included
in the docs.

This choice simplifies the docs writing process and saves us from painful
copying-and-pasting relevant code outputs (including plots) into a separate reST files.
It comes with a minor caveat, however; the default markup language for
Jupyter notebooks is `Markdown`_, not reST.
So some contributors, working across docs categories, may have to do a bit of context switching.
The good news is, while less comprehensive, Markdown is easier to learn and write than reST.
You would most probably be more familiar already with Markdown than with reST.

The figure below illustrates the different components and processes involved
in building the ``minterpy`` docs.

.. figure:: /assets/images/contributors/documentation-tooling.png
  :align: center

  The components and processes to build and, eventually, deploy the ``minterpy``
  documentation online.

.. note::

   In summary, the ``minterpy`` docs has four main docs *categories*:

   - Getting Started Guides (tutorials)
   - How-to Guides
   - Fundamentals
   - API Reference

   The Contributors Guides are how-to guides for contributors
   to the ``minterpy`` project.

   There are three *types of document* for the docs:

   - reStructuredText (reST) files (with an ``.rst`` extension)
   - Jupyter notebooks (with an ``.ipynb`` extension)
   - Docstrings embedded in the ``minterpy`` Python source code
     (with an ``.py`` extension)

.. rubric:: Footnotes
.. [#] If you don't know what that is, here is `a Wikipedia article`_ about it.
.. [#] Other formats like PDF are also possible.
.. [#] `Read the Docs`_ actually runs an instance of Sphinx to build the docs
       online before serving it.

.. _Documentation System by Divio : https://documentation.divio.com/
.. _Documentation System: https://documentation.divio.com/
.. _Pycon 2017: https://www.youtube.com/watch?v=azf6yzuJt54
.. _docs-like-code framework: https://www.docslikecode.com/
.. _same repository: https://gitlab.hzdr.de/interpol/minterpy
.. _Sphinx documentation generator: https://www.sphinx-doc.org/en/master/
.. _Read the Docs: https://readthedocs.org/
.. _reStructuredText (reST): https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _docstrings: https://www.python.org/dev/peps/pep-0257/
.. _PEP 287: https://www.python.org/dev/peps/pep-0287/
.. _Jupyter notebooks: https://jupyter.org/
.. _Markdown: https://daringfireball.net/projects/markdown/syntax
.. _a Wikipedia article: https://en.wikipedia.org/wiki/Encyclopedia

