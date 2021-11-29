#############################
How to Contribute to the Docs
#############################

This page describes the ways in you can contribute to the ``minterpy`` docs.

What to contribute
##################

We value a useful, readable, and consistent docs;
so we value your contributions to it, small and big.
Below are some examples of how you can join and help us maintain
and improve the ``minterpy`` docs:

- **General language improvements**: If you are disturbed by typos, grammatical
  mistakes, or imprecise and misleading writing (we are as well!),
  help us correct them when you find them in the docs.
- **Fixing technical errors**: If you find technical errors in the docs
  such as downright wrong statements, broken links, wrong usage examples,
  or missing parameters in the API reference, let us know about them.
- **Adding new tutorials and how-to guides**: If you think the current docs
  is missing a relevant tutorial or how-to guide, go ahead and suggest it to us.
  Or, you can also create one yourself and contribute to it.
- **Documenting new features**: If you're developing a new feature for ``minterpy``,
  we strongly encourage you to write the docs yourself including some tutorials
  and how-to guides. Note that your eventual feature contribution to the main
  branch of ``minterpy`` *must* include the docs. The reviews are carried out
  both for the code and the docs; we value them equally.

When you have a suggestion or thing you want to do to improve the docs,
you should open an issue in the ``minterpy`` repository
(see :ref:`contributors/contrib-docs/how-to-contrib:Creating an issue in the repository` below)
and describe briefly the reasoning.
This way we can track all the issues regarding the docs as well as your contributions.

Then you can wait until someone else reviews your suggestion
(and, if verified, does it for you);
or you may go ahead make the necessary changes yourself and do a merge request
(see :ref:`contributors/contrib-docs/how-to-contrib:Contributing directly to the docs` below).

Before contributing
###################

Before making your contribution, we encourage you to consider the following:

- **Read about how** :doc:`the docs is designed </contributors/contrib-docs/about-the-docs>`.
  You'll notice that we try to consistently distinguish four categories of docs:
  tutorials, how-to guides, fundamentals, and API reference.
  We build the docs around these four categories.
  If you're thinking of adding new content to the docs, think about in which
  category your contribution will fall.
- **Read through some of the existing docs including the source code**.
  This way you can get a feeling of how things are currently organized.
  By looking at the docs source code,
  you'll get an idea of how a page is structured and the content styled.
- **Check the** `issues labeled with docs`_
  in the ``minterpy`` repository `Issue Tracker`_;
  perhaps someone else has opened an issue similar to yours.
  Second or comment on the issue.
- **Familiarize yourself with** `Sphinx`_, `reStructuredText (reST)`_,
  and `Jupyter notebooks`_.
  As explained in the :ref:`About the Docs <contributors/contrib-docs/about-the-docs:Frameworks and tools>`,
  we're using Sphinx to generate the ``minterpy`` docs.
  Most of the docs are written in reST, the default markup language for Sphinx.
  However, the tutorials and how-to guides are full of code examples
  and they are written instead as Jupyter notebooks to capture the outputs directly.
  Jupyter notebooks support `Markdown`_ markup language so you can include narrative text around the codes.
  Depending on which part of the documentation you'd like to contribute to,
  be sure to familiarize yourself with the mechanics of the underlying tools.

Docs source structure
#####################

The ``minterpy`` docs source is part of the ``minterpy`` repository
and stored in the |docs directory|_.
The docs consists of five main (top-level) sections:

- The :doc:`Getting Started Guides </getting-started/index>` (``getting-started``)
- The :doc:`How-to Guides </how-to/index>` (``how-to``)
- The :doc:`Fundamentals </fundamentals/index>` (``fundamentals``)
- The :doc:`API Reference </api/index>` (``api``)
- The :doc:`Contributors Guide </contributors/index>` (``contributors``)

These five sections correspond to the five top-level directories inside the ``docs`` directory.

.. important::

   The main sections are meant to be stable;
   changes at this level change the hierarchy of the information we'd like to present to the readers
   and may require layout modifications as well.
   A new top-level directories within the ``docs`` should not be added without consulting
   the ``minterpy`` project maintainers.

Inside each section, contents are organized into *Subsections* and *Pages*.
*Pages* are individual reStructuredText (reST) document files (with an ``.rst`` extension);
*Subsections* are directories that group topically-related pages together
and correspond to subsections within one of the top-level sections.

For example, the :doc:`/contributors/index` has subsections such as
:doc:`/contributors/contrib-dev/index` and :doc:`/contributors/contrib-docs/index`,
and individual pages such as :doc:`/contributors/about-us`
and :doc:`/contributors/code-of-conduct`.
The docs source reflect that structure as shown below:

.. code-block::

   docs
   |--- api
   |--- contributors
   |    |--- contrib-dev
   |    |--- contrib-docs
   |    |--- about-us.rst
   |    |--- code-of-conduct.rst
   |    |--- index.rst
   ...

The file ``index.rst`` in the top ``docs`` directory is the main index (root)
file of the docs.
This file defines what you see when you navigate
to the ``minterpy`` :doc:`docs homepage </index>`.
Each of the main sections also has its own index file that serves
as the main page of the section;
it lists all the pages that belong to that section.
Some of the subsections inside the main sections may contain
their own index file as well.

Building the docs locally
#########################

To build the ``minterpy`` docs locally,
make sure you've cloned a version of ``minterpy`` from the repository,
installed it in your system, including all the requirements for the docs.
You can install the docs requirements from your local ``minterpy`` source directory
by:

.. code-block:: bash

   $ pip install .[docs]

in Linux or macOS, and:

.. code-block::

   ..:\> pip install .[docs]

in Windows.

Then from the ``docs`` directory, build the HTML docs by invoking the ``make`` command:

.. code-block:: bash

   $ make html

in Linux or macOS, and:

.. code-block::

   ..:\> make.bat html

in Windows.

If you're making modifications to the docs, you need to invoke
the ``make`` command every time.
It may be useful to clean the built directory from time to time by invoking
the following command from the ``docs`` directory:

.. code-block:: bash

   $ make clean

in Linux or macOS, and:

.. code-block::

   ..:\> make.bat clean

in Windows.

.. tip::

   If you don't see your source modifications (after saving them) in the HTML docs,
   do this cleaning as a first troubleshooting step.

While you're working with the docs, you might prefer to have a live-reload.
You can use `sphinx-autobuild`_ (part of the docs requirements) to automatically
rebuild the HTML docs on source changes.
You need to invoke the following command
from the main ``minterpy`` source directory
(*not* from the ``docs`` directory; it's one level above it):

.. code-block:: bash

   $ sphinx-autobuild docs ./docs/build/html

in Linux or mac OS, and:

.. code-block::

   ..:\> sphinx-autobuild docs .\docs\build\html

in Windows.

This will start a local server accessible at the shown address;
open the address in your web browser.
sphinx-autobuild watches for any changes in the ``docs`` directory;
when it detects them, it rebuilds the docs automatically.

Creating an issue in the repository
###################################

Whether you have a suggestion about the docs or you want to change the docs directly,
you should start by creating a new issue in the `Issue Tracker`_ of the ``minterpy`` `repository`_.

.. figure:: /assets/images/contributors/issue-tracker-docs.png
  :align: center

  Open a new issue regarding the docs in the `Issue Tracker`_;
  make sure you've checked the already created issues regarding the docs by
  using the label filter.

Provide your issue with a descriptive title and then fill in the description
of the issue with the following (at the very least):

- The docs section (and, when applicable, the page) where the problem occurs
  or your suggestions apply to.
- A proposed solution.

.. figure:: /assets/images/contributors/new-issue-docs.png
  :align: center

  Fill in the new issue form; give a descriptive title, briefly describe
  the problem or your suggestions about the doc, and write a possible solution.

Finally, don't forget to assign "Issue" as the **Type** and "Docs" as the **Label**
before you click on the **Create Issue** button.

.. tip::

   Creating an issue does not mean you're responsible for actually doing it
   (unless you want to)!
   Someone else in the project will verify your issue
   and, if verified, does it for you.
   If you want to do it,
   then assign the issue to yourself using the **Assignee** field.

Contributing directly to the docs
#################################

.. TODO::

   This section is still empty. It should synchronize well with the
   contribution to the codebase because they are very similar.
   Refer as much as possible to the corresponding section in the contribution
   to the development.

.. _issues labeled with docs: https://gitlab.hzdr.de/interpol/minterpy/-/issues?scope=all&state=opened&label_name[]=docs
.. _Issue Tracker: https://gitlab.hzdr.de/interpol/minterpy/-/issues
.. _Sphinx: https://www.sphinx-doc.org/en/master/
.. _reStructuredText (reST): https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _Jupyter notebooks: https://jupyter-notebook.readthedocs.io/en/stable/notebook.html
.. _Markdown: https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Working%20With%20Markdown%20Cells.html
.. |docs directory| replace:: ``docs`` directory
.. _docs directory: https://gitlab.hzdr.de/interpol/minterpy/-/tree/dev/docs
.. _sphinx-autobuild: https://github.com/executablebooks/sphinx-autobuild
.. _repository: https://gitlab.hzdr.de/interpol/minterpy