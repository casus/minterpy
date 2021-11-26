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
  help us correct them.
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
(see :ref:`developers/contrib-doc/how-to-contrib:Creating an issue in the repository` below)
and describe briefly the reasoning.
This way we can track all the issues regarding the docs as well as your contributions.
Then you can wait until someone else reviews your suggestion
(and, if verified, does it for you);
or you may go ahead make the necessary changes yourself and do a merge request
(see :ref:`developers/contrib-doc/how-to-contrib:Contributing directly to the docs` below).

Before contributing
###################

Before making your contribution, we encourage you to consider the following:

- **Read about how** :doc:`the docs is designed </developers/contrib-doc/about-the-doc>`.
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
  As explained in the :ref:`About the Docs <developers/contrib-doc/about-the-doc:Tools and frameworks>`,
  we're using Sphinx to generate the `minterpy` docs.
  Most of the docs are written in reST, the default markup language for Sphinx.
  However, the tutorials and how-to guides are full of code examples
  and they are written instead as Jupyter notebooks to capture the outputs directly.
  Jupyter notebooks support `Markdown`_ markup language so you can include narrative text around the codes.
  Depending on which part of the documentation you'd like to contribute to,
  be sure to familiarize yourself with the mechanics of the underlying tools.

Docs source structure
######################

Building the docs locally
#########################

Creating an issue in the repository
###################################

Contributing directly to the docs
#################################


.. _issues labeled with docs: https://gitlab.hzdr.de/interpol/minterpy/-/issues?scope=all&state=opened&label_name[]=docs
.. _Issue Tracker: https://gitlab.hzdr.de/interpol/minterpy/-/issues
.. _Sphinx: https://www.sphinx-doc.org/en/master/
.. _reStructuredText (reST): https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _Jupyter notebooks: https://jupyter-notebook.readthedocs.io/en/stable/notebook.html
.. _Markdown: https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Working%20With%20Markdown%20Cells.html