#################################
Getting started with ``minterpy``
#################################

..
    .. todo::

       Getting started is the most important part of the documentation for new users
       and developers. It should provide users what is it like to use ``minterpy``
       to solve interpolation problems.
       This chapter might be broken into parts with increasing complexity,
       progressively disclosing more difficult or specific terms related to interpolation problems or ``minterpy``  itself.

       For example:

       - Part 1: Installation
       - Part 2: One-dimensional interpolation (make sure users are convinced that it works for the most simple problems while introducing ``minterpy`` basic usage, say ``mp.interpolate``)
       - Part 3: Multi-dimensional interpolation (it works for one-dimension, it also works for higher-dimensions, introduce other aspects of ``minterpy`` not apparent in one-dimensional problem)
       - Part 4: Interpolator objects (this is also an exposed interface for users, so it might be good idea to introduce how to create and handle such objects)

Have you installed ``minterpy``?
Read on how to install it before moving on.

Installation
############

Since this implementation is a prototype,
we currently only provide the installation by self-building from source.
We recommend to use ``git`` to get the ``minterpy`` source:

.. code-block:: bash

   git clone https://gitlab.hzdr.de/interpol/minterpy.git

Within the source directory,
you may use the following package manager to install ``minterpy``.

A best practice is to create a virtual environment for `minterpy`.
You can do this with the help of `conda`_ and the ``environment.yaml`` by:

.. code-block::

   conda env create -f environment.yaml


A new conda environment called ``minterpy`` is created.
Activate the new environment by:

.. code-block::

   conda activate minterpy

From within the environment, install the ``minterpy`` using `pip`_:

.. code-block::

   pip install [-e] .[all,dev,docs]

where the flag ``-e`` means the package is directly linked
into the python site-packages of your Python version.
The options ``[all,dev,docs]`` refer to the requirements defined
in the ``options.extras_require`` section in ``setup.cfg``.

You **must not** use the command :code:`python setup.py install` to install `minterpy`,
as you cannot always assume the files ``setup.py`` will always be present
in the further development of ``minterpy``.

Finally, if you want to deactivate the conda environment, type:

.. code-block::

   conda deactivate

Testing the installation
========================

After installation, we encourage you to at least run the unit tests of ``minterpy``,
where we use `pytest`_ to run the tests.

If you want to run all tests, type:

.. code-block:: bash

   pytest [-vvv]

from within the ``minterpy`` source directory.

What's next
###########

The best way to learn about ``minterpy`` is to read and play around
with the tutorials below.

Once you're more familiar with ``minterpy`` and would like to achieve a particular task,
be sure to check out the :doc:`/how-to/index`!

Tutorials table of contents
###########################

.. toctree::
   :maxdepth: 3

   one-dimensional-function-interpolation
   polynomial-regression

.. _conda: https://docs.conda.io/
.. _pip: https://pip.pypa.io/en/stable/
.. _pytest: https://docs.pytest.org/en/6.2.x/