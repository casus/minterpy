# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Warning: do not change the path here. To use autodoc, you need to install the
# package first.

from typing import List

from pkg_resources import get_distribution

# -- Project information -----------------------------------------------------

project = "minterpy"
copyright = "2021, Minterpy development team"
author = "Uwe Hernandez Acosta"

version = get_distribution(project).version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.autosectionlabel",
    "sphinxcontrib.bibtex",
]


# Intersphinx configuration
intersphinx_mapping = {
    "neps": ("https://numpy.org/neps", None),
    "python": ("https://docs.python.org/dev", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "pytest": ("https://docs.pytest.org/en/stable", None),
}

# bibtex config
bibtex_bibfiles = ["refs.bib"]

# configure numpydoc

numpydoc_show_class_members = False

# Display todos by setting to True
todo_include_todos = True

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints", "Thumbs.db", ".DS_Store", ".env"]

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "autolink"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_title = f"Minterpy {version[0:3]}"

html_baseurl = "https://minterpy.readthedocs.io/en/latest/"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path: List[str] = []
