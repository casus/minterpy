# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Warning: do not change the path here. To use autodoc, you need to install the
# package first.

from typing import List

from pkg_resources import get_distribution

from sphinx.ext.autosummary import Autosummary
from sphinx.ext.autosummary import get_documenter
from docutils.parsers.rst import directives
from sphinx.util.inspect import safe_getattr

# -- Project information -----------------------------------------------------

project = "minterpy"
copyright = "2022, Minterpy development team"
author = "Uwe Hernandez Acosta"

version = get_distribution(project).version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.autosectionlabel",
    "sphinxcontrib.bibtex",
    "myst_nb",
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

# --- bibtex config
bibtex_bibfiles = ["bibliography.bib"]


# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["build", "**.ipynb_checkpoints", "Thumbs.db", ".DS_Store", ".env"]

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "autolink"

# Make sure label sections are unique
autosectionlabel_prefix_document = True

# Math configurations
math_eqref_format = 'Eq. ({number})'

# --- ToDo  options

# Display todos by setting to True
todo_include_todos = True

# --- Autodoc customization

# Don't sort the API elements alphabetically; instead, follow the source
autodoc_member_order = 'bysource'

# Don't expand/evaluate the default value in function signatures
autodoc_preserve_defaults = True

# Only show typehints in the functions/methods description, not signature
autodoc_typehints = "description"

# --- napoleon options
napoleon_use_rtype = True
napoleon_use_param = True

# --- Autosummary options
autosummary_generate = True
autosummary_imported_members = True

# --- MyST-NB options
myst_enable_extensions = [
   "amsmath",
   "dollarmath",  # Enable LaTeX-style dollar syntax for math.
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

html_title = f"Minterpy {version[0:3]}"

html_baseurl = "https://minterpy.readthedocs.io/en/latest/"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path: List[str] = []

html_favicon = './assets/favicon.ico'
html_logo = './assets/minterpy-logo.png'


# --- Custom directives -------------------------------------------------------

class ClassAutosummary(Autosummary):
    """Create a summary of class attributes, properties, and methods.

    Notes
    -----
    - Only public attributes, methods, and properties (do not start with
      "_") will be summarized.
    - Only class, not instance, attributes will be summarized. That means,
      attributes defined under `__init__` will not be visible.
    - This implementation is adapted from an `answer`_ in Stack Overflow. Note
      that the answer will not work out-of-the-box.

    .. _answer: https://stackoverflow.com/a/30783465
    """

    option_spec = {
        "attributes": directives.unchanged,  # Class, not instance, attribs
        "properties": directives.unchanged,
        "methods": directives.unchanged,
    }

    required_arguments = 1

    @staticmethod
    def get_members(app, obj, typ, include_public=None):
        if not include_public:
            include_public = []
        items = []
        for name in dir(obj):
            try:
                documenter = get_documenter(app, safe_getattr(obj, name), obj)
            except AttributeError:
                continue
            if documenter.objtype == typ:
                items.append(name)
        public = [
            x for x in items if x in include_public or not x.startswith("_")
        ]

        return public, items

    def run(self):
        """Execute run() method when the directive is used."""

        # Get the current Sphinx application attached to self
        try:
            app = self.state.document.settings.env.app
        except AttributeError:
            app = None

        cls = self.arguments[0]
        (module_name, class_name) = cls.rsplit(".", 1)
        m = __import__(module_name, globals(), locals(), [class_name])
        c = getattr(m, class_name)
        if "methods" in self.options:
            # Always include __init__
            _, methods = self.get_members(app, c, "method", ["__init__"])

            self.content = [
                f"~{cls}.{method}"
                for method in methods
                if not method.startswith("_")
            ]

        if "attributes" in self.options:
            _, attribs = self.get_members(app, c, "attribute")

            self.content = [
                f"~{cls}.{attrib}"
                for attrib in attribs
                if not attrib.startswith("_")
            ]

        if "properties" in self.options:
            _, props = self.get_members(app, c, "property")

            self.content = [
                f"~{cls}.{prop}" for prop in props if not prop.startswith("_")
            ]

        return super(ClassAutosummary, self).run()


def setup(app):
    try:
        app.add_directive("classautosummary", ClassAutosummary)
    except BaseException as e:
        raise e
