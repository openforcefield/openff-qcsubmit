# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/stable/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import openff.qcsubmit

sys.path.insert(0, os.path.abspath(os.pardir))


# -- Project information -----------------------------------------------------

project = "OpenFF QCSubmit"
copyright = "2021, Open Force Field Consortium"
author = "Open Force Field Consortium"

# The short X.Y version
version = openff.qcsubmit.__version__
# The full version, including alpha/beta/rc tags
release = openff.qcsubmit.__version__


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "nbsphinx",
    "nbsphinx_link",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.autodoc_pydantic",
    "myst_parser",
]

source_suffix = ".rst"

master_doc = "index"

language = 'en'

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

templates_path = ["_templates"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "default"

# Autodoc settings
autosummary_generate = True

autodoc_default_options = {
    "member-order": "bysource",
    "members": True,
    "inherited-members": "BaseModel",
}

autodoc_mock_imports = [
    "rdkit",
]

# Napoleon settings
napoleon_numpy_docstring = True
napoleon_use_rtype = False

# autodoc_pydantic settings
autodoc_pydantic_show_config = False
autodoc_pydantic_model_show_config = False
autodoc_pydantic_show_validators = False
autodoc_pydantic_model_show_validators = False

autodoc_typehints = "description"

# nbsphinx settings
nbsphinx_execute = "never"

# sphinx bibtext settings
bibtex_bibfiles = ["index.bib"]

# Set up the intershinx mappings.
intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "openff.toolkit": (
        "https://open-forcefield-toolkit.readthedocs.io/en/latest/",
        None,
    ),
    # Broken
    # "qcportal": ("http://docs.qcarchive.molssi.org/projects/qcportal/en/latest/", None),
    # "qcelemental": (
    #     "http://docs.qcarchive.molssi.org/projects/qcelemental/en/latest/",
    #     None,
    # ),
    "openff.docs": (
        "https://docs.openforcefield.org/en/latest/",
        None,
    ),
}

# Set up mathjax.
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"

myst_enable_extensions = [
    "deflist",
    "smartquotes",
    "replacements",
    "dollarmath",
    "colon_fence",
]

# sphinx-notfound-page
# https://github.com/readthedocs/sphinx-notfound-page
# Renders a 404 page with absolute links
import importlib

if importlib.util.find_spec("notfound"):
    extensions.append("notfound.extension")

    notfound_context = {
        "title": "404: File Not Found",
        "body": """
    <h1>404: File Not Found</h1>
    <p>
        Sorry, we couldn't find that page. This often happens as a result of
        following an outdated link. Please check the latest stable version
        of the docs, unless you're sure you want an earlier version, and
        try using the search box or the navigation menu on the left.
    </p>
    <p>
    </p>
    """,
    }

# -- Options for HTML output -------------------------------------------------


# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
extensions.append("openff_sphinx_theme")
html_theme = "openff_sphinx_theme"

html_sidebars = {
    "**": ["globaltoc.html", "localtoc.html", "searchbox.html"],
}

# Theme options are theme-specific and customize the look and feel of a
# theme further.
html_theme_options = {
    # Repository integration
    # Set the repo url for the link to appear
    "repo_url": "https://github.com/openforcefield/openff-qcsubmit",
    # The name of the repo. If must be set if repo_url is set
    "repo_name": "openff-qcsubmit",
    # Must be one of github, gitlab or bitbucket
    "repo_type": "github",
    # Colour for sidebar captions and other accents. One of
    # openff-blue, openff-toolkit-blue, openff-dataset-yellow,
    # openff-evaluator-orange, aquamarine, lilac, amaranth, grape,
    # violet, pink, pale-green, green, crimson, eggplant, turquoise,
    # or a tuple of three ints in the range [0, 255] corresponding to
    # a position in RGB space.
    "color_accent": "openff-dataset-yellow",
    "html_minify": False,
    "html_prettify": False,
    "css_minify": False,
}

html_static_path = ["_static"]

html_css_files = [
    "_static/css/theme_overrides.css",  # override wide tables in RTD theme
]

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "qcsubmitdoc"

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": "",
    "figure_align": "htbp",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "qcsubmit.tex", "OpenFF QCSubmit Documentation", author, "manual"),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, "openff-qcsubmit", "OpenFF QCSubmit Documentation", [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "openff-qcsubmit",
        "OpenFF QCSubmit Documentation",
        author,
        "openff-qcsubmit",
        "Automated tools for submitting molecules to QCFractal.",
        "Miscellaneous",
    ),
]
