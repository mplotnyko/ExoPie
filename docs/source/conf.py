# -*- coding: utf-8 -*-
import os
import sys
from importlib.metadata import PackageNotFoundError, version

# Path adjustment: Two levels up from docs/source/ to reach the project root
sys.path.insert(0, os.path.abspath("../.."))

try:
    __version__ = version("exopie")
except PackageNotFoundError:
    __version__ = "0.0.0" # Fallback if not installed

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "myst_nb",
    "IPython.sphinxext.ipython_console_highlighting",
]

myst_enable_extensions = ["dollarmath", "colon_fence"]
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".md": "myst-nb",
}

master_doc = "index"
project = 'ExoPie'
copyright = '2026, Mykhaylo Plotnykov'
author = 'Mykhaylo Plotnykov'
version = __version__
release = __version__

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = "exopie"
html_static_path = ["_static"] # Usually docs/source/_static/

html_theme_options = {
    "repository_url": "https://github.com/mplotnyko/exopie",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "docs/source", # Updated to match your folder structure
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "jupyterlab",
    },
}

nb_execution_mode = "off"
