import os
import sys
# This tells Sphinx to look two folders up (the root of your repository) to find the 'exopie' package
sys.path.insert(0, os.path.abspath('../..'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ExoPie'
copyright = '2026, Mykhaylo Plotnykov'
author = 'Mykhaylo Plotnykov'
release = '2.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'sphinx.ext.autodoc',  # For pulling docstrings from your code
    'sphinx.ext.napoleon', # If you use Google or NumPy style docstrings
    'sphinx.ext.mathjax',  # For rendering math in notebooks
    'nbsphinx',            # The notebook converter
    'myst_parser',         # markdown parser
]

# Change the HTML theme
html_theme = 'sphinx_rtd_theme'

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
