# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import numpy
import numpyro
import jax
sys.path.insert(0, os.path.abspath('../..'))
import bayespec


project = 'BayeSpec'
copyright = '2023, Wang-Chen Xue & contributors'
author = 'Wang-Chen Xue'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'numpydoc.numpydoc'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autodoc_mock_imports = ['jax', 'numpyro']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# import sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static']
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
