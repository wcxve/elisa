# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from __future__ import annotations

import pathlib
import sys
from datetime import datetime

path = pathlib.Path(__file__)
sys.path.append(path.parent.parent.as_posix() + '/src')
import elisa  # noqa: E402

project = 'ELISA'
copyright = f'2023-{datetime.now().year}, Wang-Chen Xue & contributors'
release = elisa.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'numpydoc',
    'sphinx_autodoc_typehints',
    'sphinx_book_theme',
    'sphinx_copybutton',
    'sphinx_design',
    'myst_nb',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
}

templates_path = ['_templates']
exclude_patterns = ['_build']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_theme_options = {
    'github_url': 'https://github.com/wcxve/elisa',
    'repository_url': 'https://github.com/wcxve/elisa',
    'repository_branch': 'main',
    'path_to_docs': 'docs',
    'launch_buttons': {
        'binderhub_url': 'https://mybinder.org',
        'colab_url': 'https://colab.research.google.com/',
        'notebook_interface': 'jupyterlab',
    },
    'navigation_with_keys': False,
    'use_edit_page_button': True,
    'use_repository_button': True,
    'use_download_button': True,
    'use_issues_button': True,
}
html_title = 'ELISA'
html_static_path = ['_static']
html_logo = '_static/logo2.png'
html_favicon = '_static/favicon.svg'
html_baseurl = 'https://elisa-lib.readthedocs.io/en/latest/'
html_show_sourcelink = False
master_doc = 'index'

add_module_names = False
autodoc_member_order = 'bysource'

copybutton_selector = 'div:not(.output) > div.highlight pre'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3.9', None),
    'arviz': ('https://python.arviz.org/en/stable/', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'numpyro': ('https://num.pyro.ai/en/stable/', None),
    'tinygp': ('https://tinygp.readthedocs.io/en/latest/', None),
}

myst_enable_extensions = [
    'amsmath',
    'dollarmath',
    'colon_fence',
]

nb_ipywidgets_js = {
    'https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js': {
        'integrity': 'sha256-Ae2Vz/4ePdIu6ZyI/5ZGsYnb+m0JlOmKPjt6XZ9JJkA=',
        'crossorigin': 'anonymous',
    },
    'https://cdn.jsdelivr.net/npm/'
    '@jupyter-widgets/html-manager@*/dist/embed-amd.js': {
        'data-jupyter-widgets-cdn': 'https://cdn.jsdelivr.net/npm/',
        'crossorigin': 'anonymous',
    },
}
nb_execution_mode = 'off'
nb_execution_timeout = -1

numpydoc_attributes_as_param_list = False
numpydoc_class_members_toctree = False
numpydoc_show_class_members = True
numpydoc_show_inherited_class_members = True
numpydoc_xref_aliases = {
    'Data': 'elisa.data.ogip.Data',
    'Model': 'elisa.models.model.Model',
    'Parameter': 'elisa.models.parameter.Parameter',
}
numpydoc_xref_param_type = True
numpydoc_xref_ignore = {
    'or',
    'optional',
    'type_without_description',
    'BadException',
}
# Run docstring validation as part of build process
# numpydoc_validation_checks = {"all", "GL01", "SA04", "RT03"}
# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

typehints_document_rtype = False
typehints_use_signature = True
typehints_use_signature_return = True
