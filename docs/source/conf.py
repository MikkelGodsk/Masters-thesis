# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Master's thesis"
copyright = '2024, Mikkel Godsk Jørgensen'
author = 'Mikkel Godsk Jørgensen'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',     # Automatically document all modules and classes
    'sphinx.ext.napoleon',    # Allows for Google-style docstrings
]

templates_path = ['_templates']
exclude_patterns = ['Thumbs.db', '.DS_Store', 'setup', 'setup.py']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme' #'alabaster'
html_static_path = ['_static']


import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # It needs to go up two levels to find the package, because I said "separate source and build" in the setup... Otherwise only one level.


os.environ['SPHINX_APIDOC_OPTIONS']='members,show-inheritance'   # Otherwise it gives a warning about `WARNING: duplicate object description`
## Setting up Sphinx required a bit of help from ChatGPT🤖