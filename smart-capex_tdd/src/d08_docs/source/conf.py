"""Configuration file for the Sphinx documentation builder"""
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import sys
import os

from pathlib import Path
from unittest.mock import MagicMock

parent = Path(__file__).parent
parents_parent = Path(__file__).parents[1]
#sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("../../.."))

#MOCK_MODULES = ['prophet']
#sys.modules.update((mod_name, MagicMock()) for mod_name in MOCK_MODULES)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Smart Capex TDD'
copyright = '2023, Orange'
author = 'Orange'
release = '0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.napoleon',
              "sphinx.ext.viewcode"]

templates_path = ['_templates']
exclude_patterns = []

#autodoc_mock_imports = ['src']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
