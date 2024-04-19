# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import importlib.metadata
__version__ = importlib.metadata.version("torchgpe")

project = 'TorchGPE'
copyright = '2024, Quantum Optics group @ ETH Zurich'
author = 'Quantum Optics group @ ETH Zurich'
release = __version__


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx.ext.autosummary",
]

templates_path = ['_templates']
exclude_patterns = []

autodoc_typehints = "description"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']
html_css_files = ["style.css"]


html_theme_options = {
    "footer_items": ["copyright", "sphinx-and-theme-versions"],
    "navbar_end": ["theme-switcher","navbar-icon-links"],
    "secondary_sidebar_items": ["page-toc"],
    "navbar_persistent": [],
    "primary_sidebar_end": [],
    "logo": {
        "text": "TorchGPE package documentation"
   },
    "favicons": [
      {
         "rel": "icon",
         "sizes": "32x32",
         "href": "https://www.quantumoptics.ethz.ch/favicon.ico",
      }
    ],
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/torchgpe",
            "icon": "fa-solid fa-box",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/qo-eth/TorchGPE",
            "icon": "fa-brands fa-github",
        }
    ]
}

html_sidebars = {
    "**": ["searchbox.html", "sidebar-nav-bs"]
}

autosummary_generate = True
autodoc_default_flags = ['members', 'undoc-members']