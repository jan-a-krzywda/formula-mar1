# Sphinx configuration — Formula Mar1 (https://jan-a-krzywda.github.io/formula-mar1/)
# Build: pip install -r ../requirements.txt && sphinx-build -b html . ../build/html

import os
import sys

_conf_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_conf_dir, "..", ".."))
sys.path.insert(0, _repo_root)

project = "Formula Mar1"
copyright = "2026, Jan A. Krzywda"
author = "Jan A. Krzywda"

release = "0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# GitHub Project Pages — used for canonical URLs in meta tags
html_baseurl = "https://jan-a-krzywda.github.io/formula-mar1/"

autodoc_mock_imports = [
    "jax",
    "jax.numpy",
    "jax.random",
    "jax.nn",
    "jax.lax",
    "flax",
    "flax.linen",
    "optax",
    "tensorboardX",
    "matplotlib",
    "matplotlib.pyplot",
    "PIL",
    "PIL.Image",
    "PIL.ImageDraw",
    "PIL.ImageFont",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
