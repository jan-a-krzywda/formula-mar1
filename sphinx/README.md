# Sphinx documentation

Auto-generated **Python API** reference (classes, functions, modules) via `sphinx.ext.autodoc`.

## Local build

From the **repository root**:

```bash
pip install -r sphinx/requirements.txt
sphinx-build -b html sphinx/source sphinx/build/html
touch sphinx/build/html/.nojekyll   # required for GitHub Pages (so _static/ is served)
```

Open `sphinx/build/html/index.html`.

Heavy runtime deps (**JAX**, **Flax**, **Optax**, etc.) are **mocked** in `source/conf.py` so the docs build without a GPU stack.

## Regenerate module `.rst` files

After adding or renaming top-level modules:

```bash
sphinx-apidoc -o sphinx/source -f -e . sphinx site runs dev __pycache__ .venv
```

Then remove `load_telemetry` from `modules.rst` if the stub module should stay undocumented, and edit `modules.rst` / `index.rst` as needed.
