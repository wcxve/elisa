---
html_theme.sidebar_secondary.remove:
---

# ELISA: Efficient Library for Spectral Analysis in High-Energy Astrophysics

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/astro-elisa?color=blue&logo=Python&logoColor=white&style=for-the-badge)](https://pypi.org/project/astro-elisa)
[![PyPI - Version](https://img.shields.io/pypi/v/astro-elisa?color=blue&logo=PyPI&logoColor=white&style=for-the-badge)](https://pypi.org/project/astro-elisa)
[![License: GPL v3](https://img.shields.io/github/license/wcxve/elisa?color=blue&logo=open-source-initiative&logoColor=white&style=for-the-badge)](https://www.gnu.org/licenses/gpl-3.0)<br>
[![Coverage Status](https://img.shields.io/codecov/c/github/wcxve/elisa?logo=Codecov&logoColor=white&style=for-the-badge)](https://app.codecov.io/github/wcxve/elisa)
[![Documentation Status](https://img.shields.io/readthedocs/astro-elisa?logo=Read-the-Docs&logoColor=white&style=for-the-badge)](https://astro-elisa.readthedocs.io/en/latest/?badge=latest)

``ELISA`` aims to provide a modern and efficient tool to explore and
analyze the spectral data. It is designed to be user-friendly and flexible.
The key features of ``ELISA`` include:

- **Ease of Use**: Simple and intuitive interfaces
- **Robustness**: Utilizing the state-of-the-art algorithm to fit, test, and compare models
- **Performance**: Efficient computation backend based on [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)
- **Data Support**: Direct handling of OGIP-style spectra, backgrounds, and responses
- **Flexible Modeling**: Composition of built-in, XSPEC, and custom spectral components
- **Diagnostics**: Built-in visualization and summary tools for fits, residuals, and posterior distributions

```{admonition} How to find your way around?
:class: tip

🖥️ Ready to give it a try? Start with the {ref}`installation`.

📚 Curious about the details? The {ref}`tutorials` have you covered, including
the {ref}`api`.

💡 Encountering issues? See the {ref}`troubleshooting` page for helpful tips
and tricks.

🐛 If the {ref}`troubleshooting` section doesn’t clear things up, or if you
stumble upon bugs, we’d love your input! Check out our {ref}`contributing`
section and share your findings on the
[GitHub issues page](https://github.com/wcxve/elisa/issues).
```

**NOTE**: The documentation evolves together with the package. If you find a
gap, unclear explanation, or missing example, please open an issue on the
[GitHub issues page](https://github.com/wcxve/elisa/issues).

```{toctree}
:maxdepth: 2
:hidden:

guide
tutorials
contributing
api
GitHub Repository <https://github.com/wcxve/elisa>
```
