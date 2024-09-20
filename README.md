# ELISA: Efficient Library for Spectral Analysis in High-Energy Astrophysics

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/astro-elisa?color=blue&logo=Python&logoColor=white&style=for-the-badge)](https://pypi.org/project/astro-elisa)
[![PyPI - Version](https://img.shields.io/pypi/v/astro-elisa?color=blue&logo=PyPI&logoColor=white&style=for-the-badge)](https://pypi.org/project/astro-elisa)
[![License: GPL v3](https://img.shields.io/github/license/wcxve/elisa?color=blue&logo=open-source-initiative&logoColor=white&style=for-the-badge)](https://www.gnu.org/licenses/gpl-3.0)<br>
[![Coverage Status](https://img.shields.io/coverallsCoverage/github/wcxve/elisa?logo=Coveralls&logoColor=white&style=for-the-badge)](https://coveralls.io/github/wcxve/elisa)
[![Documentation Status](https://img.shields.io/readthedocs/astro-elisa?logo=Read-the-Docs&logoColor=white&style=for-the-badge)](https://astro-elisa.readthedocs.io/en/latest/?badge=latest)

``ELISA`` aims to provide a modern and efficient tool to explore and
analyze the spectral data. It is designed to be user-friendly and flexible.
The key features of ``ELISA`` include:

- **Ease of Use**: Simple and intuitive interfaces
- **Robustness**: Utilizing the state-of-the-art algorithm to fit, test, and compare models
- **Performance**: Efficient computation backend based on [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)
- ...

**NOTE**: ``ELISA`` is currently under active development. Please be aware of
potential brittleness, bugs, and changes to the API as the design evolves.

-----

**Table of Contents**

- [Installation](#installation)
- [Documentation](#documentation)
- [License](#license)

## Installation

### Stable Version

It is recommended to install ``ELISA`` in a new [``conda``](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html)
environment as follows:

1. Create a new ``conda`` environment. The following command creates a new
   environment named "elisa" with ``Python`` 3.9:

    ```console
    conda create -n elisa python=3.9
    ```

   Note that you can customize the environment name to your preference,
   and the ``Python`` version should range from 3.9 to 3.11.

2. Activate the environment we just created:

    ```console
    conda activate elisa
    ```

3. Install ``ELISA`` using ``pip``:

    ```console
    pip install astro-elisa
    ```


## Use ``Xspec`` Models
   If you want to use models from [Xspec](https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/Models.html),
   make sure ``HEASoft`` and ``Xspec v12.12.1+`` are installed on your system,
   and the ``HEASoft`` environment is initialized, then use the following
   command to install [``xspex``](https://github.com/wcxve/xspex):

   ```console
   pip install xspex
   ```


### Development Version
The latest version of ``ELISA`` can be installed by the following command:

   ```console
   pip install -U git+https://github.com/wcxve/elisa.git
   ```


## Documentation

Read the documentation at: https://astro-elisa.readthedocs.io

## License

`ELISA` is distributed under the terms of the [GPL-3.0](https://www.gnu.org/licenses/gpl-3.0-standalone.html) license.
