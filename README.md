# elisa

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/elisa-lib?color=blue&logo=Python&logoColor=white&style=for-the-badge)](https://pypi.org/project/elisa-lib)
[![PyPI - Version](https://img.shields.io/pypi/v/elisa-lib?color=blue&logo=PyPI&logoColor=white&style=for-the-badge)](https://pypi.org/project/elisa-lib)
[![License: GPL v3](https://img.shields.io/github/license/wcxve/elisa?color=blue&logo=open-source-initiative&logoColor=white&style=for-the-badge)](https://www.gnu.org/licenses/gpl-3.0)<br>
[![Coverage Status](https://img.shields.io/coverallsCoverage/github/wcxve/elisa?logo=Coveralls&logoColor=white&style=for-the-badge)](https://coveralls.io/github/wcxve/elisa)
[![Documentation Status](https://img.shields.io/readthedocs/elisa-lib?logo=Read-the-Docs&logoColor=white&style=for-the-badge)](https://elisa-lib.readthedocs.io/en/latest/?badge=latest)

**An efficient library for spectral analysis in high-energy astrophysics.**

-----

**Table of Contents**

- [Installation](#installation)
- [Documentation](#documentation)
- [License](#license)

## Installation

### Stable Version

It is recommended to install `elisa` as follows:

1. Create a new `conda` environment. The following command creates a new
   environment named "elisa" with Python 3.9:

    ```console
    conda create -n elisa python=3.9
    ```

2. Activate the environment:

    ```console
    conda activate elisa
    ```

3. Install `elisa` using `pip`:

    ```console
    pip install elisa-lib
    ```

   If you want to use models of [Xspec](https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/Models.html),
   make sure `HEASoft` and `Xspec` are installed on your system, and the
   `HEASoft` environment is initialized, then use the following command to
   install `elisa`:

    ```console
    pip install elisa-lib[xspec]
    ```


### Development Version
The latest version of `elisa` can be installed by the following command:

   ```console
   pip install -U git+https://github.com/wcxve/elisa.git
   ```


## Documentation

Read the documentation at: https://elisa-lib.readthedocs.io

## License

`elisa` is distributed under the terms of the [GPL-3.0](https://www.gnu.org/licenses/gpl-3.0-standalone.html) license.
