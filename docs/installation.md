(installation)=

# Installation Guide

## Stable Version

It is recommended to install ``ELISA`` in a new [``conda``](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html)
environment as follows:

1. Create a new ``conda`` environment. The following command creates a new
   environment named "elisa" with Python 3.9:

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
    pip install elisa-lib
    ```


## Use ``Xspec`` Models
   If you want to use models from [Xspec](https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/Models.html),
   make sure ``HEASoft`` and ``Xspec v12.12.1+`` are installed on your system, and the
   ``HEASoft`` environment is initialized, then use the following command to
   install [``xspex``](https://github.com/wcxve/xspex):

   ```console
   pip install xspex
   ```


## Development Version
The latest version of ``ELISA`` can be installed by the following command:

   ```console
   pip install -U git+https://github.com/wcxve/elisa.git
   ```
