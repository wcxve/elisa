(contributing)=

# Contributor Guide

Welcome to the contributor guide. Contributions are welcome in the form of bug
reports, documentation improvements, tests, new models, and core feature work.
This page summarizes the local setup and checks that match the current
repository layout.

## How to Contribute

### Report Bugs or Request Features

If you have found a bug or want to suggest an improvement, please open an
issue on the [GitHub issue tracker](https://github.com/wcxve/elisa/issues).
Useful reports usually include:

- a minimal reproducer;
- the exact error message or unexpected behavior;
- package versions and platform details;
- whether the issue depends on XSPEC, HEASoft, or external data files.

### Set Up a Development Environment

Create a clean environment and install the package in editable mode:

```console
conda create -n elisa-dev python=3.12
conda activate elisa-dev
pip install -e '.[dev,test,docs]'
```

If you plan to work on XSPEC-related functionality, install HEASoft / XSPEC
first, initialize the HEASoft environment, and then install the optional extra:

```console
pip install -e '.[xspec]'
```

To enable the repository hooks used by the project:

```console
pre-commit install
```

### Run Local Checks

The main local checks are:

```console
ruff check .
ruff format .
pytest
```

Some tests require external curated data. Those tests are skipped unless the
`CURATED_TEST_DATA` environment variable points to the dataset root:

```console
CURATED_TEST_DATA=/path/to/curated-test-data pytest
```

XSPEC tests are also skipped automatically when the `HEADAS` environment
variable is unavailable.

### Build the Documentation

Documentation source files live under `docs/`. To build the docs locally, run:

```console
bash docs/build.sh
```

This command regenerates the API reference and executes the notebooks as part
of the Sphinx build. Documentation-only changes are ignored by the main CI
workflow, so it is important to verify docs locally before opening a pull
request.

### Documentation and Notebook Conventions

- Edit source files such as `docs/*.md` and `docs/notebooks/*.ipynb`.
- Do not edit generated files under `docs/_build/`.
- Keep notebook examples lightweight and reproducible.
- The repository uses `nbstripout`, so notebook output may be removed when
  hooks run.

### Pull Request Checklist

Before opening a pull request, make sure that:

1. the change is scoped and described clearly;
2. relevant tests or documentation have been updated;
3. formatting and tests pass locally for the part you changed;
4. new public behavior is covered by at least one example, test, or doc note.

Small, focused pull requests are easier to review and merge than large mixed
changes.
