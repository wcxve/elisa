(tutorials)=

# Tutorials

The tutorials below are intended to be read in order, from the first complete
fit to more advanced model construction topics.

## Before You Begin

Make sure you have completed the {ref}`installation` steps first. The notebook
examples assume that ``ELISA`` imports successfully and that the optional data
files shipped under `docs/notebooks/data/` are available.

## Tutorial Map

- {ref}`quick-start`: load OGIP data, define a spectral model, run Bayesian
  inference with NUTS, and compare it with a maximum-likelihood fit.
- {ref}`model-building`: learn how additive, multiplicative, and convolution
  components are combined, and how model parameters can be configured or
  linked.
- {ref}`custom-model`: implement your own spectral components when built-in or
  XSPEC models are not enough.

## Suggested Reading Order

If you are new to the project, begin with {ref}`quick-start`. Once the basic
fit workflow is clear, continue to {ref}`model-building` to understand how
models are assembled. Finish with {ref}`custom-model` if you need to wrap
project-specific physics or legacy numerical code.

```{toctree}
:maxdepth: 1

notebooks/quick-start
notebooks/model-building
notebooks/custom-model
```
