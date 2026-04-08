(guide)=

# User Guide

The user guide collects the practical information needed to install
``ELISA``, prepare a working environment, and troubleshoot common problems.
If you would rather start from an end-to-end example, jump directly to the
{ref}`tutorials`.

## Typical Workflow

Most analyses with ``ELISA`` follow the same pattern:

1. Install the package and optional extras such as XSPEC support.
2. Load OGIP-compliant spectral data with {class}`elisa.data.ogip.Data`.
3. Build a spectral model from additive, multiplicative, and convolution
   components.
4. Fit the model with either {class}`elisa.infer.fit.BayesFit` or
   {class}`elisa.infer.fit.MaxLikeFit`.
5. Inspect the result using tables, posterior summaries, and diagnostic plots.

The {ref}`quick-start` tutorial walks through this full sequence on real
multi-instrument X-ray data.

## What To Read Next

- Start with {ref}`installation` if you have not set up an environment yet.
- Go to {ref}`tutorials` for worked examples, including model construction and
  custom components.
- Use {ref}`troubleshooting` when data loading, environment setup, or fitting
  does not behave as expected.
- Browse the {ref}`api` reference when you need the full callable signature or
  class-level documentation.

```{toctree}
:maxdepth: 1

installation
troubleshooting
```
