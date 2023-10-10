import sys
from datetime import datetime
from functools import partial
from typing import Dict, Literal, Optional, Sequence, Union

import arviz as az
import jax
import numpy as np
import numpyro
import xarray as xr
from arviz.data.base import make_attrs
from numpyro.infer import MCMC, NUTS
from pymc import Model, modelcontext
from pymc.backends.arviz import (
    coords_and_dims_for_inferencedata, find_constants, find_observations
)
from pymc.initial_point import StartDict
from pymc.sampling.jax import (
    get_jaxified_graph,
    get_jaxified_logp,
    _get_batched_jittered_initial_points,
    _get_log_likelihood,
    _postprocess_samples,
    _sample_stats_to_xarray,
    _update_coords_and_dims,
    _update_numpyro_nuts_kwargs,
)
from pymc.util import (
    RandomState,
    _get_seeds_per_chain,
    get_default_varnames,
)

__all__ = ['sample_numpyro_nuts']


def sample_numpyro_nuts(
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.8,
    random_seed: Optional[RandomState] = None,
    initvals: Optional[Union[StartDict, Sequence[Optional[StartDict]]]] = None,
    jitter: bool = True,
    model: Optional[Model] = None,
    var_names: Optional[Sequence[str]] = None,
    progressbar: bool = True,
    keep_untransformed: bool = False,
    chain_method: str = "parallel",
    postprocessing_backend: Literal["cpu", "gpu"] | None = None,
    postprocessing_vectorize: Literal["vmap", "scan"] = "scan",
    idata_kwargs: Optional[Dict] = None,
    nuts_kwargs: Optional[Dict] = None,
) -> az.InferenceData:
    """
    Draw samples from the posterior using the NUTS method from the ``numpyro`` library.

    Parameters
    ----------
    draws : int, default 1000
        The number of samples to draw. The number of tuned samples are discarded by
        default.
    tune : int, default 1000
        Number of iterations to tune. Samplers adjust the step sizes, scalings or
        similar during tuning. Tuning samples will be drawn in addition to the number
        specified in the ``draws`` argument.
    chains : int, default 4
        The number of chains to sample.
    target_accept : float in [0, 1].
        The step size is tuned such that we approximate this acceptance rate. Higher
        values like 0.9 or 0.95 often work better for problematic posteriors.
    random_seed : int, RandomState or Generator, optional
        Random seed used by the sampling steps.
    initvals: StartDict or Sequence[Optional[StartDict]], optional
        Initial values for random variables provided as a dictionary (or sequence of
        dictionaries) mapping the random variable (by name or reference) to desired
        starting values.
    jitter : bool, optional
        Whether to use jittered initial vals.
    model : Model, optional
        Model to sample from. The model needs to have free random variables. When inside
        a ``with`` model context, it defaults to that model, otherwise the model must be
        passed explicitly.
    var_names : sequence of str, optional
        Names of variables for which to compute the posterior samples. Defaults to all
        variables in the posterior.
    progressbar : bool, default True
        Whether to display a progress bar in the command line. The bar shows the
        percentage of completion, the sampling speed in samples per second (SPS), and
        the estimated remaining time until completion ("expected time of arrival"; ETA).
    keep_untransformed : bool, default False
        Include untransformed variables in the posterior samples. Defaults to False.
    chain_method : str, default "parallel"
        Specify how samples should be drawn. The choices include "sequential",
        "parallel", and "vectorized".
    postprocessing_backend : Optional[str]
        Specify how postprocessing should be computed. gpu or cpu
    postprocessing_vectorize: Literal["vmap", "scan"], default "scan"
        How to vectorize the postprocessing: vmap or sequential scan
    idata_kwargs : dict, optional
        Keyword arguments for :func:`arviz.from_dict`. It also accepts a boolean as
        value for the ``log_likelihood`` key to indicate that the pointwise log
        likelihood should not be included in the returned object. Values for
        ``observed_data``, ``constant_data``, ``coords``, and ``dims`` are inferred from
        the ``model`` argument if not provided in ``idata_kwargs``. If ``coords`` and
        ``dims`` are provided, they are used to update the inferred dictionaries.
    nuts_kwargs: dict, optional
        Keyword arguments for :func:`numpyro.infer.NUTS`.

    Returns
    -------
    InferenceData
        ArviZ ``InferenceData`` object that contains the posterior samples, together
        with their respective sample stats and pointwise log likeihood values (unless
        skipped with ``idata_kwargs``).
    """
    model = modelcontext(model)

    if var_names is None:
        var_names = model.unobserved_value_vars

    vars_to_sample = list(get_default_varnames(var_names, include_transformed=keep_untransformed))

    coords = {
        cname: np.array(cvals) if isinstance(cvals, tuple) else cvals
        for cname, cvals in model.coords.items()
        if cvals is not None
    }

    dims = {
        var_name: [dim for dim in dims if dim is not None]
        for var_name, dims in model.named_vars_to_dims.items()
    }

    (random_seed,) = _get_seeds_per_chain(random_seed, 1)

    tic1 = datetime.now()
    print("Compiling...", file=sys.stdout)

    init_params = _get_batched_jittered_initial_points(
        model=model,
        chains=chains,
        initvals=initvals,
        random_seed=random_seed,
        jitter=jitter
    )

    logp_fn = get_jaxified_logp(model, negative_logp=False)

    nuts_kwargs = _update_numpyro_nuts_kwargs(nuts_kwargs)
    nuts_kernel = NUTS(
        potential_fn=logp_fn,
        target_accept_prob=target_accept,
        **nuts_kwargs,
    )

    pmap_numpyro = MCMC(
        nuts_kernel,
        num_warmup=tune,
        num_samples=draws,
        num_chains=chains,
        postprocess_fn=None,
        chain_method=chain_method,
        progress_bar=progressbar,
    )

    tic2 = datetime.now()
    print("Compilation time = ", tic2 - tic1, file=sys.stdout)

    print("Sampling...", file=sys.stdout)

    map_seed = jax.random.PRNGKey(random_seed)
    if chains > 1:
        map_seed = jax.random.split(map_seed, chains)

    pmap_numpyro.run(
        map_seed,
        init_params=init_params,
        extra_fields=(
            "num_steps",
            "potential_energy",
            "energy",
            "adapt_state.step_size",
            "accept_prob",
            "diverging",
        ),
    )

    raw_mcmc_samples = pmap_numpyro.get_samples(group_by_chain=True)

    tic3 = datetime.now()
    print("Sampling time = ", tic3 - tic2, file=sys.stdout)

    print("Transforming variables...", file=sys.stdout)
    jax_fn = get_jaxified_graph(inputs=model.value_vars, outputs=vars_to_sample)
    result = _postprocess_samples(
        jax_fn, raw_mcmc_samples, postprocessing_backend, postprocessing_vectorize
    )
    mcmc_samples = {v.name: r for v, r in zip(vars_to_sample, result)}

    tic4 = datetime.now()
    print("Transformation time = ", tic4 - tic3, file=sys.stdout)

    if idata_kwargs is None:
        idata_kwargs = {}
    else:
        idata_kwargs = idata_kwargs.copy()

    if idata_kwargs.pop("log_likelihood", False):
        tic5 = datetime.now()
        print("Computing Log Likelihood...", file=sys.stdout)
        log_likelihood = _get_log_likelihood(
            model,
            raw_mcmc_samples,
            backend=postprocessing_backend,
            postprocessing_vectorize=postprocessing_vectorize,
        )
        tic6 = datetime.now()
        print("Log Likelihood time = ", tic6 - tic5, file=sys.stdout)
    else:
        log_likelihood = None

    attrs = {
        "sampling_time": (tic3 - tic2).total_seconds(),
    }

    # posterior = mcmc_samples
    # # Update 'coords' and 'dims' extracted from the model with user 'idata_kwargs'
    # # and drop keys 'coords' and 'dims' from 'idata_kwargs' if present.
    # _update_coords_and_dims(coords=coords, dims=dims, idata_kwargs=idata_kwargs)
    # # Use 'partial' to set default arguments before passing 'idata_kwargs'
    # to_trace = partial(
    #     az.from_dict,
    #     log_likelihood=log_likelihood,
    #     observed_data=find_observations(model),
    #     constant_data=find_constants(model),
    #     sample_stats=_sample_stats_to_xarray(pmap_numpyro),
    #     coords=coords,
    #     dims=dims,
    #     attrs=make_attrs(attrs, library=numpyro),
    # )
    # az_trace = to_trace(posterior=posterior, **idata_kwargs)
    coords, dims = coords_and_dims_for_inferencedata(model)
    # Update 'coords' and 'dims' extracted from the model with user 'idata_kwargs'
    # and drop keys 'coords' and 'dims' from 'idata_kwargs' if present.
    _update_coords_and_dims(coords=coords, dims=dims, idata_kwargs=idata_kwargs)
    # Use 'partial' to set default arguments before passing 'idata_kwargs'
    to_trace = partial(
        az.from_dict,
        log_likelihood=log_likelihood,
        observed_data=find_observations(model),
        constant_data=find_constants(model),
        sample_stats=_sample_stats_to_xarray(pmap_numpyro),
        coords=coords,
        dims=dims,
        attrs=make_attrs(attrs, library=numpyro),
    )
    az_trace = to_trace(posterior=mcmc_samples, **idata_kwargs)
    return az_trace


def sample_posterior_predictive(idata, statistics_dict, seed):
    rng = np.random.default_rng(seed)

    data_set = xr.Dataset(coords=idata['log_likelihood'].coords)

    all_channel = []

    for name, stat in statistics_dict.items():
        coords = ('chain', 'draw', f'{name}_channel')

        mu_on = idata['posterior'][f'{name}_TOTAL']

        if stat == 'chi':
            sigma = idata['constant_data'][f'{name}_spec_error']
            spec_counts = rng.normal(mu_on, sigma)

            all_channel.append(spec_counts)
            data_set[f'{name}_Non'] = (coords, spec_counts)
            data_set[f'{name}_Net'] = (coords, spec_counts)

        elif stat == 'cstat':
            spec_counts = rng.poisson(mu_on).astype(np.float64)

            all_channel.append(spec_counts)
            data_set[f'{name}_Non'] = (coords, spec_counts)
            data_set[f'{name}_Net'] = (coords, spec_counts)

        elif stat == 'pstat':
            spec_counts = rng.poisson(mu_on).astype(np.float64)
            back_counts = np.tile(
                idata.constant_data[f'{name}_back_counts'],
                (idata.posterior.chain.size, idata.posterior.draw.size, 1)
            )
            a = idata.constant_data[f'{name}_spec_exposure'].data / \
                idata.constant_data[f'{name}_back_exposure'].data
            net_counts = spec_counts - a * back_counts

            all_channel.append(net_counts)
            data_set[f'{name}_Non'] = (coords, spec_counts)
            data_set[f'{name}_Net'] = (coords, net_counts)

        elif stat == 'pgstat':
            mu_off = idata['posterior'][f'{name}_BKG']
            sigma = idata['constant_data'][f'{name}_back_error']
            spec_counts = rng.poisson(mu_on).astype(np.float64)
            back_counts = rng.normal(mu_off, sigma)
            a = idata.constant_data[f'{name}_spec_exposure'].data / \
                idata.constant_data[f'{name}_back_exposure'].data
            net_counts = spec_counts - a * back_counts

            all_channel.append(net_counts)
            data_set[f'{name}_Non'] = (coords, spec_counts)
            data_set[f'{name}_Noff'] = (coords, back_counts)
            data_set[f'{name}_Net'] = (coords, net_counts)

        elif stat == 'wstat':
            mu_off = idata['posterior'][f'{name}_BKG']
            spec_counts = rng.poisson(mu_on).astype(np.float64)
            back_counts = rng.poisson(mu_off).astype(np.float64)
            a = idata.constant_data[f'{name}_spec_exposure'].data / \
                idata.constant_data[f'{name}_back_exposure'].data
            net_counts = spec_counts - a * back_counts

            all_channel.append(net_counts)
            data_set[f'{name}_Non'] = (coords, spec_counts)
            data_set[f'{name}_Noff'] = (coords, back_counts)
            data_set[f'{name}_Net'] = (coords, net_counts)

        else:
            raise ValueError(f'statistic "{stat}" is not support')

    all_channel = np.concatenate(all_channel, axis=2)
    data_set['all_channel'] = (('chain', 'draw', 'channel'), all_channel)
    idata.add_groups({'posterior_predictive': data_set})

    return idata
