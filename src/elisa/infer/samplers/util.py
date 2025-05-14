from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.flatten_util import ravel_pytree
from numpyro.handlers import reparam, seed, trace
from numpyro.infer.initialization import init_to_uniform
from numpyro.infer.reparam import Reparam
from numpyro.infer.util import (
    Predictive,
    _guess_max_plate_nesting,
    _validate_model,
    initialize_model,
    log_density,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import dtype, floating
    from numpy.typing import NDArray


class ModelInfo(NamedTuple):
    """Model information."""

    ndim: int
    """Model dimension."""

    init: dict[str, NDArray[floating]]
    """Initial parameters values in unconstrained space."""

    init_ravel: NDArray[floating]
    """Raveled initial parameters values in unconstrained space."""

    unravel: Callable[[NDArray[floating]], dict[str, NDArray[floating]]]
    """Function to unravel parameters values."""

    log_prob_fn: Callable[[dict[str, NDArray[floating]]], float]
    """Log probability function given parameters in unconstrained space."""

    postprocess_fn: Callable[
        [dict[str, NDArray[floating]]],
        dict[str, NDArray[floating]],
    ]
    """Postprocess function given parameters in unconstrained space."""

    params_names: list[str]
    """Names of parameters."""

    params_dtype: list[tuple[str, dtype, tuple[int, ...]]]
    """NumPy dtypes of parameters in constrained space."""

    deterministic_names: list[str]
    """Names of deterministic sites."""

    deterministic_dtype: list[tuple[str, dtype, tuple[int, ...]]]
    """NumPy dtypes of deterministic sites."""


def get_model_info(
    model: Callable,
    init_strategy: Callable = init_to_uniform,
    model_args: tuple = (),
    model_kwargs: dict | None = None,
    forward_mode_differentiation: bool = False,
    validate_grad: bool = True,
    rng_seed: int = 42,
) -> ModelInfo:
    """Get model information."""
    model_info = initialize_model(
        rng_key=jax.random.PRNGKey(rng_seed),
        model=model,
        init_strategy=init_strategy,
        model_args=model_args,
        model_kwargs=model_kwargs,
        forward_mode_differentiation=forward_mode_differentiation,
        validate_grad=validate_grad,
    )

    potential_fn = model_info.potential_fn
    log_prob_fn = jax.jit(lambda z: -potential_fn(z))
    postprocess_fn = jax.jit(model_info.postprocess_fn)

    init = model_info.param_info.z
    init_ravel, unravel = ravel_pytree(init)
    init_ravel = jax.device_get(init_ravel)

    samples = postprocess_fn(init)

    params_names = list(init.keys())
    params_dtype = [
        (i, samples[i].dtype, samples[i].shape) for i in params_names
    ]

    deterministic_names = [i for i in samples if i not in params_names]
    deterministic_dtype = [
        (i, samples[i].dtype, samples[i].shape) for i in deterministic_names
    ]

    return ModelInfo(
        ndim=len(init_ravel),
        init=init,
        init_ravel=init_ravel,
        unravel=unravel,
        log_prob_fn=log_prob_fn,
        postprocess_fn=postprocess_fn,
        params_names=params_names,
        params_dtype=params_dtype,
        deterministic_names=deterministic_names,
        deterministic_dtype=deterministic_dtype,
    )


def ravel_params_names(name: str, shape: tuple[int, ...]) -> list[str]:
    """Ravel parameter names."""
    if shape == ():
        return [str(name)]

    indices = np.indices(shape).reshape(len(shape), -1).T
    indices = indices.astype(str).tolist()
    return [f'{name}[{",".join(i)}]' for i in indices]


# >>> Codes below are adapted from numpyro.contrib.nested_sampling >>>


class UniformReparam(Reparam):
    """Reparameterize a distribution to a Uniform over the unit hypercube.

    Most univariate distribution uses inverse CDF for reparameterization.
    """

    def __call__(self, name, fn, obs):
        if obs is not None:
            raise ValueError(
                'UniformReparam does not support observe statements'
            )
        shape = fn.shape()
        fn, expand_shape, event_dim = self._unwrap(fn)
        transform = uniform_reparam_transform(fn)
        tiny = jnp.finfo(jnp.result_type(float)).tiny

        x = numpyro.sample(
            name=f'u_{name}',
            fn=dist.Uniform(tiny, 1)
            .expand(shape)
            .to_event(event_dim)
            .mask(False),
        )

        # Simulate a numpyro.deterministic() site.
        return None, transform(x)


@singledispatch
def uniform_reparam_transform(d):
    """A helper for :class:`UniformReparam` to get the transform that maps a
    uniform distribution over a unit hypercube to the target distribution `d`.
    """
    if isinstance(d, dist.TransformedDistribution):
        outer_transform = dist.transforms.ComposeTransform(d.transforms)

        def transform(q):
            return outer_transform(uniform_reparam_transform(d.base_dist)(q))

    elif isinstance(
        d,
        dist.Independent | dist.ExpandedDistribution | dist.MaskedDistribution,
    ):

        def transform(q):
            return uniform_reparam_transform(d.base_dist)(q)

    else:
        transform = d.icdf

    return transform


@uniform_reparam_transform.register(dist.MultivariateNormal)
def _(d):
    outer_transform = dist.transforms.LowerCholeskyAffine(d.loc, d.scale_tril)

    def transform(q):
        return outer_transform(dist.Normal(0, 1).icdf(q))

    return transform


@uniform_reparam_transform.register(dist.BernoulliLogits)
@uniform_reparam_transform.register(dist.BernoulliProbs)
def _(d):
    def transform(q):
        x = q < d.probs
        return jnp.astype(x, jnp.result_type(x, int))

    return transform


@uniform_reparam_transform.register(dist.CategoricalLogits)
@uniform_reparam_transform.register(dist.CategoricalProbs)
def _(d):
    def transform(q):
        return jnp.sum(jnp.cumsum(d.probs, axis=-1) < q[..., None], axis=-1)

    return transform


@uniform_reparam_transform.register(dist.Dirichlet)
def _(d):
    gamma_dist = dist.Gamma(d.concentration)

    def transform_fn(q):
        # NB: icdf is not available yet for Gamma distribution
        # so this will raise an NotImplementedError for now.
        # We will need scipy.special.gammaincinv, which is not available yet
        # in JAX, see issue: https://github.com/google/jax/issues/5350
        # TODO: consider wrap jaxns GammaPrior transform implementation
        gammas = uniform_reparam_transform(gamma_dist)(q)
        return gammas / gammas.sum(-1, keepdims=True)

    return transform_fn


def uniform_reparam_model(
    model: Callable,
    model_args: tuple = (),
    model_kwargs: dict | None = None,
    rng_seed: int = 42,
) -> ModelInfo:
    seed_key, pred_key = random.split(random.PRNGKey(rng_seed))

    if model_kwargs is None:
        model_kwargs = {}

    model_trace = trace(seed(model, seed_key)).get_trace(
        *model_args, **model_kwargs
    )

    # params in constrained space
    params = {
        site['name']: site['value']
        for site in model_trace.values()
        if (
            (site['type'] == 'sample')
            and (not site['is_observed'])
            and (site['infer'].get('enumerate', '') != 'parallel')
        )
    }
    params_names = list(params.keys())
    params_dtype = [(k, v.dtype, v.shape) for k, v in params.items()]

    # deterministic sites
    deterministic_names = [
        site['name']
        for site in model_trace.values()
        if site['type'] == 'deterministic'
    ]

    # reparam the model so that latent sites have Uniform(0, 1) priors
    reparam_model = reparam(
        model, config={k: UniformReparam() for k in params_names}
    )

    # hyper cube
    cube = {f'u_{v[0]}': jnp.full(v[2], 0.5, v[1]) for v in params_dtype}
    cube_ravel, unravel = ravel_pytree(cube)
    cube = jax.device_get(cube)
    cube_ravel = jax.device_get(cube_ravel)

    # enable enum if needed
    has_enum = any(
        site['type'] == 'sample'
        and site['infer'].get('enumerate', '') == 'parallel'
        for site in model_trace.values()
    )
    if has_enum:
        from numpyro.contrib.funsor import enum, log_density as log_density_fn

        max_plate_nesting = _guess_max_plate_nesting(model_trace)
        _validate_model(model_trace)
        reparam_model = enum(reparam_model, -max_plate_nesting - 1)
    else:
        log_density_fn = log_density

    @jax.jit
    def log_prob_fn(params_cube):
        log_prob, _ = log_density_fn(
            reparam_model, model_args, model_kwargs, params_cube
        )
        return log_prob

    @jax.jit
    def postprocess_fn(params_cube):
        return Predictive(
            reparam_model,
            params_cube,
            return_sites=params_names + deterministic_names,
            batch_ndims=0,
        )(pred_key, *model_args, **model_kwargs)

    samples = postprocess_fn(cube)
    deterministic = {i: samples[i] for i in deterministic_names}
    deterministic_dtype = [
        (k, v.dtype, v.shape) for k, v in deterministic.items()
    ]

    return ModelInfo(
        ndim=len(cube_ravel),
        init=cube,
        init_ravel=cube_ravel,
        unravel=unravel,
        log_prob_fn=log_prob_fn,
        postprocess_fn=postprocess_fn,
        params_names=params_names,
        params_dtype=params_dtype,
        deterministic_names=deterministic_names,
        deterministic_dtype=deterministic_dtype,
    )


# <<< Codes above are adapted from numpyro.contrib.nested_sampling <<<
