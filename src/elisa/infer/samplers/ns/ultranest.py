from __future__ import annotations

import warnings
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from ultranest import ReactiveNestedSampler, read_file

from elisa.infer.samplers.util import ravel_params_names, uniform_reparam_model

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


class UltraNestSampler:
    def __init__(
        self,
        numpyro_model: Callable,
        model_args: tuple = (),
        model_kwargs: dict | None = None,
        seed: int = 42,
        ignore_nan: bool = False,
        **kwargs: dict,
    ):
        if ignore_nan:
            warnings.warn(
                'setting `ignore_nan` to True may fail to spot potential '
                'issues of the model',
                Warning,
            )

        self._model_info = mi = uniform_reparam_model(
            numpyro_model,
            model_args,
            model_kwargs,
            rng_seed=seed,
        )

        @jax.jit
        @jax.vmap
        def log_prob_fn(cube_and_derived):
            log_p = mi.log_prob_fn(mi.unravel(cube_and_derived[: mi.ndim]))
            if ignore_nan:
                log_p = jnp.nan_to_num(log_p, nan=-1e300)
            return log_p

        self._log_prob_fn = log_prob_fn
        self._sampler = None
        self._sampler_constructor = partial(ReactiveNestedSampler, **kwargs)
        self._seed = seed

    def run(
        self,
        viz_sample_names: list[str] | None = None,
        read_file_config: dict | None = None,
        **kwargs: dict,
    ) -> dict[str, NDArray[float]]:
        mi = self._model_info

        if viz_sample_names is None:
            viz_sample_names = mi.params_names
        else:
            viz_sample_names = list(map(str, viz_sample_names))

        @jax.jit
        @jax.vmap
        def transform(cube):
            samples = mi.postprocess_fn(mi.unravel(cube))
            viz = jnp.hstack([samples[i].ravel() for i in viz_sample_names])
            return jnp.append(cube, viz)

        params_names = []
        for i in mi.params_dtype:
            shape = i[2]
            name = f'u_{i[0]}'
            params_names.extend(ravel_params_names(name, shape))

        samples_dtype = mi.params_dtype + mi.deterministic_dtype
        derived_names = []
        for i in viz_sample_names:
            filtered = list(filter(lambda x: x[0] == i, samples_dtype))
            if any(filtered):
                shape = filtered[0][2]
                derived_names.extend(ravel_params_names(i, shape))

        if read_file_config is None:
            prev_state = np.random.get_state()
            np.random.seed(self._seed)
            sampler = self._sampler = self._sampler_constructor(
                param_names=params_names,
                loglike=lambda x: jax.device_get(self._log_prob_fn(x)),
                transform=lambda x: jax.device_get(transform(x)),
                derived_param_names=derived_names,
                vectorized=True,
            )
            sampler.run(**kwargs)
            np.random.set_state(prev_state)
            u_samples = sampler.results['samples'][:, : mi.ndim]
        else:
            read_file_config = dict(read_file_config)
            read_file_config['x_dim'] = mi.ndim
            sequence, final = read_file(**read_file_config)
            results = sequence | final
            u_samples = results['samples'][:, : mi.ndim]

        u_samples = jax.vmap(mi.unravel)(u_samples)
        samples = jax.vmap(mi.postprocess_fn)(u_samples)
        samples = jax.device_get(samples)
        return samples

    def print_results(self, use_unicode: bool = True):
        if not hasattr(self._sampler, 'results'):
            raise RuntimeError(
                'no results found, please run the sampler first.'
            )
        self._sampler.print_results(use_unicode=use_unicode)

    @property
    def ess(self) -> int:
        if not hasattr(self._sampler, 'results'):
            return 0
        else:
            return int(self._sampler.results['ess'])

    @property
    def lnZ(self) -> tuple[float | None, float | None]:
        if not hasattr(self._sampler, 'results'):
            return None, None
        else:
            return (
                self._sampler.results['logz'],
                self._sampler.results['logzerr'],
            )
