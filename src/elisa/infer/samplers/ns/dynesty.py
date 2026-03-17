from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import multiprocess as mp
import numpy as np
from dynesty import DynamicNestedSampler, NestedSampler

from elisa.infer.samplers.util import uniform_reparam_model

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


class DynestySampler:
    def __init__(
        self,
        numpyro_model: Callable,
        model_args: tuple = (),
        model_kwargs: dict | None = None,
        seed: int = 42,
        ignore_nan: bool = False,
        dynamic: bool = False,
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
        def log_prob_fn(cube):
            log_p = mi.log_prob_fn(mi.unravel(cube))
            if ignore_nan:
                log_p = jnp.nan_to_num(log_p, nan=-1e300)
            return log_p

        self._log_prob_fn = log_prob_fn
        self._dynamic = bool(dynamic)
        self._seed = int(seed)
        self._sampler = None
        self._run_results = None
        self._pool = None

        if 'pool' not in kwargs and kwargs.get('queue_size', 1) > 1:
            old_method = mp.get_start_method()
            if old_method != 'spawn':
                mp.set_start_method('spawn', force=True)
            else:
                old_method = ''
            self._pool = mp.Pool(int(kwargs['queue_size']))
            kwargs['pool'] = self._pool
            kwargs.setdefault(
                'use_pool',
                {
                    'prior_transform': False,
                    'loglikelihood': True,
                    'propose_point': False,
                    'update_bound': False,
                },
            )
            if old_method:
                mp.set_start_method(old_method, force=True)

        sampler_cls = DynamicNestedSampler if self._dynamic else NestedSampler
        self._sampler_cls = sampler_cls
        self._sampler_constructor_kwargs = dict(kwargs)

    def run(
        self,
        resume_file: str | None = None,
        **kwargs: dict,
    ) -> dict[str, NDArray[float]]:
        mi = self._model_info
        kwargs.setdefault('print_progress', True)

        if resume_file and Path(resume_file).exists():
            sampler = self._sampler_cls.restore(
                resume_file,
                pool=self._sampler_constructor_kwargs.get('pool'),
            )
        else:
            sampler = self._sampler_cls(
                loglikelihood=lambda x: jax.device_get(self._log_prob_fn(x)),
                prior_transform=lambda x: x,
                ndim=mi.ndim,
                **self._sampler_constructor_kwargs,
            )

        prev_state = np.random.get_state()
        np.random.seed(self._seed)
        sampler.run_nested(**kwargs)
        np.random.set_state(prev_state)

        self._sampler = sampler
        results = self._run_results = sampler.results

        u_samples = getattr(results, 'samples_u', None)
        if u_samples is None:
            u_samples = np.asarray(results.samples)

        weights = np.exp(results.logwt - results.logz[-1])
        weights = weights / np.sum(weights)
        rng = np.random.default_rng(self._seed)
        idx = rng.choice(
            len(weights), size=len(weights), replace=True, p=weights
        )
        u_samples = u_samples[idx]

        u_samples = jnp.asarray(u_samples)
        u_samples = jax.vmap(mi.unravel)(u_samples)
        samples = jax.vmap(mi.postprocess_fn)(u_samples)
        samples = jax.device_get(samples)
        return samples

    def print_results(self):
        if self._run_results is None:
            raise RuntimeError(
                'no results found, please run the sampler first.'
            )
        self._run_results.summary()

    @property
    def ess(self) -> int:
        if self._run_results is None:
            return 0

        weights = np.exp(self._run_results.logwt - self._run_results.logz[-1])
        weights = weights / np.sum(weights)
        return int(1.0 / np.sum(weights**2))

    @property
    def lnZ(self) -> tuple[float | None, float | None]:
        if self._run_results is None:
            return None, None
        return (
            float(self._run_results.logz[-1]),
            float(self._run_results.logzerr[-1]),
        )

    def __del__(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
