from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import multiprocess as mp
import nautilus
import nautilus.pool as nautilus_pool

from elisa.infer.samplers.util import uniform_reparam_model

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


class NautilusSampler:
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
        def log_prob_fn(cube_and_derived):
            log_p = mi.log_prob_fn(mi.unravel(cube_and_derived[: mi.ndim]))
            if ignore_nan:
                log_p = jnp.nan_to_num(log_p, nan=-1e300)
            return log_p

        if 'pool' in kwargs:
            kwargs['vectorized'] = False
            old_method = mp.get_start_method()
            if old_method != 'spawn':
                mp.set_start_method('spawn', force=True)
            else:
                old_method = ''
            # monkey patching the pool for compatibility with JAX
            old_pool = nautilus_pool.Pool
            nautilus_pool.Pool = mp.Pool
        else:
            kwargs['vectorized'] = True
            log_prob_fn = jax.jit(jax.vmap(log_prob_fn))
            old_method = ''
            old_pool = None

        self._sampler = nautilus.Sampler(
            prior=lambda x: x,
            likelihood=lambda x: jax.device_get(log_prob_fn(x)),
            n_dim=mi.ndim,
            pass_dict=False,
            seed=seed,
            **kwargs,
        )

        if old_method:
            mp.set_start_method(old_method, force=True)

        if old_pool is not None:
            nautilus_pool.Pool = old_pool

    def run(self, **kwargs) -> dict[str, NDArray[float]]:
        kwargs.setdefault('verbose', True)
        kwargs['discard_exploration'] = True
        sampler = self._sampler
        success = sampler.run(**kwargs)
        if success:
            u_samples, *_ = sampler.posterior(
                return_as_dict=False,
                equal_weight=True,
            )
            u_samples = jax.vmap(self._model_info.unravel)(u_samples)
            samples = jax.vmap(self._model_info.postprocess_fn)(u_samples)
            samples = jax.device_get(samples)
            return samples
        else:
            raise RuntimeError(
                'Sampling failed due to limits were reached, please set a '
                'larger `n_like_max` or `timeout`. You can also resume the '
                'sampler from previous one, providing `filepath` and `resume`.'
            )

    @property
    def ess(self) -> int:
        return int(self._sampler.n_eff)

    @property
    def lnZ(self) -> float | None:
        return self._sampler.log_z
