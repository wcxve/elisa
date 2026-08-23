from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import multiprocess as mp
import nautilus
import nautilus.pool as nautilus_pool
import numpy as np

from elisa.infer.samplers.util import uniform_reparam_model

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

DEFAULT_ESS_MULTIPLIER = 2.0


def resolve_ess_multiplier(k: float | None) -> float:
    """Return and validate the requested or default ESS multiplier."""
    if k is None:
        k = DEFAULT_ESS_MULTIPLIER
    return check_ess_multiplier(k)


def check_ess_multiplier(k: float) -> float:
    """Validate the target-draws multiplier: finite and ``>= 1``.

    ``k`` controls the PSIS-LOO reff through ``reff ~ 1 / k``; values below 1
    would allow reff > 1, which is why ``>= 1`` (not merely ``> 0``) is
    enforced, matching the documentation.
    """
    k = float(k)
    if not math.isfinite(k) or k < 1.0:
        raise ValueError(
            f'`ess_multiplier` must be finite and >= 1, got {k!r}'
        )
    return k


def check_equal_weight_boost(b: float) -> float:
    """Validate an explicit ``equal_weight_boost``: finite and positive."""
    b = float(b)
    if not math.isfinite(b) or b <= 0.0:
        raise ValueError(
            f'`equal_weight_boost` must be finite and positive, got {b!r}'
        )
    return b


def weighted_bases(log_w) -> float:
    """Expected number of equal-weight draws that nautilus would return at
    ``equal_weight_boost == 1``.

    From the nautilus ``posterior()`` source, point ``i`` is repeated a number
    of times with mean ``exp(log_w_i - max(log_w)) * boost``, so by linearity
    the expected total draw count at boost ``b`` is ``b * n_base`` where

        n_base = sum(exp(log_w - max(log_w)))

    ``log_w`` must contain the native unequal posterior weights returned by
    ``posterior(equal_weight=False)``.  The formula is invariant to the
    constant offset added by weight normalisation.
    """
    log_w = np.asarray(log_w, dtype=float)
    if log_w.size == 0:
        raise ValueError('`log_w` must not be empty')
    if np.isnan(log_w).any() or np.isposinf(log_w).any():
        raise ValueError('`log_w` must not contain NaN or positive infinity')

    max_log_w = float(np.max(log_w))
    if not math.isfinite(max_log_w):
        raise ValueError('`log_w` must contain at least one finite value')

    return float(np.exp(log_w - max_log_w).sum())


def adaptive_equal_weight_boost(
    weighted_ess: float,
    n_base: float,
    ess_multiplier: float,
) -> float:
    """Choose a boost targeting ``ess_multiplier * weighted_ess`` draws."""
    ess_multiplier = check_ess_multiplier(ess_multiplier)
    weighted_ess = float(weighted_ess)
    n_base = float(n_base)
    if not math.isfinite(weighted_ess) or weighted_ess <= 0.0:
        raise ValueError('`weighted_ess` must be finite and positive')
    if not math.isfinite(n_base) or n_base <= 0.0:
        raise ValueError('`n_base` must be finite and positive')
    return max(1.0, ess_multiplier * weighted_ess / n_base)


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

    def run(
        self,
        ess_multiplier: float | None = None,
        equal_weight_boost: float | None = None,
        **kwargs,
    ) -> dict[str, NDArray[float]]:
        kwargs.setdefault('verbose', True)
        kwargs['discard_exploration'] = True
        sampler = self._sampler
        success = sampler.run(**kwargs)
        if not success:
            raise RuntimeError(
                'Sampling failed due to limits were reached, please set a '
                'larger `n_like_max` or `timeout`. You can also resume the '
                'sampler from previous one, providing `filepath` and `resume`.'
            )

        # Native (weighted) posterior: the returned ``log_w`` here are the
        # true posterior weights.  Calling with ``equal_weight=False`` is
        # REQUIRED -- with ``equal_weight=True`` nautilus resets ``log_w`` to
        # the uniform weights of the resampled posterior (see nautilus source).
        _u_native, log_w_native, _log_l = sampler.posterior(
            return_as_dict=False,
            equal_weight=False,
        )
        n_base = weighted_bases(log_w_native)
        weighted_ess = float(sampler.n_eff)

        if equal_weight_boost is None:
            ess_multiplier = resolve_ess_multiplier(ess_multiplier)
            # Expected number of equal-weight draws at boost b is b * n_base,
            # so pick b to make the expected draw count k * weighted_ess:
            #   E[draws] = b * n_base = k * weighted_ess
            #   reff (PSIS-LOO) = weighted_ess / n_draws ~ 1 / k
            equal_weight_boost = adaptive_equal_weight_boost(
                weighted_ess,
                n_base,
                ess_multiplier,
            )
        else:
            equal_weight_boost = check_equal_weight_boost(equal_weight_boost)

        u_samples, _ew_logw, _log_l = sampler.posterior(
            return_as_dict=False,
            equal_weight=True,
            equal_weight_boost=float(equal_weight_boost),
        )
        n_draws = len(u_samples)
        if n_draws == 0:
            raise RuntimeError(
                '`equal_weight_boost` produced no posterior draws; increase '
                'its value or use the adaptive default.'
            )
        u_samples = jax.vmap(self._model_info.unravel)(u_samples)
        samples = jax.vmap(self._model_info.postprocess_fn)(u_samples)
        samples = jax.device_get(samples)

        # expose the quantities needed by fit.py for the PSIS-LOO reff contract
        self._weighted_ess = weighted_ess
        self._n_base = float(n_base)
        self._equal_weight_boost = float(equal_weight_boost)
        self._n_draws = n_draws
        return samples

    @property
    def ess(self) -> int:
        return int(self._sampler.n_eff)

    @property
    def weighted_ess(self) -> float:
        """Native (weighted) effective sample size of the posterior."""
        return self._weighted_ess

    @property
    def n_draws(self) -> int:
        """Number of equal-weight posterior draws exported by :meth:`run`."""
        return self._n_draws

    @property
    def n_base(self) -> float:
        """Expected draw count when ``equal_weight_boost=1``."""
        return self._n_base

    @property
    def equal_weight_boost(self) -> float:
        """Boost actually used by the last :meth:`run` call."""
        return self._equal_weight_boost

    @property
    def lnZ(self) -> float | None:
        return self._sampler.log_z
