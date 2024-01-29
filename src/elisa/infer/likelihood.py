"""Likelihood functions with goodness term."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
from jax import lax
from jax.scipy.special import xlogy
from numpyro.distributions import Normal, Poisson
from numpyro.distributions.util import validate_sample

NDArray = jnp.ndarray


def pgstat_background(
    s: NDArray, n: NDArray, b_est: NDArray, b_err: NDArray, a: float | NDArray
) -> jax.Array:
    """Optimized background for PG-statistics given estimate of source counts.

    Parameters
    ----------
    s : array_like
        Estimate of source counts.
    n : array_like
        Observed source and background counts.
    b_est : array_like
        Estimate of background counts.
    b_err : array_like
        Uncertainty of background counts.
    a : float or array_like
        Exposure ratio between source and background observations.

    Returns
    -------
    b : jax.Array
        The profile background.

    References
    ----------
    .. [1] https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSappendixStatistics.html

    """
    variance = b_err * b_err
    e = b_est - a * variance
    f = a * variance * n + e * s
    c = a * e - s
    d = jnp.sqrt(c * c + 4.0 * a * f)
    b = jnp.where(
        jnp.bitwise_or(jnp.greater_equal(e, 0.0), jnp.greater_equal(f, 0.0)),
        jnp.where(jnp.greater(n, 0.0), (c + d) / (2 * a), e),
        0.0,
    )
    return b


def wstat_background(
    s: NDArray,
    n_on: NDArray,
    n_off: NDArray,
    a: float | NDArray,
) -> jax.Array:
    """Optimized background for W-statistics given estimate of source counts.

    Parameters
    ----------
    s : array_like
        Estimate of source counts.
    n_on : array_like
        Observed source and background counts in "on" observation.
    n_off : array_like
        Observed background counts in "off" observation.
    a : array_like
        Exposure ratio between "on" and "off" observations.

    Returns
    -------
    b : jax.Array
        The profile background.

    References
    ----------
    .. [1] Wachter, K., Leach, R., & Kellogg, E. (1979). Parameter estimation
       in X-ray astronomy using maximum likelihood. ApJ, 230, 274–287.

    """
    c = a * (n_on + n_off) - (a + 1) * s
    d = jnp.sqrt(c * c + 4 * a * (a + 1) * n_off * s)
    b = jnp.where(
        jnp.equal(n_on, 0),
        n_off / (1 + a),
        jnp.where(
            jnp.equal(n_off, 0),
            jnp.where(
                jnp.less_equal(s, a / (a + 1) * n_on),
                n_on / (1 + a) - s / a,
                0.0,
            ),
            (c + d) / (2 * a * (a + 1)),
        ),
    )
    return b


class NormalWithGoodness(Normal):
    @validate_sample
    def log_prob(self, value):
        value_scaled = (value - self.loc) / self.scale
        return -0.5 * value_scaled * value_scaled


class PoissonWithGoodness(Poisson):
    @validate_sample
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        if (
            self.is_sparse
            and not isinstance(value, jax.core.Tracer)
            and jnp.size(value) > 1
        ):
            broadcast = jnp.broadcast_to
            shape = lax.broadcast_shapes(self.batch_shape, jnp.shape(value))
            rate = broadcast(self.rate, shape).reshape(-1)
            nonzero = broadcast(jax.device_get(value) > 0, shape).reshape(-1)
            value = broadcast(value, shape).reshape(-1)
            sparse_value = value[nonzero]
            sparse_rate = rate[nonzero]
            tmp = xlogy(sparse_value, sparse_rate)
            gof = xlogy(sparse_value, sparse_value) - sparse_value
            return jnp.clip(
                jnp.asarray(-rate, dtype=jnp.result_type(float))
                .at[nonzero]
                .add(tmp - gof)
                .reshape(shape),
                a_max=0.0,
            )

        else:
            logp = xlogy(value, self.rate) - self.rate
            gof = xlogy(value, value) - value
            return jnp.clip(logp - gof, a_max=0.0)


def chi2(
    model: jnp.ndarray,
    name: str,
    spec: jnp.ndarray,
    error: jnp.ndarray,
    predictive: bool,
):
    """Chi-squared statistic, i.e. Gaussian likelihood."""
    spec_data = numpyro.primitives.mutable(f"{name}_Non_data", spec)

    spec_model = numpyro.deterministic(f"{name}_Non_model", model)

    with numpyro.plate(name, len(spec_data)):
        numpyro.sample(
            name=f"{name}_Non",
            fn=NormalWithGoodness(spec_model, error),
            obs=None if predictive else spec_data,
        )


def cstat(model: jnp.ndarray, name: str, spec: jnp.ndarray, predictive: bool):
    """C-statistic, i.e. Poisson likelihood."""
    spec_data = numpyro.primitives.mutable(f"{name}_Non_data", spec)

    spec_model = numpyro.deterministic(f"{name}_Non_model", model)

    with numpyro.plate(name, len(spec_data)):
        numpyro.sample(
            name=f"{name}_Non",
            fn=PoissonWithGoodness(spec_model),
            obs=None if predictive else spec_data,
        )


def pstat(
    model: jnp.ndarray,
    name: str,
    spec: jnp.ndarray,
    back: jnp.ndarray,
    ratio: jnp.ndarray | float,
    predictive: bool,
):
    """P-statistic, i.e. Poisson likelihood for data with known background."""
    spec_data = numpyro.primitives.mutable(f"{name}_Non_data", spec)

    b = back
    spec_model = numpyro.deterministic(f"{name}_Non_model", model + ratio * b)

    with numpyro.plate(name, len(spec_data)):
        numpyro.sample(
            name=f"{name}_Non",
            fn=PoissonWithGoodness(spec_model),
            obs=None if predictive else spec_data,
        )


def pgstat(
    model: NDArray,
    name: str,
    spec: NDArray,
    back: NDArray,
    back_error: NDArray,
    ratio: NDArray | float,
    predictive: bool,
):
    """PG-statistic, i.e. Poisson likelihood for data and profile Gaussian
    likelihood for background.
    """
    spec_data = numpyro.primitives.mutable(f"{name}_Non_data", spec)
    back_data = numpyro.primitives.mutable(f"{name}_Noff_data", back)

    b = pgstat_background(model, spec_data, back_data, back_error, ratio)

    spec_model = numpyro.deterministic(f"{name}_Non_model", model + ratio * b)
    back_model = numpyro.deterministic(f"{name}_Noff_model", b)

    with numpyro.plate(name, len(spec_data)):
        numpyro.sample(
            name=f"{name}_Non",
            fn=PoissonWithGoodness(spec_model),
            obs=None if predictive else spec_data,
        )

        numpyro.sample(
            name=f"{name}_Noff",
            fn=NormalWithGoodness(back_model, back_error),
            obs=None if predictive else back_data,
        )


def wstat(
    model: NDArray,
    name: str,
    spec: NDArray,
    back: NDArray,
    ratio: NDArray | float,
    predictive: bool,
):
    """W-statistic, i.e. Poisson likelihood for data and profile Poisson
    likelihood for background.
    """
    spec_data = numpyro.primitives.mutable(f"{name}_Non_data", spec)
    back_data = numpyro.primitives.mutable(f"{name}_Noff_data", back)

    b = wstat_background(model, spec_data, back_data, ratio)

    spec_model = numpyro.deterministic(f"{name}_Non_model", model + ratio * b)
    back_model = numpyro.deterministic(f"{name}_Noff_model", b)

    with numpyro.plate(name, len(spec_data)):
        numpyro.sample(
            name=f"{name}_Non",
            fn=PoissonWithGoodness(spec_model),
            obs=None if predictive else spec_data,
        )

        numpyro.sample(
            name=f"{name}_Noff",
            fn=PoissonWithGoodness(back_model),
            obs=None if predictive else back_data,
        )
