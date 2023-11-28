"""Various likelihood function with a goodness term."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro

from jax import lax
from jax.scipy.special import xlogy
from numpyro.distributions import Normal, Poisson
from numpyro.distributions.util import validate_sample

NDArray = np.ndarray


def pgstat_background(
    s: NDArray,
    n: NDArray,
    b_est: NDArray,
    b_err: NDArray,
    a: float | NDArray
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

    """
    variance = b_err * b_err
    e = b_est - a * variance
    f = a * variance * n + e * s
    c = a * e - s
    d = jnp.sqrt(c * c + 4.0 * a * f)
    b = lax.switch(
        lax.bitwise_or(lax.ge(e, 0.0), lax.ge(f, 0.0)),
        lax.switch(
            lax.gt(n, 0.0),
            (c + d) / (2 * a),
            e
        ),
        0.0
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

    """
    c = a * (n_on + n_off) - (a + 1) * s
    d = jnp.sqrt(c * c + 4 * a * (a + 1) * n_off * s)
    b = lax.switch(
        lax.eq(n_on, 0),
        n_off / (1 + a),
        lax.switch(
            lax.eq(n_off, 0),
            lax.switch(
                lax.le(s, a / (a + 1) * n_on),
                n_on / (1 + a) - s / a,
                0.0
            ),
            (c + d) / (2 * a * (a + 1))
        )
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
            tmp = sparse_value * jnp.log(sparse_rate)
            gof = xlogy(sparse_value, sparse_value) - sparse_value
            return (
                jnp.asarray(-rate, dtype=jnp.result_type(float))
                .at[nonzero]
                .add(tmp - gof)
                .reshape(shape)
            )

        else:
            logp = value * jnp.log(self.rate) - self.rate
            gof = xlogy(value, value) - value
            return logp - gof


def chi2(name, model, spec, error):
    with numpyro.plate('data', len(spec)):
        numpyro.sample(
            name=f'{name}_Non',
            fn=NormalWithGoodness(model, error),
            obs=spec
        )


def cstat(name, model, spec):
    with numpyro.plate('data', len(spec)):
        numpyro.sample(
            name=f'{name}_Non',
            fn=PoissonWithGoodness(model),
            obs=spec
        )


def pstat(name, model, spec, back, ratio):
    with numpyro.plate('data', len(spec)):
        numpyro.sample(
            name=f'{name}_Non',
            fn=PoissonWithGoodness(model + ratio * back),
            obs=spec
        )


def pgstat(name, model, spec, back, back_error, ratio):
    b = pgstat_background(model, spec, back, back_error, ratio)
    with numpyro.plate('data', len(spec)):
        numpyro.sample(
            name=f'{name}_Non',
            fn=PoissonWithGoodness(model + ratio * b),
            obs=spec
        )

        numpyro.sample(
            name=f'{name}_Noff',
            fn=NormalWithGoodness(b, back_error),
            obs=back
        )


def wstat(name, model, spec, back, ratio):
    b = wstat_background(model, spec, back, ratio)
    with numpyro.plate('data', len(spec)):
        numpyro.sample(
            name=f'{name}_Non',
            fn=PoissonWithGoodness(model + ratio * b),
            obs=spec
        )

        numpyro.sample(
            name=f'{name}_Noff',
            fn=PoissonWithGoodness(b),
            obs=back
        )


# def model():
#     def f():
#         return numpyro.sample("mu", numpyro.distributions.Uniform(0, 1000))
#     mu = f()
#     n = 1000
#     np.random.seed(42)
#     data = np.random.poisson(50, n)
#     with numpyro.plate("data", n):
#         numpyro.sample("y", PoissonWithGoodness(mu), obs=data)
#
#
# def model2():
#     mu = numpyro.sample("mu", numpyro.distributions.Uniform(0, 1000))
#     n = 1000
#     np.random.seed(42)
#     data = np.random.poisson(50, n)
#     with numpyro.plate("data", n):
#         numpyro.sample("y", Poisson(mu), obs=data)


# import arviz as az
# import numpyro
# from numpyro import infer
#
# numpyro.set_host_device_count(2)
#
# from numpyro.infer.util import log_likelihood
#
# from jax.scipy.optimize import minimize
#
# def array_to_dict(params, names):
#     return {name: param for param, name in zip(params, names)}
#
# def deviance_uncon(params, names):
#     lnL_single = log_likelihood(model, array_to_dict(params, names))
#     lnL_total = jax.tree_util.tree_reduce(
#         lambda x, y: x + y,
#         jax.tree_map(lambda x: x.sum(), lnL_single)
#     )
#     return -2.0*lnL_total
# def deviance_uncon2(params, names):
#     lnL_single = log_likelihood(model2, array_to_dict(params, names))
#     lnL_total = jax.tree_util.tree_reduce(
#         lambda x, y: x + y,
#         jax.tree_map(lambda x: x.sum(), lnL_single)
#     )
#     return -2.0*lnL_total
# jax.config.update("jax_enable_x64", True)
# print(minimize(deviance_uncon, np.array([50.]), (['mu'],), method='BFGS'))
# print(minimize(deviance_uncon2, np.array([50.]), (['mu'],), method='BFGS'))
#
# np.random.seed(42)
# data = np.random.poisson(50, 1000)
# mu = 49.801
# print(-2*np.sum(-mu + data*np.log(mu) - (data*np.log(data)-data)))
