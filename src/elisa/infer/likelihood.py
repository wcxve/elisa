"""Likelihood functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, get_args

import jax
import jax.numpy as jnp
import numpyro
from jax import lax
from jax.experimental.sparse import BCSR
from jax.scipy.special import xlogy
from numpyro.distributions import Normal, Poisson
from numpyro.distributions.util import validate_sample

if TYPE_CHECKING:
    from typing import Callable

    from elisa.data.base import FixedData
    from elisa.util.typing import (
        ArrayLike,
        JAXArray,
        ModelCompiledFn,
        ParamNameValMapping,
    )


# TODO:
#   It should be noted that 'lstat' does not have long run coverage property
#   for source estimation, which is probably due to the choice of conjugate
#   prior of Poisson background data.
#   'lstat' will be included here with a proper prior at some point.
Statistic = Literal['chi2', 'cstat', 'pstat', 'pgstat', 'wstat']

_STATISTIC_OPTIONS: frozenset[str] = frozenset(get_args(Statistic))
_STATISTIC_SPEC_NORMAL: frozenset[str] = frozenset({'chi2'})
_STATISTIC_BACK_NORMAL: frozenset[str] = frozenset({'pgstat'})
_STATISTIC_WITH_BACK: frozenset[str] = frozenset({'pgstat', 'wstat'})


def pgstat_background(
    s: ArrayLike,
    n: ArrayLike,
    b_est: ArrayLike,
    b_err: ArrayLike,
    a: ArrayLike,
) -> JAXArray:
    """Optimized background for PG-statistics given estimate of source counts.

    .. note::
        The optimized background here is always non-negative, which differs
        from XSPEC [1]_.

    Parameters
    ----------
    s : array_like
        Estimate of source counts.
    n : array_like
        Observed counts (source and background).
    b_est : array_like
        Estimate of background counts.
    b_err : array_like
        Uncertainty of background counts.
    a : float or array_like
        Exposure ratio between source and background observations.

    Returns
    -------
    JAXArray
        The profile background.

    References
    ----------
    .. [1] `XSPEC Manual Appendix B: Statistics in XSPEC <https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSappendixStatistics.html>`__.
    """
    variance = b_err * b_err
    e = jnp.array(b_est - a * variance)
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
    s: ArrayLike,
    n_on: ArrayLike,
    n_off: ArrayLike,
    a: ArrayLike,
) -> JAXArray:
    """Optimized background for W-statistics [1]_ given the estimate of source.

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
    JAXArray
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


class BetterNormal(Normal):
    @validate_sample
    def log_prob(self, value):
        value_scaled = (value - self.loc) / self.scale
        return -0.5 * value_scaled * value_scaled


class BetterPoisson(Poisson):
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


def _get_resp_matrix(data: FixedData) -> JAXArray | BCSR:
    if data.response_sparse:
        return BCSR.from_scipy_sparse(data.sparse_matrix.T)
    else:
        return jnp.array(data.response_matrix.T, float)


def chi2(
    data: FixedData,
    model: ModelCompiledFn,
) -> Callable[[ParamNameValMapping, bool], None]:
    """S^2 statistic, Gaussian likelihood."""
    name = str(data.name)
    spec = jnp.array(data.net_counts, float)
    error = jnp.array(data.net_errors, float)
    photon_egrid = jnp.array(data.photon_egrid, float)
    channel_width = jnp.array(data.channel_width, float)
    resp_matrix = _get_resp_matrix(data)
    area_scale = jnp.array(data.area_scale, float)
    exposure = jnp.array(data.spec_exposure, float)

    def likelihood(
        params: ParamNameValMapping,
        predictive: bool = False,
    ) -> None:
        """Gaussian likelihood defined via numpyro primitives."""
        unfold = model(photon_egrid, params)
        unfold = jnp.clip(unfold, a_min=-1e-300, a_max=1e300)
        source_rate = resp_matrix @ unfold * area_scale
        numpyro.deterministic(name, source_rate / channel_width)
        source_counts = source_rate * exposure
        source_counts = jnp.clip(source_counts, a_min=1e-30, a_max=1e15)
        spec_data = numpyro.primitives.mutable(f'{name}_Non_data', spec)
        spec_model = numpyro.deterministic(f'{name}_Non_model', source_counts)

        with numpyro.plate(f'{name}_plate', len(spec)):
            dist_on = BetterNormal(spec_model, error)
            numpyro.sample(
                name=f'{name}_Non',
                fn=dist_on,
                obs=None if predictive else spec_data,
            )

        # record log likelihood into chains to avoid re-computation
        if not predictive:
            loglike_on = numpyro.deterministic(
                name=f'{name}_Non_loglike', value=dist_on.log_prob(spec_data)
            )
            numpyro.deterministic(name=f'{name}_loglike', value=loglike_on)

    return likelihood


def cstat(
    data: FixedData,
    model: ModelCompiledFn,
) -> Callable[[ParamNameValMapping, bool], None]:
    """C-statistic, Poisson likelihood."""
    name = str(data.name)
    spec = jnp.array(data.spec_counts, float)
    photon_egrid = jnp.array(data.photon_egrid, float)
    channel_width = jnp.array(data.channel_width, float)
    resp_matrix = _get_resp_matrix(data)
    area_scale = jnp.array(data.area_scale, float)
    exposure = jnp.array(data.spec_exposure, float)

    def likelihood(
        params: ParamNameValMapping,
        predictive: bool = False,
    ) -> None:
        """Poisson likelihood defined via numpyro primitives."""
        unfold = model(photon_egrid, params)
        unfold = jnp.clip(unfold, a_min=-1e-300, a_max=1e300)
        source_rate = resp_matrix @ unfold * area_scale
        numpyro.deterministic(name, source_rate / channel_width)
        source_counts = source_rate * exposure
        source_counts = jnp.clip(source_counts, a_min=1e-30, a_max=1e15)
        spec_data = numpyro.primitives.mutable(f'{name}_Non_data', spec)
        spec_model = numpyro.deterministic(f'{name}_Non_model', source_counts)

        with numpyro.plate(f'{name}_plate', len(spec)):
            dist_on = BetterPoisson(spec_model)
            numpyro.sample(
                name=f'{name}_Non',
                fn=dist_on,
                obs=None if predictive else spec_data,
            )

        # record log likelihood into chains to avoid re-computation
        if not predictive:
            loglike_on = numpyro.deterministic(
                name=f'{name}_Non_loglike', value=dist_on.log_prob(spec_data)
            )
            numpyro.deterministic(name=f'{name}_loglike', value=loglike_on)

    return likelihood


def pstat(
    data: FixedData,
    model: ModelCompiledFn,
) -> Callable[[ParamNameValMapping, bool], None]:
    """P-statistic, Poisson likelihood for data with a known background."""
    assert data.has_back, 'Data must have background'

    name = str(data.name)
    spec = jnp.array(data.spec_counts, float)
    back = jnp.array(data.back_counts, float)
    photon_egrid = jnp.array(data.photon_egrid, float)
    channel_width = jnp.array(data.channel_width, float)
    resp_matrix = _get_resp_matrix(data)
    area_scale = jnp.array(data.area_scale, float)
    exposure = jnp.array(data.spec_exposure, float)
    back_ratio = jnp.array(data.back_ratio, float)

    def likelihood(
        params: ParamNameValMapping,
        predictive: bool = False,
    ) -> None:
        """Poisson likelihood defined via numpyro primitives."""
        unfold = model(photon_egrid, params)
        unfold = jnp.clip(unfold, a_min=-1e-300, a_max=1e300)
        source_rate = resp_matrix @ unfold * area_scale
        numpyro.deterministic(name, source_rate / channel_width)
        model_counts = source_rate * exposure + back_ratio * back
        model_counts = jnp.clip(model_counts, a_min=1e-30, a_max=1e15)
        spec_data = numpyro.primitives.mutable(f'{name}_Non_data', spec)
        spec_model = numpyro.deterministic(f'{name}_Non_model', model_counts)

        with numpyro.plate(f'{name}_plate', len(spec_data)):
            dist_on = BetterPoisson(spec_model)
            numpyro.sample(
                name=f'{name}_Non',
                fn=dist_on,
                obs=None if predictive else spec_data,
            )

        # record log likelihood into chains to avoid re-computation
        if not predictive:
            loglike_on = numpyro.deterministic(
                name=f'{name}_Non_loglike', value=dist_on.log_prob(spec_data)
            )
            numpyro.deterministic(name=f'{name}_loglike', value=loglike_on)

    return likelihood


def pgstat(
    data: FixedData,
    model: ModelCompiledFn,
) -> Callable[[ParamNameValMapping, bool], None]:
    """PG-statistic, Poisson likelihood for data and profile Gaussian
    likelihood for background.
    """
    assert data.has_back, 'Data must have background'

    name = str(data.name)
    spec = jnp.array(data.spec_counts, float)
    back = jnp.array(data.back_counts, float)
    back_error = jnp.array(data.back_errors, float)
    photon_egrid = jnp.array(data.photon_egrid, float)
    channel_width = jnp.array(data.channel_width, float)
    resp_matrix = _get_resp_matrix(data)
    area_scale = jnp.array(data.area_scale, float)
    exposure = jnp.array(data.spec_exposure, float)
    back_ratio = jnp.array(data.back_ratio, float)

    def likelihood(params: ParamNameValMapping, predictive: bool = False):
        """Poisson and Gaussian likelihood defined via numpyro primitives."""
        unfold = model(photon_egrid, params)
        unfold = jnp.clip(unfold, a_min=-1e-300, a_max=1e300)
        source_rate = resp_matrix @ unfold * area_scale
        numpyro.deterministic(name, source_rate / channel_width)
        spec_data = numpyro.primitives.mutable(f'{name}_Non_data', spec)
        back_data = numpyro.primitives.mutable(f'{name}_Noff_data', back)
        source_counts = source_rate * exposure
        source_counts = jnp.clip(source_counts, a_min=1e-30, a_max=1e15)
        b = pgstat_background(
            source_counts, spec_data, back_data, back_error, back_ratio
        )
        spec_model = source_counts + back_ratio * b
        spec_model = numpyro.deterministic(f'{name}_Non_model', spec_model)
        back_model = numpyro.deterministic(f'{name}_Noff_model', b)

        with numpyro.plate(f'{name}_plate', len(spec_data)):
            dist_on = BetterPoisson(spec_model)
            dist_off = BetterNormal(back_model, back_error)
            numpyro.sample(
                name=f'{name}_Non',
                fn=dist_on,
                obs=None if predictive else spec_data,
            )
            numpyro.sample(
                name=f'{name}_Noff',
                fn=dist_off,
                obs=None if predictive else back_data,
            )

        # record log likelihood into chains to avoid re-computation
        if not predictive:
            loglike_on = numpyro.deterministic(
                name=f'{name}_Non_loglike', value=dist_on.log_prob(spec_data)
            )
            loglike_off = numpyro.deterministic(
                name=f'{name}_Noff_loglike', value=dist_off.log_prob(back_data)
            )
            numpyro.deterministic(
                name=f'{name}_loglike', value=loglike_on + loglike_off
            )

    return likelihood


def wstat(
    data: FixedData,
    model: ModelCompiledFn,
) -> Callable[[ParamNameValMapping, bool], None]:
    """W-statistic, i.e. Poisson likelihood for data and profile Poisson
    likelihood for background.
    """
    assert data.has_back, 'Data must have background'

    name = str(data.name)
    spec = jnp.array(data.spec_counts, float)
    back = jnp.array(data.back_counts, float)
    photon_egrid = jnp.array(data.photon_egrid, float)
    channel_width = jnp.array(data.channel_width, float)
    resp_matrix = _get_resp_matrix(data)
    area_scale = jnp.array(data.area_scale, float)
    exposure = jnp.array(data.spec_exposure, float)
    back_ratio = jnp.array(data.back_ratio, float)

    def likelihood(params: ParamNameValMapping, predictive: bool = False):
        """Poisson and Poisson likelihood defined via numpyro primitives."""
        unfold = model(photon_egrid, params)
        unfold = jnp.clip(unfold, a_min=-1e-300, a_max=1e300)
        source_rate = resp_matrix @ unfold * area_scale
        numpyro.deterministic(name, source_rate / channel_width)
        spec_data = numpyro.primitives.mutable(f'{name}_Non_data', spec)
        back_data = numpyro.primitives.mutable(f'{name}_Noff_data', back)
        source_counts = source_rate * exposure
        source_counts = jnp.clip(source_counts, a_min=1e-30, a_max=1e15)
        b = wstat_background(source_counts, spec_data, back_data, back_ratio)
        model_counts = source_counts + back_ratio * b
        spec_model = numpyro.deterministic(f'{name}_Non_model', model_counts)
        back_model = numpyro.deterministic(f'{name}_Noff_model', b)

        with numpyro.plate(f'{name}_plate', len(spec_data)):
            dist_on = BetterPoisson(spec_model)
            dist_off = BetterPoisson(back_model)
            numpyro.sample(
                name=f'{name}_Non',
                fn=dist_on,
                obs=None if predictive else spec_data,
            )
            numpyro.sample(
                name=f'{name}_Noff',
                fn=dist_off,
                obs=None if predictive else back_data,
            )

        # record log likelihood into chains to avoid re-computation
        if not predictive:
            loglike_on = numpyro.deterministic(
                name=f'{name}_Non_loglike', value=dist_on.log_prob(spec_data)
            )
            loglike_off = numpyro.deterministic(
                name=f'{name}_Noff_loglike', value=dist_off.log_prob(back_data)
            )
            numpyro.deterministic(
                name=f'{name}_loglike', value=loglike_on + loglike_off
            )

    return likelihood
