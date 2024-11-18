from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from elisa.infer.fit import BayesFit, MaxLikeFit
from elisa.models.add import PowerLaw

if TYPE_CHECKING:
    from typing import Callable

    from elisa import Data
    from elisa.infer.results import MLEResult, PosteriorResult


@pytest.fixture(scope='session')
def powerlaw_fn() -> Callable:
    @jax.jit
    def _(alpha, K, egrid):
        one_minus_alpha = 1.0 - alpha
        f = K / one_minus_alpha * jnp.power(egrid, one_minus_alpha)
        return f[1:] - f[:-1]

    return _


@pytest.fixture(scope='session')
def powerlaw_flux(powerlaw_fn) -> Callable:
    @jax.jit
    def _(alpha, K, emin, emax):
        flux_keV = powerlaw_fn(alpha - 1, K, jnp.array([emin, emax]))[0]
        flux_erg = flux_keV * 1.602176634e-9
        return flux_erg

    return _


@pytest.fixture(scope='session')
def simulation() -> Data:
    """Simulate a simple power-law spectrum with a known flux and index."""
    # Setup simulation configuration
    seed = 42
    emin = 1.0
    emax = 100.0
    nbins = 200
    photon_egrid = np.linspace(emin, emax, nbins + 1)
    channel_emin = photon_egrid[:-1]
    channel_emax = photon_egrid[1:]
    response_matrix = np.eye(nbins)
    spec_exposure = 50.0

    # Setup model and simulate data
    alpha = 0.0
    K = 10.0  # photon flux between emin and emax, when alpha = 0
    compiled_model = PowerLaw(K=[K], alpha=alpha).compile()
    data = compiled_model.simulate(
        photon_egrid=photon_egrid,
        channel_emin=channel_emin,
        channel_emax=channel_emax,
        response_matrix=response_matrix,
        spec_exposure=spec_exposure,
        spec_poisson=True,
        seed=seed,
    )
    return data


@pytest.fixture(scope='session')
def mle_result(simulation) -> MLEResult:
    data = simulation
    model = PowerLaw(K=[10.0], alpha=0.0)
    return MaxLikeFit(data, model).mle()


@pytest.fixture(scope='session')
def mle_result2(simulation) -> MLEResult:
    data = simulation
    model = PowerLaw(K=[10.0], alpha=[0.0])
    return MaxLikeFit(data, model).mle()


@pytest.fixture(scope='session')
def mle_result2_covar(simulation, powerlaw_fn) -> tuple[Callable, Callable]:
    data = simulation.spec_counts
    expo = simulation.spec_exposure
    egrid = simulation.resp_data.photon_egrid

    @jax.jit
    @jax.hessian
    @jax.jit
    def hess(params):
        alpha, K = params
        model = powerlaw_fn(alpha, K, egrid) * expo
        return jnp.sum(data * jnp.log(model) - model)

    @jax.jit
    def covar(params):
        return jnp.linalg.inv(-hess(params))

    @functools.partial(jax.jit, static_argnums=1)
    @jax.jacobian
    @functools.partial(jax.jit, static_argnums=1)
    def jacobian(params, fn):
        return jnp.hstack([params, fn(params)])

    @functools.partial(jax.jit, static_argnums=1)
    def covar_fn(params, fn):
        cov = covar(params)
        jac = jacobian(params, fn)
        return jac @ cov @ jac.T

    return covar, covar_fn


@pytest.fixture(scope='session')
def posterior_result(simulation) -> PosteriorResult:
    data = simulation
    model = PowerLaw(K=[10.0])
    return BayesFit(data, model).nuts()
