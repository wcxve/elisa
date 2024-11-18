from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from elisa.infer.fit import BayesFit, MaxLikeFit
from elisa.models.add import PowerLaw

if TYPE_CHECKING:
    from elisa import Data
    from elisa.infer.results import MLEResult, PosteriorResult


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
def posterior_result(simulation) -> PosteriorResult:
    data = simulation
    model = PowerLaw(K=[10.0])
    return BayesFit(data, model).nuts()
