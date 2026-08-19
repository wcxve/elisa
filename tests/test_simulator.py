import numpy as np
import pytest
from scipy import stats

from elisa import MaxLikeFit
from elisa.models import PowerLaw


def _simulate_data_for_stat(stat: str):
    """Create a minimal dataset suitable for the requested statistic."""
    nbins = 16
    photon_egrid = np.linspace(1.0, 10.0, nbins + 1)
    channel_emin = photon_egrid[:-1]
    channel_emax = photon_egrid[1:]
    response_matrix = np.eye(nbins)

    spec_exposure = 50.0
    back_exposure = 100.0

    compiled_model = PowerLaw(K=[80.0], alpha=0.0).compile()

    if stat == 'chi2':
        # Gaussian data requires explicit errors.
        spec_errors = np.linspace(0.8, 1.2, nbins)
        return compiled_model.simulate(
            photon_egrid=photon_egrid,
            channel_emin=channel_emin,
            channel_emax=channel_emax,
            response_matrix=response_matrix,
            spec_exposure=spec_exposure,
            spec_poisson=False,
            spec_errors=spec_errors,
            name='chi2_sim',
            seed=11,
        )

    if stat == 'cstat':
        return compiled_model.simulate(
            photon_egrid=photon_egrid,
            channel_emin=channel_emin,
            channel_emax=channel_emax,
            response_matrix=response_matrix,
            spec_exposure=spec_exposure,
            spec_poisson=True,
            name='cstat_sim',
            seed=12,
        )

    if stat == 'pstat':
        # Background exists but is treated as known;
        # pstat must be set explicitly.
        back_counts = np.full(nbins, 40.0)
        return compiled_model.simulate(
            photon_egrid=photon_egrid,
            channel_emin=channel_emin,
            channel_emax=channel_emax,
            response_matrix=response_matrix,
            spec_exposure=spec_exposure,
            spec_poisson=True,
            back_counts=back_counts,
            back_exposure=back_exposure,
            back_poisson=True,
            name='pstat_sim',
            seed=13,
        )

    if stat == 'pgstat':
        back_counts = np.full(nbins, 40.0)
        # Make Gaussian sigma far from Poisson sqrt(mu) to clearly distinguish.
        back_errors = np.linspace(0.4, 0.7, nbins)
        return compiled_model.simulate(
            photon_egrid=photon_egrid,
            channel_emin=channel_emin,
            channel_emax=channel_emax,
            response_matrix=response_matrix,
            spec_exposure=spec_exposure,
            spec_poisson=True,
            back_counts=back_counts,
            back_errors=back_errors,
            back_exposure=back_exposure,
            back_poisson=False,
            name='pgstat_sim',
            seed=14,
        )

    if stat == 'wstat':
        back_counts = np.full(nbins, 40.0)
        return compiled_model.simulate(
            photon_egrid=photon_egrid,
            channel_emin=channel_emin,
            channel_emax=channel_emax,
            response_matrix=response_matrix,
            spec_exposure=spec_exposure,
            spec_poisson=True,
            back_counts=back_counts,
            back_exposure=back_exposure,
            back_poisson=True,
            name='wstat_sim',
            seed=15,
        )

    raise ValueError(f'Unknown statistic: {stat}')


def _assert_uniform_pit(u: np.ndarray, alpha: float = 1e-6) -> None:
    """Assert PIT values are consistent with Uniform(0, 1) via SciPy."""
    u = np.asarray(u, float).ravel()
    assert np.all((u >= 0.0) & (u <= 1.0))
    res = stats.kstest(u, 'uniform')
    assert res.pvalue > alpha


def _assert_sim_matches_poisson(samples: np.ndarray, lam: np.ndarray) -> None:
    """Goodness-of-fit checks for Poisson simulator.

    Uses SciPy randomized PIT + KS test, plus basic moment sanity checks.
    """
    samples = np.asarray(samples, float)
    lam = np.asarray(lam, float)

    # Poisson samples should be non-negative integers.
    assert np.all(samples >= 0.0)
    assert np.all(samples == np.round(samples))

    n = samples.shape[0]
    mean = samples.mean(axis=0)
    var = samples.var(axis=0, ddof=1)

    lam_safe = np.clip(lam, 1e-6, None)
    z_mean = (mean - lam_safe) / np.sqrt(lam_safe / n)
    assert np.nanmax(np.abs(z_mean)) < 6.0

    rel_var = np.abs(var - lam_safe) / lam_safe
    assert np.nanmax(rel_var) < 0.2

    # Randomized PIT for discrete distributions should be Uniform(0, 1).
    k = samples.astype(int)
    cdf_hi = np.asarray(stats.poisson.cdf(k, mu=lam_safe), float)
    cdf_lo = np.asarray(stats.poisson.cdf(k - 1, mu=lam_safe), float)
    rng = np.random.default_rng(0)
    u = cdf_lo + rng.random(size=samples.shape) * (cdf_hi - cdf_lo)
    _assert_uniform_pit(u)


def _assert_sim_matches_normal(
    samples: np.ndarray, mu: np.ndarray, sigma: np.ndarray
) -> None:
    """Goodness-of-fit checks for Normal simulator.

    Uses SciPy PIT + KS test, plus basic moment sanity checks.
    """
    samples = np.asarray(samples, float)
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)

    assert np.all(sigma > 0.0)

    n = samples.shape[0]
    mean = samples.mean(axis=0)
    var = samples.var(axis=0, ddof=1)

    z_mean = (mean - mu) / (sigma / np.sqrt(n))
    assert np.nanmax(np.abs(z_mean)) < 6.0

    rel_var = np.abs(var - sigma * sigma) / (sigma * sigma)
    assert np.nanmax(rel_var) < 0.2

    # Normal samples should not be (almost all) integers.
    frac_int = np.mean(samples == np.round(samples))
    assert frac_int < 0.05

    # PIT for continuous distributions should be Uniform(0, 1).
    u = np.asarray(stats.norm.cdf(samples, loc=mu, scale=sigma), float)
    _assert_uniform_pit(u)


@pytest.mark.parametrize(
    'stat',
    [
        pytest.param('chi2', id='chi2'),
        pytest.param('cstat', id='cstat'),
        pytest.param('pstat', id='pstat'),
        pytest.param('pgstat', id='pgstat'),
        pytest.param('wstat', id='wstat'),
    ],
)
def test_helper_simulator_matches_sampling_dist(stat: str):
    """Statistical checks that helper.simulate matches helper.sampling_dist."""
    data = _simulate_data_for_stat(stat)

    # pstat must be explicitly requested; other stats can be auto-selected.
    fit = MaxLikeFit(
        data,
        PowerLaw(alpha=0.0),
        stat='pstat' if stat == 'pstat' else None,
    )
    helper = fit._helper

    unconstr_arr = helper.free_default['unconstr_arr']
    assert not isinstance(unconstr_arr, dict)
    model_values = helper.get_sites(unconstr_arr)['models']
    sim = helper.simulate(2025, model_values, 4096)

    name = data.name
    for key in (f'{name}_Non', f'{name}_Noff'):
        if key not in helper.sampling_dist:
            assert key not in sim
            continue

        dist, args = helper.sampling_dist[key]
        mu = np.asarray(model_values[f'{key}_model'], float)
        samples = np.asarray(sim[key], float)

        if dist == 'poisson':
            _assert_sim_matches_poisson(samples, mu)
        elif dist == 'norm':
            (sigma,) = args
            _assert_sim_matches_normal(samples, mu, np.asarray(sigma, float))
        else:
            raise AssertionError(f'Unexpected dist in sampling_dist: {dist}')
