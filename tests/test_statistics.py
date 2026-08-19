import numpy as np
import pytest

from elisa import MaxLikeFit
from elisa.models import PowerLaw


def _simulate_data(stat: str):
    """Create a minimal dataset for each fit statistic."""
    nbins = 32
    photon_egrid = np.linspace(1.0, 10.0, nbins + 1)
    channel_emin = photon_egrid[:-1]
    channel_emax = photon_egrid[1:]
    response_matrix = np.eye(nbins)

    spec_exposure = 50.0
    back_exposure = 100.0

    compiled_model = PowerLaw(K=[50.0], alpha=0.0).compile()

    if stat == 'chi2':
        spec_errors = np.full(nbins, 1.0)
        return compiled_model.simulate(
            photon_egrid=photon_egrid,
            channel_emin=channel_emin,
            channel_emax=channel_emax,
            response_matrix=response_matrix,
            spec_exposure=spec_exposure,
            spec_poisson=False,
            spec_errors=spec_errors,
            name='chi2_sim',
            seed=1,
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
            seed=2,
        )

    if stat == 'pstat':
        back_counts = np.full(nbins, 30.0)
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
            seed=3,
        )

    if stat == 'pgstat':
        back_counts = np.full(nbins, 30.0)
        back_errors = np.linspace(0.5, 0.8, nbins)
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
            seed=4,
        )

    if stat == 'wstat':
        back_counts = np.full(nbins, 30.0)
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
            seed=5,
        )

    raise ValueError(f'Unknown statistic: {stat}')


@pytest.mark.parametrize(
    'dataset_stat, expected_stat',
    [
        pytest.param('chi2', 'chi2', id='auto_chi2'),
        pytest.param('cstat', 'cstat', id='auto_cstat'),
        pytest.param('pgstat', 'pgstat', id='auto_pgstat'),
        pytest.param('wstat', 'wstat', id='auto_wstat'),
    ],
)
def test_auto_statistic_selection(dataset_stat: str, expected_stat: str):
    data = _simulate_data(dataset_stat)
    fit = MaxLikeFit(data, PowerLaw(alpha=0.0))
    assert fit._helper.statistic[data.name] == expected_stat


def test_explicit_pstat_is_accepted_for_data_with_background():
    data = _simulate_data('pstat')
    fit = MaxLikeFit(data, PowerLaw(alpha=0.0), stat='pstat')
    assert fit._helper.statistic[data.name] == 'pstat'


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
def test_mle_runs_for_each_statistic(stat: str):
    data = _simulate_data(stat)
    fit = MaxLikeFit(
        data,
        PowerLaw(alpha=0.0),
        stat='pstat' if stat == 'pstat' else None,
    )
    result = fit.mle(max_steps=5000)
    assert np.isfinite(result.mle['PowerLaw.K'][0])
    assert result.mle['PowerLaw.K'][0] > 0.0


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
def test_sampling_dist_metadata(stat: str):
    data = _simulate_data(stat)
    fit = MaxLikeFit(
        data,
        PowerLaw(alpha=0.0),
        stat='pstat' if stat == 'pstat' else None,
    )
    helper = fit._helper
    name = data.name

    non_key = f'{name}_Non'
    assert non_key in helper.sampling_dist

    if stat == 'chi2':
        dist, args = helper.sampling_dist[non_key]
        assert dist == 'norm'
        (sigma,) = args
        assert np.allclose(sigma, helper.data[name].spec_errors)
        assert f'{name}_Noff' not in helper.sampling_dist

    elif stat in {'cstat', 'pstat'}:
        assert helper.sampling_dist[non_key] == ('poisson', ())
        assert f'{name}_Noff' not in helper.sampling_dist

    elif stat == 'pgstat':
        assert helper.sampling_dist[non_key] == ('poisson', ())
        dist, args = helper.sampling_dist[f'{name}_Noff']
        assert dist == 'norm'
        (sigma,) = args
        assert np.allclose(sigma, helper.data[name].back_errors)

    elif stat == 'wstat':
        assert helper.sampling_dist[non_key] == ('poisson', ())
        assert helper.sampling_dist[f'{name}_Noff'] == ('poisson', ())

    else:
        raise AssertionError(f'Unhandled statistic in test: {stat}')


@pytest.mark.parametrize(
    'bad_stat',
    [
        pytest.param('pstat', id='pstat'),
        pytest.param('pgstat', id='pgstat'),
        pytest.param('wstat', id='wstat'),
    ],
)
def test_background_required_for_background_statistics(bad_stat: str):
    data = _simulate_data('cstat')
    with pytest.raises(ValueError):
        MaxLikeFit(data, PowerLaw(alpha=0.0), stat=bad_stat)


@pytest.mark.parametrize(
    'bad_stat',
    [
        pytest.param('cstat', id='cstat'),
        pytest.param('pstat', id='pstat'),
        pytest.param('pgstat', id='pgstat'),
        pytest.param('wstat', id='wstat'),
    ],
)
def test_gaussian_data_requires_chi2(bad_stat: str):
    data = _simulate_data('chi2')
    with pytest.raises(ValueError):
        MaxLikeFit(data, PowerLaw(alpha=0.0), stat=bad_stat)


def test_cstat_is_invalid_with_background():
    data = _simulate_data('wstat')
    with pytest.raises(ValueError):
        MaxLikeFit(data, PowerLaw(alpha=0.0), stat='cstat')
