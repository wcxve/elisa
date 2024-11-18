import numpy as np
import pytest

from elisa import BayesFit, MaxLikeFit
from elisa.models import PowerLaw


@pytest.mark.parametrize(
    'method',
    [
        pytest.param('minuit', id='minuit'),
        pytest.param('lm', id='lm'),
        pytest.param('ns', id='ns'),
    ],
)
def test_trivial_max_like_fit(simulation, method):
    data = simulation
    model = PowerLaw(alpha=0.0)
    result = MaxLikeFit(data, model).mle(method=method)

    # Check that the fit result is correct,
    # note that the analytic result is known for alpha = 0 and uniform egrid
    mle_fit, err_fit = result.mle['PowerLaw.K']
    mle_analytic = np.mean(data.ce)
    nbins = data.resp_data.channel_number
    spec_exposure = data.spec_exposure
    de = np.diff(data.resp_data.photon_egrid)[0]
    err_analytic = np.sqrt(mle_analytic / nbins / spec_exposure / de)
    ci = result.ci().errors['PowerLaw.K']

    assert np.isclose(mle_fit, mle_analytic)
    assert np.isclose(err_fit, err_analytic)
    assert np.isclose(ci[0], -err_analytic, rtol=5e-3, atol=0)
    assert np.isclose(ci[1], err_analytic, rtol=5e-3, atol=0)


@pytest.mark.parametrize(
    'method, options',
    [
        pytest.param('nuts', {}, id='nuts'),
        pytest.param('jaxns', {}, id='jaxns'),
        pytest.param(
            'aies', {'chain_method': 'parallel', 'n_parallel': 4}, id='aies'
        ),
        pytest.param(
            'ess', {'chain_method': 'parallel', 'n_parallel': 4}, id='ess'
        ),
        pytest.param('ultranest', {}, id='ultranest'),
        pytest.param('nautilus', {}, id='nautilus'),
    ],
)
def test_trivial_bayes_fit(simulation, method, options):
    data = simulation
    model = PowerLaw()
    model.PowerLaw.K.log = True

    # Get Bayesian fit result, i.e. posterior
    result = getattr(BayesFit(data, model), method)(**options)

    # check convergence
    assert all(i < 1.01 for i in result.rhat.values() if not np.isnan(i))

    # check the true parameters values are within the 68% CI
    ci = result.ci(cl=1).intervals
    assert ci['PowerLaw.K'][0] < 10.0 < ci['PowerLaw.K'][1]
    assert ci['PowerLaw.alpha'][0] < 0.0 < ci['PowerLaw.alpha'][1]
