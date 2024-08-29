import numpy as np

from elisa import BayesFit, MaxLikeFit
from elisa.models import PowerLaw


def test_trivial_max_like_fit():
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
    model = PowerLaw(K=[K], alpha=alpha)
    compiled_model = model.compile()
    data = compiled_model.simulate(
        photon_egrid=photon_egrid,
        channel_emin=channel_emin,
        channel_emax=channel_emax,
        response_matrix=response_matrix,
        spec_exposure=spec_exposure,
        spec_poisson=True,
        seed=seed,
    )

    # Get mle result
    result = MaxLikeFit(data, model).mle()

    # Check that the fit result is correct,
    # note that the analytic result is known for alpha = 0 and uniform egrid
    mle_fit, err_fit = result.mle['PowerLaw.K']
    mle_analytic = np.mean(data.ce)
    de = np.diff(photon_egrid)[0]
    err_analytic = np.sqrt(mle_analytic / nbins / spec_exposure / de)
    ci = result.ci().errors['PowerLaw.K']
    assert np.isclose(mle_fit, mle_analytic)
    assert np.isclose(err_fit, err_analytic)
    assert np.isclose(ci[0], -err_analytic, rtol=1e-3, atol=0)
    assert np.isclose(ci[1], err_analytic, rtol=1e-3, atol=0)
    assert result.ndata['total'] == nbins
    assert result.dof == nbins - 1

    # Check various methods of mle result
    result.boot(1009)
    result.ci(method='boot')
    result.flux(1, 2)
    result.lumin(1, 10000, z=1)
    result.eiso(1, 10000, z=1, duration=spec_exposure)
    result.summary()
    result.plot()
    result.plot('data ne ene eene Fv vFv rq pit corner')
    _ = result.deviance
    _ = result.aic
    _ = result.bic
    assert all(i > 0.05 for i in result.gof.values())

    plotter = result.plot
    plotter.plot_qq('rd')
    plotter.plot_qq('rp')
    plotter.plot_qq('rq')


def test_trivial_bayes_fit():
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
    alpha = 0.5
    K = 10.0
    model = PowerLaw(K=[K], alpha=[alpha])
    model.PowerLaw.K.log = True
    compiled_model = model.compile()
    data = compiled_model.simulate(
        photon_egrid=photon_egrid,
        channel_emin=channel_emin,
        channel_emax=channel_emax,
        response_matrix=response_matrix,
        spec_exposure=spec_exposure,
        spec_poisson=True,
        seed=seed,
    )

    sampling_kwargs = {
        'nuts': {},
        'jaxns': {},
        'aies': {
            'chain_method': 'parallel',
            'n_parallel': 4,
        },
        'ultranest': {},
        'nautilus': {},
    }

    for method, kwargs in sampling_kwargs.items():
        # Get Bayesian fit result, i.e. posterior
        result = getattr(BayesFit(data, model), method)(**kwargs)

        # check the true parameters values are within the 90% CI
        ci = result.ci(cl=0.9).intervals
        assert ci['PowerLaw.K'][0] < K < ci['PowerLaw.K'][1]
        assert ci['PowerLaw.alpha'][0] < alpha < ci['PowerLaw.alpha'][1]

        # Check various methods of posterior result
        assert result.ndata['total'] == nbins
        assert result.dof == nbins - 2

        result.ppc(1009)
        result.flux(1, 2)
        result.lumin(1, 10000, z=1)
        result.eiso(1, 10000, z=1, duration=spec_exposure)
        result.summary()
        result.plot()
        result.plot('data ne ene eene Fv vFv rq pit corner khat')
        _ = result.deviance
        _ = result.loo
        _ = result.waic
        assert all(i > 0.05 for i in result.gof.values())

        plotter = result.plot
        plotter.plot_qq('rd')
        plotter.plot_qq('rp')
        plotter.plot_qq('rq')
