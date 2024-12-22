import jax.numpy as jnp
import numpy as np
import pytest

from elisa.models.add import PowerLaw


def test_mle_result(simulation, mle_result):
    data = simulation
    result = mle_result

    nbins = len(data.channel)

    assert result.ndata['total'] == nbins
    assert result.dof == nbins - 1

    # Check various methods of mle result
    result.boot(1009)
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

    result.save('mle.pkl.gz', 'gzip')
    result.load('mle.pkl.gz', 'gzip')
    result.save('mle.pkl.bz2', 'bz2')
    result.load('mle.pkl.bz2', 'bz2')
    result.save('mle.pkl.xz', 'lzma')
    result.load('mle.pkl.xz', 'lzma')


@pytest.mark.parametrize(
    'method, rtol',
    [
        pytest.param('hess', 1e-3, id='hess'),
        pytest.param('boot', 0.1, id='boot'),
    ],
)
def test_mle_covar(
    mle_result2, mle_result2_covar, powerlaw_fn, powerlaw_flux, method, rtol
):
    alpha_mle = mle_result2.mle['PowerLaw.alpha'][0]
    K_mle = mle_result2.mle['PowerLaw.K'][0]
    params_mle = jnp.array([alpha_mle, K_mle])
    covar_true = mle_result2_covar[0](params_mle)

    if method == 'boot':
        mle_result2.boot(4000)

    covar = mle_result2.covar(method=method).matrix
    assert np.allclose(covar, covar_true, rtol=rtol, atol=0.0)

    if method == 'boot':
        covar_ = mle_result2.covar(method=method, parallel=False).matrix
        assert np.allclose(covar, covar_)

    flux = lambda params: powerlaw_flux(
        params['PowerLaw.alpha'] - 1.0, params['PowerLaw.K'], 1.0, 10.0
    )

    covar_true = mle_result2_covar[1](
        params_mle,
        lambda x: flux(dict(zip(['PowerLaw.alpha', 'PowerLaw.K'], x))),
    )

    covar = mle_result2.covar(fn={'flux': flux}, method=method).matrix
    assert np.allclose(covar, covar_true, rtol=rtol, atol=0.0)

    if method == 'boot':
        covar_ = mle_result2.covar(
            fn={'flux': flux}, method=method, parallel=False
        ).matrix
        assert np.allclose(covar, covar_)


def test_mle_ci_fn(mle_result2, powerlaw_flux):
    result = mle_result2
    result.boot()

    alpha_mle = result.mle['PowerLaw.alpha'][0]
    K_mle = result.mle['PowerLaw.K'][0]
    emin = 1.0
    emax = 10.0
    fn_mle = powerlaw_flux(alpha_mle, K_mle, emin, emax)
    fn = lambda params: powerlaw_flux(
        params['PowerLaw.alpha'], params['PowerLaw.K'], emin, emax
    )

    ci0 = result.ci(params=[], fn={'fn': fn}, rtol={'fn': 1e-10})
    ci1 = result.ci(params=[], fn={'fn': fn})
    ci2 = result.ci(params=[], fn={'fn': fn}, method='boot')
    ci3 = result.ci(params=[], fn={'fn': fn}, method='boot', parallel=False)
    assert np.allclose(ci1.mle['fn'], fn_mle)
    assert np.allclose(ci1.errors['fn'], ci0.errors['fn'])
    assert np.allclose(ci1.errors['fn'], ci2.errors['fn'], rtol=5e-2, atol=0.0)
    assert np.allclose(ci2.errors['fn'], ci3.errors['fn'])


def test_mle_flux(mle_result2, powerlaw_fn, powerlaw_flux):
    result = mle_result2
    result.boot(4000)
    alpha_mle = result.mle['PowerLaw.alpha'][0]
    K_mle = result.mle['PowerLaw.K'][0]
    emin = 1.0
    emax = 10.0

    flux_mle = powerlaw_fn(alpha_mle, K_mle, jnp.array([emin, emax]))[0]
    ci1 = result.flux(emin, emax, energy=False, method='profile')
    ci2 = result.flux(emin, emax, energy=False, method='boot')
    assert np.allclose(ci1.mle['simulation'].value, flux_mle)
    assert np.allclose(
        ci2.errors['simulation'][0].value,
        ci1.errors['simulation'][0].value,
        rtol=5e-2,
        atol=0.0,
    )
    assert np.allclose(
        ci2.errors['simulation'][1].value,
        ci1.errors['simulation'][1].value,
        rtol=5e-2,
        atol=0.0,
    )

    eflux_mle = powerlaw_flux(alpha_mle, K_mle, emin, emax)
    ci1 = result.flux(emin, emax, energy=True, method='profile')
    ci2 = result.flux(emin, emax, energy=True, method='boot')
    assert np.allclose(ci1.mle['simulation'].value, eflux_mle)
    assert np.allclose(
        ci2.errors['simulation'][0].value,
        ci1.errors['simulation'][0].value,
        rtol=5e-2,
        atol=0.0,
    )
    assert np.allclose(
        ci2.errors['simulation'][1].value,
        ci1.errors['simulation'][1].value,
        rtol=5e-2,
        atol=0.0,
    )


def test_mle_lumin(mle_result2):
    result = mle_result2
    pl = PowerLaw().compile()
    result.boot(4000)
    params_mle = {k: v[0] for k, v in result.mle.items()}
    emin = 1.0
    emax = 10.0
    z = 1.0

    lumin_mle = pl.lumin(emin, emax, z, params=params_mle)
    ci1 = result.lumin(emin, emax, z, method='profile')
    ci2 = result.lumin(emin, emax, z, method='boot')
    assert np.allclose(ci1.mle['simulation'].value, lumin_mle.value)
    assert np.allclose(
        ci2.errors['simulation'][0].value,
        ci1.errors['simulation'][0].value,
        rtol=5e-2,
        atol=0.0,
    )
    assert np.allclose(
        ci2.errors['simulation'][1].value,
        ci1.errors['simulation'][1].value,
        rtol=5e-2,
        atol=0.0,
    )


def test_mle_eiso(simulation, mle_result2):
    result = mle_result2
    pl = PowerLaw().compile()
    result.boot(4000)
    params_mle = {k: v[0] for k, v in result.mle.items()}
    emin = 1.0
    emax = 10.0
    z = 1.0
    expo = simulation.spec_exposure

    eiso_mle = pl.eiso(emin, emax, z, expo, params=params_mle)
    ci1 = result.eiso(emin, emax, z, expo, method='profile')
    ci2 = result.eiso(emin, emax, z, expo, method='boot')
    assert np.allclose(ci1.mle['simulation'].value, eiso_mle.value)
    assert np.allclose(
        ci2.errors['simulation'][0].value,
        ci1.errors['simulation'][0].value,
        rtol=5e-2,
        atol=0.0,
    )
    assert np.allclose(
        ci2.errors['simulation'][1].value,
        ci1.errors['simulation'][1].value,
        rtol=5e-2,
        atol=0.0,
    )


def test_posterior_result(simulation, posterior_result):
    data = simulation
    result = posterior_result

    nbins = len(data.channel)

    # Check various methods of posterior result
    assert result.ndata['total'] == nbins
    assert result.dof == nbins - 2
    result.ppc(1009)
    result.summary()
    result.plot()
    result.plot('data ne ene eene Fv vFv rq pit corner khat trace')
    _ = result.deviance
    _ = result.loo
    _ = result.waic
    assert all(i > 0.05 for i in result.gof.values())

    plotter = result.plot
    plotter.plot_qq('rd')
    plotter.plot_qq('rp')
    plotter.plot_qq('rq')


def test_posterior_covar(
    posterior_result, mle_result2_covar, powerlaw_fn, powerlaw_flux
):
    result = posterior_result
    alpha_mle = result.mle['PowerLaw.alpha'][0]
    K_mle = result.mle['PowerLaw.K'][0]
    params_mle = jnp.array([alpha_mle, K_mle])
    covar_true = mle_result2_covar[0](params_mle)

    covar = result.covar().matrix
    assert np.allclose(covar, covar_true, rtol=1e-1, atol=0.0)

    flux = lambda params: powerlaw_flux(
        params['PowerLaw.alpha'] - 1.0, params['PowerLaw.K'], 1.0, 10.0
    )

    covar_true = mle_result2_covar[1](
        params_mle,
        lambda x: flux(dict(zip(['PowerLaw.alpha', 'PowerLaw.K'], x))),
    )

    covar = result.covar(fn={'flux': flux}).matrix
    assert np.allclose(covar, covar_true, rtol=1e-1, atol=0.0)

    covar_ = result.covar(fn={'flux': flux}, parallel=False).matrix
    assert np.allclose(covar, covar_)


def test_posterior_ci_fn(posterior_result, powerlaw_flux):
    result = posterior_result
    alpha_mle = result.mle['PowerLaw.alpha'][0]
    K_mle = result.mle['PowerLaw.K'][0]
    emin = 1.0
    emax = 10.0
    flux_mle = powerlaw_flux(alpha_mle, K_mle, emin, emax)
    fn = lambda params: powerlaw_flux(
        params['PowerLaw.alpha'], params['PowerLaw.K'], emin, emax
    )

    ci1 = result.ci(params=[], fn={'fn': fn})
    ci2 = result.ci(params=[], fn={'fn': fn}, parallel=False)
    assert np.allclose(ci1.median['fn'], flux_mle, rtol=1e-2, atol=0.0)
    assert np.allclose(ci1.errors['fn'], ci2.errors['fn'])


def test_posterior_flux(posterior_result, powerlaw_fn, powerlaw_flux):
    result = posterior_result
    alpha_mle = result.mle['PowerLaw.alpha'][0]
    K_mle = result.mle['PowerLaw.K'][0]
    emin = 1.0
    emax = 10.0
    flux_mle = powerlaw_fn(alpha_mle, K_mle, jnp.array([emin, emax]))[0]
    ci1 = result.flux(emin, emax, energy=False)
    assert np.allclose(
        ci1.median['simulation'].value, flux_mle, rtol=1e-2, atol=0.0
    )

    eflux_mle = powerlaw_flux(alpha_mle, K_mle, emin, emax)
    ci1 = result.flux(emin, emax, energy=True)
    assert np.allclose(
        ci1.median['simulation'].value, eflux_mle, rtol=1e-2, atol=0.0
    )


def test_posterior_lumin(posterior_result):
    result = posterior_result
    pl = PowerLaw().compile()
    params_mle = {k: v[0] for k, v in result.mle.items()}
    emin = 1.0
    emax = 10.0
    z = 1.0

    lumin_mle = pl.lumin(emin, emax, z, params=params_mle)
    ci1 = result.lumin(emin, emax, z)
    assert np.allclose(
        ci1.median['simulation'].value, lumin_mle.value, rtol=1e-2, atol=0.0
    )


def test_posterior_eiso(simulation, posterior_result):
    result = posterior_result
    pl = PowerLaw().compile()
    params_mle = {k: v[0] for k, v in result.mle.items()}
    emin = 1.0
    emax = 10.0
    z = 1.0
    expo = simulation.spec_exposure

    eiso_mle = pl.eiso(emin, emax, z, expo, params=params_mle)
    ci1 = result.eiso(emin, emax, z, expo)
    assert np.allclose(
        ci1.median['simulation'].value, eiso_mle.value, rtol=1e-2, atol=0.0
    )
