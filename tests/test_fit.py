import sys
import types
from importlib.util import find_spec

import numpy as np
import pytest

from elisa import BayesFit, MaxLikeFit
from elisa.models import PowerLaw

DYNESTY_SKIP_MARK = pytest.mark.skipif(
    not bool(find_spec('dynesty')),
    reason='dynesty is not installed',
)


JAXNS_XFAIL_MARK = pytest.mark.xfail(
    not bool(find_spec('jaxns')) and sys.version_info >= (3, 14),
    reason='jaxns==2.6.9 is incompatible with python>=3.14',
)


@pytest.mark.parametrize(
    'method',
    [
        pytest.param('minuit', id='iminuit'),
        pytest.param('lm', id='optimistix.LevenbergMarquardt'),
        pytest.param('ns', marks=JAXNS_XFAIL_MARK, id='JAXNS'),
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
        # NumPyro samplers
        pytest.param('nuts', {}, id='NUTS'),
        pytest.param('barkermh', {}, id='BarkerMH'),
        pytest.param('blackjax_nuts', {}, id='BlackJAX_NUTS'),
        pytest.param('sa', {'warmup': 20000}, id='SA'),
        pytest.param('aies', {}, id='AIES'),
        pytest.param('aies', {'n_parallel': 1}, id='AIES_1'),
        pytest.param('ess', {}, id='ESS'),
        pytest.param('ess', {'n_parallel': 1}, id='ESS_1'),
        # JAX backend nested sampler
        pytest.param('jaxns', {}, marks=JAXNS_XFAIL_MARK, id='JAXNS'),
        # Non-JAX backends samplers
        pytest.param('emcee', {}, id='emcee'),
        pytest.param('emcee', {'n_parallel': 1}, id='emcee_1'),
        pytest.param('zeus', {'steps': 2000}, id='zeus'),
        pytest.param('zeus', {'steps': 2000, 'n_parallel': 1}, id='zeus_1'),
        # Non-JAX backends nested samplers
        pytest.param('nautilus', {}, id='Nautilus'),
        pytest.param('ultranest', {}, id='UltraNest'),
        pytest.param(
            'dynesty',
            {'termination_kwargs': {'maxcall': 20000}},
            marks=DYNESTY_SKIP_MARK,
            id='Dynesty',
        ),
    ],
)
def test_trivial_bayes_fit(simulation, method, options):
    data = simulation
    model = PowerLaw()
    model.PowerLaw.K.log = True

    # SA seems to converge randomly, which is really frustrating
    # we try to fix this by better init and seed of 100...
    if method == 'sa':
        model['PowerLaw']['alpha'].default = 0.0
        model['PowerLaw']['K'].default = 10.0

    # check the global random state of numpy is unaffected after the fit
    np.random.seed(0)
    test_rand = np.random.rand()
    np.random.seed(0)

    fit_fn = getattr(BayesFit(data, model, seed=100), method)

    # Bayesian fit result, i.e. posterior
    result = fit_fn(**options)

    # test refit with given post_warmup_state
    if result.sampler_state is not None:
        options['steps'] = 100
        fit_fn(post_warmup_state=result.sampler_state, **options)

    # check convergence
    assert all(i < 1.01 for i in result.rhat.values() if not np.isnan(i))

    # check the true parameters values are within the 68% CI
    ci = result.ci(cl=1).intervals
    assert ci['PowerLaw.K'][0] < 10.0 < ci['PowerLaw.K'][1]
    assert ci['PowerLaw.alpha'][0] < 0.0 < ci['PowerLaw.alpha'][1]

    # check the global random state of numpy is unaffected after the fit
    assert np.allclose(np.random.rand(), test_rand)


@pytest.mark.parametrize(
    'dynamic, expected',
    [
        pytest.param(False, None, id='Dynesty_static'),
        pytest.param(True, 3000, id='Dynesty_dynamic'),
    ],
)
def test_dynesty_termination_kwargs_dispatch(simulation, dynamic, expected):
    captured = {}
    original_generate_results = BayesFit._generate_results

    class FakeDynestySampler:
        def __init__(self, *args, dynamic: bool = False, **kwargs):
            captured['dynamic'] = dynamic
            self._ess = 8
            self._closed = False

        def run(self, **kwargs):
            captured['run_kwargs'] = kwargs
            return {
                'PowerLaw.K': np.array([10.0, 10.1]),
                'PowerLaw.alpha': np.array([0.0, 0.1]),
            }

        def print_results(self):
            captured['printed'] = True

        @property
        def ess(self):
            return self._ess

        @property
        def lnZ(self):
            return 0.0, 0.1

        def close(self):
            self._closed = True
            captured['closed'] = captured.get('closed', 0) + 1

    original = sys.modules.get('elisa.infer.samplers.ns.dynesty')
    sys.modules['elisa.infer.samplers.ns.dynesty'] = types.SimpleNamespace(
        DynestySampler=FakeDynestySampler,
    )

    try:

        def fake_generate_results(self, **kwargs):
            captured['generated'] = kwargs
            return types.SimpleNamespace(lnZ=kwargs['lnZ'])

        BayesFit._generate_results = fake_generate_results
        data = simulation
        model = PowerLaw()
        model.PowerLaw.K.log = True
        result = BayesFit(data, model, seed=100).dynesty(dynamic=dynamic)
    finally:
        BayesFit._generate_results = original_generate_results
        if original is None:
            del sys.modules['elisa.infer.samplers.ns.dynesty']
        else:
            sys.modules['elisa.infer.samplers.ns.dynesty'] = original

    assert captured['dynamic'] is dynamic
    assert captured.get('printed') is True
    assert captured.get('closed') == 1
    if expected is None:
        assert 'n_effective' not in captured['run_kwargs']
    else:
        assert captured['run_kwargs']['n_effective'] == expected
    assert result.lnZ[0] == 0.0


@pytest.mark.parametrize('fail_stage', ['run', 'print', 'generate'])
def test_dynesty_closes_sampler_on_failure(simulation, fail_stage):
    captured = {'closed': 0}
    original_generate_results = BayesFit._generate_results

    class FakeDynestySampler:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, **kwargs):
            if fail_stage == 'run':
                raise RuntimeError('run failed')
            return {
                'PowerLaw.K': np.array([10.0, 10.1]),
                'PowerLaw.alpha': np.array([0.0, 0.1]),
            }

        def print_results(self):
            if fail_stage == 'print':
                raise RuntimeError('print failed')

        @property
        def ess(self):
            return 2

        @property
        def lnZ(self):
            return 0.0, 0.1

        def close(self):
            captured['closed'] += 1

    original = sys.modules.get('elisa.infer.samplers.ns.dynesty')
    sys.modules['elisa.infer.samplers.ns.dynesty'] = types.SimpleNamespace(
        DynestySampler=FakeDynestySampler,
    )

    try:
        if fail_stage == 'generate':

            def fake_generate_results(self, **kwargs):
                raise RuntimeError('generate failed')

            BayesFit._generate_results = fake_generate_results

        data = simulation
        model = PowerLaw()
        model.PowerLaw.K.log = True
        with pytest.raises(RuntimeError, match=f'{fail_stage} failed'):
            BayesFit(data, model, seed=100).dynesty()
    finally:
        BayesFit._generate_results = original_generate_results
        if original is None:
            del sys.modules['elisa.infer.samplers.ns.dynesty']
        else:
            sys.modules['elisa.infer.samplers.ns.dynesty'] = original

    assert captured['closed'] == 1


@DYNESTY_SKIP_MARK
def test_dynesty_sampler_restores_rng_state_on_run_exception(monkeypatch):
    from elisa.infer.samplers.ns import dynesty as dynesty_module

    class FakeModelInfo:
        ndim = 1

        @staticmethod
        def log_prob_fn(x):
            return 0.0

        @staticmethod
        def unravel(x):
            return x

        @staticmethod
        def postprocess_fn(x):
            return {'PowerLaw.K': x, 'PowerLaw.alpha': x}

    class FailingNestedSampler:
        def __init__(self, *args, **kwargs):
            pass

        def run_nested(self, **kwargs):
            raise RuntimeError('run_nested failed')

    monkeypatch.setattr(
        dynesty_module,
        'uniform_reparam_model',
        lambda *args, **kwargs: FakeModelInfo(),
    )
    monkeypatch.setattr(dynesty_module, 'NestedSampler', FailingNestedSampler)

    np.random.seed(0)
    rand_ref = np.random.rand()
    np.random.seed(0)

    sampler = dynesty_module.DynestySampler(
        numpyro_model=lambda: None,
        seed=123,
    )
    with pytest.raises(RuntimeError, match='run_nested failed'):
        sampler.run(print_progress=False)

    assert np.allclose(np.random.rand(), rand_ref)


def test_dynesty_sampler_close_idempotent():
    from elisa.infer.samplers.ns import dynesty as dynesty_module

    class FakePool:
        def __init__(self):
            self.closed = 0
            self.joined = 0

        def close(self):
            self.closed += 1

        def join(self):
            self.joined += 1

    pool = FakePool()
    sampler = dynesty_module.DynestySampler.__new__(
        dynesty_module.DynestySampler
    )
    sampler._pool = pool

    sampler.close()
    sampler.close()

    assert pool.closed == 1
    assert pool.joined == 1
    assert sampler._pool is None
