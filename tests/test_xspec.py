import os

import numpy as np
import pytest

from elisa.models import Constant, PowerLaw, xspec as xs

egrid = np.linspace(0.1, 10, 100)
const = Constant()
pl = PowerLaw()


def get_test_models():
    HEADAS = os.environ.get('HEADAS', '')

    if not HEADAS:
        return False

    path = os.path.abspath(f'{HEADAS}/../spectral/modelData')
    path_alt = os.path.abspath(f'{HEADAS}/spectral/modelData')
    has_xspec_data = os.path.exists(path) or os.path.exists(path_alt)

    models = xs.generate_xspec_models()
    if not has_xspec_data:
        models_to_test = [
            'XSfeklor',  # test for additive models with norm param only
            'XSposm',  # test for additive models with norm param only
            'XStbabs',  # test multiplicative model
            'XSclumin',  # test convolution model (for additive models)
            'XSpartcov',  # test convolution model (for multiplicative models)
            'XSvmshift',  # test convolution model (for multiplicative models)
            'XSzmshift',  # test convolution model (for multiplicative models)
        ]
        models = {i: models[i] for i in models_to_test}
    else:
        models.pop('XSsmaug')

        # Fix nei models for XSPEC v12.15
        xspec_version = xs.version()
        major, minor, _ = xspec_version.split('.')
        if major == '12' and minor == '15':
            xs.set_model_string('NEIVERS', '3.1.2')

    return models


@pytest.mark.parametrize(
    'model',
    [pytest.param(v, id=k) for k, v in get_test_models().items()],
)
def test_xspec_model_eval(model):
    if not issubclass(model, xs.XspecConvolution):
        # Additive or multiplicative models
        assert np.all(np.isfinite(model().compile().eval(egrid)))
    else:
        if 'add' in model._supported:
            # Convolution models to be applied for additive models
            assert np.all(np.isfinite(model()(pl).compile().eval(egrid)))
        else:
            # Convolution models to be applied for multiplicative models
            assert np.all(np.isfinite(model()(const).compile().eval(egrid)))
