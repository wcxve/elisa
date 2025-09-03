import os

import numpy as np
import pytest

from elisa.models import Constant, PowerLaw, xs

egrid = np.linspace(0.1, 10, 100)
const = Constant()
pl = PowerLaw()


def get_test_models():
    HEADAS = os.environ.get('HEADAS', '')

    if not HEADAS:
        pytest.skip('HEADAS environment variable is not set')

    path = os.path.abspath(f'{HEADAS}/../spectral/modelData')
    path_alt = os.path.abspath(f'{HEADAS}/spectral/modelData')
    has_xspec_data = os.path.exists(path) or os.path.exists(path_alt)

    if not has_xspec_data:
        models = [
            'feklor',  # test for additive models with norm param only
            'posm',  # test for additive models with norm param only
            'tbabs',  # test multiplicative model
            'clumin',  # test convolution model (for additive models)
            'partcov',  # test convolution model (for multiplicative models)
            'vmshift',  # test convolution model (for multiplicative models)
            'zmshift',  # test convolution model (for multiplicative models)
        ]
    else:
        models = xs.list_models()
        models.remove('smaug')

    return models


@pytest.mark.parametrize(
    'name',
    [pytest.param(m, id=m) for m in get_test_models()],
)
def test_xspec_model_eval(name):
    model = getattr(xs, name)
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
