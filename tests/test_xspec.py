import os

import numpy as np
import pytest

from elisa.models import Constant, PowerLaw
from elisa.models.xspec import XspecConvolution, generate_xspec_models

egrid = np.arange(0.1, 10, 0.01)
const = Constant()
pl = PowerLaw()


def get_test_models():
    HEADAS = os.environ.get('HEADAS', '')

    if not HEADAS:
        return False

    path = os.path.abspath(f'{HEADAS}/../spectral/modelData')
    path_alt = os.path.abspath(f'{HEADAS}/spectral/modelData')
    has_xspec_data = os.path.exists(path) or os.path.exists(path_alt)

    models = generate_xspec_models()
    if not has_xspec_data:
        models_to_test = [
            'XSzpowerlw',
            'XSzvgauss',
            'XStbabs',
            'XSclumin',
            'XSpartcov',
            'XSvmshift',
            'XSzmshift',
        ]
        models = {i: models[i] for i in models_to_test}
    else:
        models.pop('XSsmaug')

    return models


@pytest.mark.parametrize(
    'model',
    [pytest.param(v, id=k) for k, v in get_test_models().items()],
)
def test_xspec_model_eval(model):
    if not issubclass(model, XspecConvolution):
        # Additive or multiplicative models
        assert np.all(np.isfinite(model().compile().eval(egrid)))
    else:
        if 'add' in model._supported:
            # Convolution models to be applied for additive models
            assert np.all(np.isfinite(model()(pl).compile().eval(egrid)))
        else:
            # Convolution models to be applied for multiplicative models
            assert np.all(np.isfinite(model()(const).compile().eval(egrid)))
