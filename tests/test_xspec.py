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
        return []

    return [
        'feklor',  # test for additive models with norm param only
        'posm',  # test for additive models with norm param only
        'tbabs',  # test multiplicative model
        'clumin',  # test convolution model (for additive models)
        'partcov',  # test convolution model (for multiplicative models)
        'vmshift',  # test convolution model (for multiplicative models)
        'zmshift',  # test convolution model (for multiplicative models)
    ]


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
