import os

import jax.numpy as jnp
import numpy as np
import pytest

from elisa.models import Constant, PowerLaw
from elisa.models.xspec import XspecConvolution, generate_xspec_models

egrid = jnp.linspace(0.1, 10, 101)
const = Constant()
pl = PowerLaw()
excluded = ['XSsmaug']
# There are 74 models depend on Xspec model data
excluded_when_no_xspec_data = [
    'XSbapec',
    'XSbcheb6',
    'XSbcie',
    'XSbcoolflow',
    'XSbcph',
    'XSbexpcheb6',
    'XSbexriv',
    'XSbvapec',
    'XSbvcempow',
    'XSbvcheb6',
    'XSbvcie',
    'XSbvcoolflow',
    'XSbvcph',
    'XSbvgadem',
    'XSbvvapec',
    'XSbvvcie',
    'XSbvvgadem',
    'XSbvvwdem',
    'XSbwdem',
    'XSc6mekl',
    'XSc6pmekl',
    'XSc6pvmkl',
    'XSc6vmekl',
    'XScempow',
    'XScevmkl',
    'XScflow',
    'XScheb6',
    'XScie',
    'XScompth',
    'XScoolflow',
    'XScph',
    'XSeqpair',
    'XSeqtherm',
    'XSexpcheb6',
    'XSgadem',
    'XSismdust',
    'XSkdblur',
    'XSkdblur2',
    'XSkerrbb',
    'XSkerrd',
    'XSlaor',
    'XSlaor2',
    'XSmeka',
    'XSmkcflow',
    'XSnlapec',
    'XSnsmaxg',
    'XSnsx',
    'XSpexriv',
    'XSrefsch',
    'XSrfxconv',
    'XStapec',
    'XSvapec',
    'XSvcempow',
    'XSvcheb6',
    'XSvcie',
    'XSvcoolflow',
    'XSvcph',
    'XSvexpcheb6',
    'XSvgadem',
    'XSvmcflow',
    'XSvmeka',
    'XSvtapec',
    'XSvvapec',
    'XSvvcie',
    'XSvvgadem',
    'XSvvtapec',
    'XSvvwdem',
    'XSvwdem',
    'XSwdem',
    'XSxilconv',
    'XSxscat',
    'XSzkerrbb',
    'XSzpowerlw',
    'XSzvgauss',
]


@pytest.fixture()
def has_xspec_data():
    HEADAS = os.environ.get('HEADAS', '')

    if not HEADAS:
        return False

    path = os.path.abspath(f'{HEADAS}/../spectral/modelData')
    path_alt = os.path.abspath(f'{HEADAS}/spectral/modelData')

    return os.path.exists(path) or os.path.exists(path_alt)


@pytest.mark.parametrize(
    'model',
    [pytest.param(v, id=k) for k, v in generate_xspec_models().items()],
)
def test_xspec_model_eval(model, has_xspec_data):
    name = model.__name__

    if name in excluded:
        pytest.skip(f'skip {name} as it is excluded')

    if (not has_xspec_data) and (name in excluded_when_no_xspec_data):
        pytest.skip(f'skip {name} as Xspec model data is missing')

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
