import jax.numpy as jnp
import numpy as np
import pytest

from elisa.util.integrate import make_integral_factory


@pytest.mark.parametrize(
    'method',
    ('quadgk', 'quadcc', 'quadts', 'romberg', 'rombergts'),
)
def test_integral_methods(method):
    def model_fn(egrid, params):
        return jnp.full_like(egrid, params['x'] ** 2)

    factory = make_integral_factory(
        'x',
        jnp.array([0.0, 1.0]),
        method=method,
    )
    result = factory(model_fn)(jnp.ones(1), {})

    np.testing.assert_allclose(result, 1.0 / 3.0, rtol=1e-5)
