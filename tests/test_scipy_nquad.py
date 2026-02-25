import sys
import types

import numpy as np


fake_nb = types.SimpleNamespace(
    types=types.SimpleNamespace(
        double=lambda *args, **kwargs: None,
        intc=object(),
        CPointer=lambda x: None,
    ),
    cfunc=lambda sig: (lambda f: types.SimpleNamespace(ctypes=f)),
    njit=lambda f: f,
    carray=lambda params, shape: params,
)
sys.modules.setdefault('numba', fake_nb)

import elisa.util.scipy_nquad as scipy_nquad


def _setup_callback_mocks(monkeypatch, jax_version: str, expected_kwargs: dict):
    monkeypatch.setattr(scipy_nquad.jax, '__version__', jax_version)
    monkeypatch.setattr(scipy_nquad.jax, 'jit', lambda f: f)
    monkeypatch.setattr(
        scipy_nquad.jax,
        'ShapeDtypeStruct',
        lambda shape, dtype: {'shape': shape, 'dtype': dtype},
    )
    monkeypatch.setattr(scipy_nquad, 'LowLevelCallable', lambda cfun: cfun)
    monkeypatch.setattr(
        scipy_nquad.integrate,
        'nquad',
        lambda fun, ranges, args, opts, full_output=False: (42.0, 0.5),
    )

    captured = {}

    def fake_pure_callback(cb, result_shape_dtype, ranges, args, **kwargs):
        captured['kwargs'] = kwargs
        captured['result_shape_dtype'] = result_shape_dtype
        return cb(ranges, args)

    monkeypatch.setattr(scipy_nquad.jax, 'pure_callback', fake_pure_callback)

    out = scipy_nquad.NQuadTransform._nquad(
        cfun=object(), opts=None, vectorized=True
    )([[0.0, 1.0]], [1.0, 2.0])

    assert captured['kwargs'] == expected_kwargs
    assert captured['result_shape_dtype']['shape'] == (2,)
    assert np.allclose(np.asarray(out), np.asarray([42.0, 0.5]))


def test_nquad_transform_uses_vmap_method_for_jax_gte_060(monkeypatch):
    _setup_callback_mocks(
        monkeypatch,
        jax_version='0.6.0',
        expected_kwargs={'vmap_method': 'sequential'},
    )


def test_nquad_transform_uses_vectorized_for_jax_lt_060(monkeypatch):
    _setup_callback_mocks(
        monkeypatch,
        jax_version='0.5.9',
        expected_kwargs={'vectorized': True},
    )
