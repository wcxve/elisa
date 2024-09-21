"""Helper for using scipy.integrate.nquad in JAX.

Contributed by @xiesl97 (https://github.com/xiesl97).
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy import LowLevelCallable, integrate

try:
    import numba as nb
except ImportError as e:
    raise ImportError(
        'To use this module, please install `numba` package. It can be'
        ' installed with `pip install numba`'
    ) from e


class NQuadTransform:
    def __init__(self, fun):
        self._c_sig = nb.types.double(
            nb.types.intc, nb.types.CPointer(nb.types.double)
        )
        self._fun = fun

    def _cfun(self):
        fun = self._fun

        @nb.cfunc(self._c_sig)
        def cfun(n, _params):
            params = nb.carray(_params, (n,))
            return fun(params)

        return cfun.ctypes

    @staticmethod
    def _nquad(cfun, opts=None, vectorized=False):
        @jax.jit
        def _nquad_scipy(ranges, args):
            ranges = jnp.asarray(ranges)
            args = jnp.asarray(args)

            def _pcb(ranges, args):
                fun_sci_call = LowLevelCallable(cfun)
                result, abserr = integrate.nquad(
                    fun_sci_call, ranges, args, opts, full_output=False
                )
                return jnp.asarray([result, abserr])

            result_shape_dtype = jax.ShapeDtypeStruct(
                shape=(2,), dtype=ranges.dtype
            )
            return jax.pure_callback(
                _pcb, result_shape_dtype, ranges, args, vectorized=vectorized
            )

        return _nquad_scipy


if __name__ == '__main__':
    """example 1"""

    @nb.njit
    def f(params):
        x, y, z, d = params
        return np.exp(-(x**2)) + y + z * d

    # integrate variables x and y
    ranges = jnp.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=jnp.float64)
    # pass constant to z and d
    args = jnp.asarray([3.0, 4.0], dtype=jnp.float64)
    # transform the function
    nqt = NQuadTransform(f)
    cfun = nqt._cfun()
    func_nquad = jax.jit(nqt._nquad(cfun, opts=None, vectorized=False))
    # print result and error
    print(func_nquad(ranges, args))

    """example 2"""
    from elisa.models.model import AnaIntAdditive, ParamConfig
    from elisa.util.misc import define_fdjvp
    from elisa.util.typing import JAXArray, NameValMapping

    # blackbody model
    @nb.njit
    def bbodyrad(params):
        e, kT, K = params
        return 1.0344e-3 * K * e * e / np.expm1(e / kT)

    # transform the model
    nqt_bbodyrad = NQuadTransform(bbodyrad)
    nqt_bbodyrad_cfun = nqt_bbodyrad._cfun()
    bbodyrad_nquad = jax.jit(
        nqt_bbodyrad._nquad(nqt_bbodyrad_cfun, opts=None, vectorized=False)
    )

    # test integrate
    ranges = jnp.asarray([[0.0, 1.0]], dtype=jnp.float64)
    args = jnp.asarray([2.0, 3.0], dtype=jnp.float64)
    print(bbodyrad_nquad(ranges, args))

    @jax.jit
    def bboduyrad_flux(ranges, args):
        return bbodyrad_nquad(ranges, args)[0]  # return result

    class BB_test(AnaIntAdditive):
        _config = (
            ParamConfig('kT', 'kT', 'keV', 3.0, 1e-4, 200.0),
            ParamConfig('K', 'K', '', 1.0, 1e-10, 1e10),
        )

        @staticmethod
        def integral(egrid: JAXArray, params: NameValMapping) -> JAXArray:
            kT = params['kT']
            K = params['K']

            # integrate energy grids
            ranges = jnp.asarray([egrid[:-1], egrid[1:]], dtype=jnp.float64).T
            ranges = jnp.reshape(ranges, (len(ranges), 1, 2))
            args = jnp.asarray([kT, K], dtype=jnp.float64)

            return jax.vmap(bboduyrad_flux, in_axes=(0, None))(ranges, args)

    # define numerical integration for model fit
    BB_test.integral = define_fdjvp(BB_test.integral, method='forward')
