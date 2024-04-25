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

    def _args_grad(self, h=1e-7):
        cfun = self._cfun()
        func_nquad = jax.jit(self._nquad(cfun))

        @jax.custom_jvp
        @jax.jit
        def _fn(ranges, args):
            return func_nquad(ranges, args)[0]

        @_fn.defjvp
        @jax.jit
        def _fn_jvp(primals, tangents):
            ranges, args = primals
            ranges_dot, args_dots = tangents
            primal_out = _fn(ranges, args)
            args_h = args + jnp.eye(len(args)) * h
            primal_dx = jax.vmap(_fn, in_axes=(None, 0))(ranges, args_h)
            primal_grad_dx = (primal_dx - primal_out) / h
            tangent_out = jnp.sum(primal_grad_dx * args_dots)
            return primal_out, tangent_out

        return _fn


if __name__ == '__main__':
    """example 1"""

    @nb.njit
    def f(params):
        x, y, z, d = params
        return np.exp(-(x**2)) + y + z * d

    ranges = jnp.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=jnp.float64)
    args = jnp.asarray([3.0, 4.0], dtype=jnp.float64)
    nqt = NQuadTransform(f)
    cfun = nqt._cfun()
    func_nquad = jax.jit(nqt._nquad(cfun, opts=None, vectorized=False))
    func_arg_grad = nqt._args_grad()
    print(func_nquad(ranges, args))
    print(jax.grad(func_arg_grad)(ranges, args))

    """example 2"""
    from elisa.models.model import AnaIntAdditive, ParamConfig
    from elisa.util.typing import JAXArray, NameValMapping

    @nb.njit
    def bbodyrad(params):
        e, kT, K = params
        return 1.0344e-3 * K * e * e / np.expm1(e / kT)

    nqt = NQuadTransform(bbodyrad)
    func_arg_grad = nqt._args_grad()

    class BlackbodyRad_test(AnaIntAdditive):
        _config = (
            ParamConfig('kT', 'kT', 'keV', 3.0, 1e-4, 200.0),
            ParamConfig('K', 'K', '', 1.0, 1e-10, 1e10),
        )

        @staticmethod
        def integral(egrid: JAXArray, params: NameValMapping) -> JAXArray:
            kT = params['kT']
            K = params['K']

            ranges = jnp.asarray([egrid[:-1], egrid[1:]], dtype=jnp.float64).T
            ranges = jnp.reshape(ranges, (len(ranges), 1, 2))
            args = jnp.asarray([kT, K], dtype=jnp.float64)

            return jax.vmap(func_arg_grad, in_axes=(0, None))(ranges, args)
