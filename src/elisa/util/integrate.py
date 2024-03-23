"""Numerical integration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, get_args

import jax.numpy as jnp
from quadax import quadcc, quadgk, quadts, romberg, rombergts

if TYPE_CHECKING:
    from typing import Any, Callable

    from elisa.util.typing import (
        JAXArray,
        JAXFloat,
        ModelCompiledFn,
        ParamID,
        ParamIDValMapping,
    )

    IntegralFactory = Callable[[ModelCompiledFn], ModelCompiledFn]

AdaptQuadMethod = Literal['quadgk', 'quadcc', 'quadts', 'romberg', 'rombergts']
_QUAD_FN = dict(
    zip(
        get_args(AdaptQuadMethod), [quadgk, quadcc, quadts, romberg, rombergts]
    )
)


def make_integral_factory(
    param_id: ParamID,
    interval: JAXArray,
    method: AdaptQuadMethod = 'quadgk',
    kwargs: dict[str, Any] | None = None,
) -> Callable[[ModelCompiledFn], ModelCompiledFn]:
    """Get integral factory over the interval.

    Parameters
    ----------
    param_id : str
        Parameter ID.
    interval: array_like
        The interval, a 2-element sequence.
    method : {'quadgk', 'quadcc', 'quadts', 'romberg', 'rombergts'}, optional
        Numerical integration method used to integrate over the parameter.
        Available options are:

            * ``'quadgk'``: global adaptive quadrature by Gauss-Konrod rule
            * ``'quadcc'``: global adaptive quadrature by Clenshaw-Curtis rule
            * ``'quadts'``: global adaptive quadrature by trapz tanh-sinh rule
            * ``'romberg'``: Romberg integration
            * ``'rombergts'``: Romberg integration by tanh-sinh
              (a.k.a. double exponential) transformation

        The default is ``'quadgk'``.
    kwargs : dict, optional
        Extra kwargs passed to integration methods. See [1]_ for details.

    Returns
    -------
    integral_factory : callable
        Given a model function, the integral factory outputs a new model
        function with the interval parameter being integrated out.

    References
    ----------
    .. [1] `quadax docs <https://quadax.readthedocs.io/en/latest/api.html#adaptive-integration-of-a-callable-function-or-method>`__
    """
    if method not in _QUAD_FN:
        raise ValueError(f'unsupported method: {method}')

    if jnp.shape(interval) != (2,):
        raise ValueError('interval must be sequence of length 2')

    quad = _QUAD_FN[method]
    interval = jnp.asarray(interval, float)
    kwargs = dict(kwargs) if kwargs is not None else {}

    def integral_factory(model_fn: ModelCompiledFn) -> ModelCompiledFn:
        """Integrate the model_fn over the interval."""

        def integrand(
            value: JAXFloat,
            egrid: JAXArray,
            params: ParamIDValMapping,
        ) -> JAXArray:
            """The integrand function."""
            params[param_id] = value
            return model_fn(egrid, params)

        def integral(egrid: JAXArray, params: ParamIDValMapping) -> JAXArray:
            """New model_fn with interval param being integrated out."""
            args = (egrid, params)
            quad_result = quad(integrand, interval, args, **kwargs)[0]
            # if the integrand is independent of the interval parameter,
            # then the result is the same due to the 1/(b-a) factor
            return quad_result / (interval[1] - interval[0])

        return integral

    return integral_factory
