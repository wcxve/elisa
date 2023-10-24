"""Spectral components with an additive type."""

from abc import ABC

import jax.numpy as jnp

from .base import SpectralComponentABC
from .integral import integral


class AdditiveComponentABC(SpectralComponentABC, ABC):
    type: str = 'add'


class NumIntAdditiveABC(AdditiveComponentABC, ABC):
    """TODO: docstring"""

    extra_kwargs = (('method', 'default'),)

    def __init__(self, *, method='default', **kwargs):
        # wrap integrand with numerical integral method
        self._eval = integral(self._eval, method)

        super().__init__(**kwargs)


class BlackBody(NumIntAdditiveABC):
    """TODO: docstring"""

    fmt: str = 'BB'
    default = (
        ('kT', '$kT$', 3.0, 0.0001, 200.0, False, False),
        ('norm', 'norm', 1.0, 1e-10, 1e10, False, False),
    )

    @staticmethod
    def _eval(e, kT, norm):
        return norm * 8.0525 * e * e / (kT * kT * kT * kT * jnp.expm1(e / kT))


class Powerlaw(AdditiveComponentABC):
    """TODO"""

    fmt: str = 'PL'
    default = (
        ('PhoIndex', r'$\alpha$', 1.01, -3.0, 10.0, False, False),
        ('norm', 'norm', 1.0, 1e-10, 1e10, False, False),
    )

    @staticmethod
    def _eval(e, PhoIndex, norm):
        tmp = 1.0 - PhoIndex
        f = norm / tmp * jnp.power(e, tmp)
        return f[1:] - f[:-1]
