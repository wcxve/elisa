"""Model of additive type."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import jax.numpy as jnp

from .base import Component
from .integral import integral, list_methods


class AdditiveComponent(Component, ABC):
    """Prototype class to define additive component."""

    @property
    def type(self) -> str:
        """Model type is additive."""
        return 'add'

    @property
    def _func(self) -> Callable:
        """Return function that integrates continnum over energy grid."""
        return self._integral

    @property
    @abstractmethod
    def _integral(self) -> Callable:
        """Overriden by subclass."""
        pass


class NumIntAdditive(AdditiveComponent, ABC):
    """Prototype class with numerical integral to define additive component."""

    _extra_kw: tuple[tuple] = (('method', 'default'),)

    def __init__(self, method='default', **kwargs):
        self.method = str(method)
        super().__init__(**kwargs)

    @property
    def method(self) -> str:
        """Numerical integral method."""
        return self._method

    @method.setter
    def method(self, value: str):
        """Numerical integral method."""
        value = str(value)

        methods = list_methods()
        if value not in methods:
            methods = '"' + '", "'.join(methods) + '"'
            raise ValueError(
                f'available numerical integral options are: {methods}, '
                f'but got "{value}"'
            )

        self._method = value

        if hasattr(self, '_node'):
            self._node.attrs['func'] = self._integral

    @property
    def _integral(self) -> Callable:
        """Wrap continnum function with numerical integral method."""
        return integral(self._continnum, self._method)

    @staticmethod
    @abstractmethod
    def _continnum(*args, **kwargs):
        """Continnum to be integrated over energy grids."""
        pass


class BlackBody(NumIntAdditive):
    """TODO: docstring"""

    _default = (
        ('kT', '$kT$', 3.0, 0.0001, 200.0, False, False),
        ('norm', '$K$', 1.0, 1e-10, 1e10, False, False),
    )

    @staticmethod
    def _continnum(egrid, kT, norm):
        e = egrid
        return norm * 8.0525 * e * e / (kT * kT * kT * kT * jnp.expm1(e / kT))


class Powerlaw(AdditiveComponent):
    """TODO"""

    _default = (
        ('PhoIndex', r'$\alpha$', 1.01, -3.0, 10.0, False, False),
        ('norm', '$K$', 1.0, 1e-10, 1e10, False, False),
    )

    @property
    def _integral(self) -> Callable:

        def powerlaw(egrid, PhoIndex, norm):
            """Powerlaw integral."""
            # here we ignore the case of PhoIndex == 1.0
            tmp = 1.0 - PhoIndex
            f = norm / tmp * jnp.power(egrid, tmp)
            return f[1:] - f[:-1]

        return powerlaw
