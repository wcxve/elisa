"""Models of additive type."""
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

    @abstractmethod
    def _integral(self, *args) -> Callable:
        """Overriden by subclass."""
        pass


class NumIntAdditive(AdditiveComponent, ABC):
    """Prototype class with numerical integral to define additive model."""

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

    @property
    def _integral(self) -> Callable:
        """Wrap continnum function with numerical integral method."""
        name = self.__class__.__name__.lower()
        f = integral(self._continnum, name, self._method)
        return f

    @staticmethod
    @abstractmethod
    def _continnum(*args, **kwargs):
        """Continnum to be integrated over energy grids."""
        pass


class BlackBody(NumIntAdditive):
    """TODO: docstring"""

    _default = (
        ('kT', 'kT', 3.0, 0.0001, 200.0, False, False),
        ('K', 'K', 1.0, 1e-10, 1e10, False, False),
    )

    @staticmethod
    def _continnum(egrid, kT, K):
        e = egrid
        return K * 8.0525 * e * e / (kT * kT * kT * kT * jnp.expm1(e / kT))


class Powerlaw(AdditiveComponent):
    """TODO"""

    _default = (
        ('alpha', r'\alpha', 1.01, -3.0, 10.0, False, False),
        ('K', 'K', 1.0, 1e-10, 1e10, False, False),
    )

    @staticmethod
    def _integral(egrid, alpha, K):
        # we ignore the case of PhoIndex = 1.0
        tmp = 1.0 - alpha
        f = K / tmp * jnp.power(egrid, tmp)
        return f[1:] - f[:-1]
