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

    def _func_generator(self, func_name: str) -> Callable:
        """Return function that integrates continnum over energy grid."""
        return self._integral_generator(func_name)

    @abstractmethod
    def _integral_generator(self, func_name: str) -> Callable:
        """Overriden by subclass."""
        pass


class AnaIntAdditive(AdditiveComponent, ABC):
    """Prototype class with analytical integral to define additive model."""

    def _integral_generator(self, func_name: str) -> Callable:
        """Copy integral function."""
        return self._integral

    @staticmethod
    @abstractmethod
    def _integral(*args, **kwargs):
        """Analytical integral over egrid, overriden by subclass."""
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

    def _integral_generator(self, func_name: str) -> Callable:
        """Wrap continnum function with numerical integral method."""
        f = integral(self._continnum, func_name, self._method)
        return f

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


class Powerlaw(AnaIntAdditive):
    """TODO"""

    _default = (
        ('PhoIndex', r'$\alpha$', 1.01, -3.0, 10.0, False, False),
        ('norm', '$K$', 1.0, 1e-10, 1e10, False, False),
    )

    @staticmethod
    def _integral(egrid, PhoIndex, norm):
        # here we ignore the case of PhoIndex == 1.0
        tmp = 1.0 - PhoIndex
        f = norm / tmp * jnp.power(egrid, tmp)
        return f[1:] - f[:-1]
