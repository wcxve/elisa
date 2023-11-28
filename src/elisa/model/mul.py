"""Models of multiplicative type."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from .base import Component


class MultiplicativeComponent(Component, ABC):
    """Prototype class to define multiplicative component."""

    @property
    def type(self) -> str:
        """Model type is multiplicative."""
        return 'mul'

    def _func_generator(self) -> Callable:
        """Return function that integrates continnum over energy grid."""
        return self._continum(func_name)


class Constant(Component):
    _default = (
        ('factor', 'A', 1.0, 1e-5, 1e5, False, False),
    )

    @staticmethod
    def _integral(egrid, PhoIndex, norm):
        # here we ignore the case of PhoIndex == 1.0
        tmp = 1.0 - PhoIndex
        f = norm / tmp * jnp.power(egrid, tmp)
        return f[1:] - f[:-1]