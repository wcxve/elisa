"""Models of multiplicative type."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import jax.numpy as jnp

from .base import Component


class MultiplicativeComponent(Component, ABC):
    """Prototype class to define multiplicative component."""

    @property
    def type(self) -> str:
        """Model type is multiplicative."""
        return 'mul'

    @property
    def _func(self) -> Callable:
        """Return function that integrates continnum over energy grid."""
        return self._continum

    @staticmethod
    @abstractmethod
    def _continum(*args):
        """Continnum to be evaluated over energy grids."""
        pass


class Constant(MultiplicativeComponent):
    _default = (
        ('factor', 'A', 1.0, 1e-5, 1e5, False, False),
    )

    @staticmethod
    def _continum(egrid, factor):
        return jnp.full(egrid.size - 1, factor)
