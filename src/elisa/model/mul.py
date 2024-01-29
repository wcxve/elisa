"""Models of multiplicative type."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import jax.numpy as jnp

from .base import Component, ParamConfig

__all__ = ["Constant"]


class MultiplicativeComponent(Component, ABC):
    """Prototype class to define multiplicative component."""

    @property
    def type(self) -> str:
        """Model type is multiplicative."""
        return "mul"

    @property
    def _func(self) -> Callable:
        """Return function that evaluate continnum over energy grid."""
        return self._continnum

    @staticmethod
    @abstractmethod
    def _continnum(*args):
        """Continnum to be evaluated over energy grids."""
        pass


class Constant(MultiplicativeComponent):
    _config = (ParamConfig("factor", "f", 1.0, 1e-5, 1e5, False, False),)

    @staticmethod
    def _continnum(egrid, factor):
        return jnp.full(egrid.size - 1, factor)
