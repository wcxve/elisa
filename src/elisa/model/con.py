"""Models of convolution type."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import jax.numpy as jnp

from .base import Component

__all__ = []


class ConvolutionComponent(Component, ABC):
    """Prototype class to define multiplicative component."""

    @property
    def type(self) -> str:
        """Model type is convolution."""
        return "con"

    @property
    def _func(self) -> Callable:
        """Return photon flux function which has been convolved."""
        return self._convolve

    @staticmethod
    @abstractmethod
    def _convolve(*args):
        """Return photon flux which has been convolved."""
        pass
