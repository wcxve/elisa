"""Multiplicative models."""
from __future__ import annotations

from typing import Literal

import jax.numpy as jnp

from elisa.model.core.model import ComponentBase, ParamConfig
from elisa.util.typing import JAXArray, JAXFloat

__all__ = ['MultiplicativeComponent', 'Constant']


class MultiplicativeComponent(ComponentBase):
    """Prototype class to define multiplicative component."""

    def __init__(self, params: dict, latex: str | None):
        super().__init__(params, latex)

    @property
    def type(self) -> Literal['mul']:
        """Model type is multiplicative."""
        return 'mul'


class Constant(MultiplicativeComponent):
    """An energy-independent multiplicative factor.

    .. math::
        M(E) = f.

    Parameters
    ----------
    f : ParameterBase
        The multiplicative factor :math:`f`, dimensionless.

    """

    _config = (ParamConfig('f', 'f', '', 1.0, 1e-5, 1e5),)

    @staticmethod
    def _eval(egrid: JAXArray, params: dict[str, JAXFloat]) -> JAXArray:
        return jnp.full(egrid.size - 1, params['f'])
