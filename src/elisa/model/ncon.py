"""Models of normalization convolution type."""
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import Callable

import jax.numpy as jnp

from .base import Component

__all__ = ['EnFlux', 'PhFlux']


class NormalizationConvolution(Component, ABC):
    """Prototype class to define convolution model of normalization type."""

    _extra_kw = (
        ('emin', None),
        ('emax', None),
        ('ngrid', 1000),
        ('log', True)
    )

    def __init__(
        self,
        emin: float | int,
        emax: float | int,
        ngrid: int = 1000,
        elog: bool = True,
        **kwargs
    ):
        self.emin = emin
        self.emax = emax
        self.ngrid = int(ngrid)
        self.elog = bool(elog)
        super().__init__(**kwargs)

    @property
    def emin(self) -> float:
        """Minimum value of photon energy grid"""
        return self._emin

    @emin.setter
    def emin(self, value: float | int):
        """Minimum value of photon energy grid"""
        self._emin = float(value)

    @property
    def emax(self) -> float:
        """Maximum value of photon energy grid"""
        return self._emax

    @emax.setter
    def emax(self, value: float | int):
        """Maximum value of photon energy grid"""
        self._emax = float(value)

    @property
    def type(self) -> str:
        """Model type is normalization convolution."""
        return 'ncon'

    @property
    def ngrid(self) -> int:
        """Photon energy grid number."""
        return self._ngrid

    @ngrid.setter
    def ngrid(self, value: int):
        """Photon energy grid number."""
        self._ngrid = int(value)

    @property
    def elog(self) -> bool:
        """Whether to use evenly-spaced energies in log space."""
        return self._elog

    @elog.setter
    def elog(self, value: bool):
        """Whether to use evenly-spaced energies in log space."""
        self._elog = bool(value)

    @property
    def _func(self) -> Callable:
        if self.emin >= self.emax:
            raise ValueError('emin must be less than emax')

        if self.elog:
            egrid = jnp.geomspace(self.emin, self.emax, self.ngrid)
            emid = jnp.sqrt(egrid[:-1] * egrid[1:])
        else:
            egrid = jnp.linspace(self.emin, self.emax, self.ngrid)
            emid = (egrid[:-1] + egrid[1:]) / 2.0

        return partial(self._convolve, egrid=egrid, emid=emid)

    @staticmethod
    @abstractmethod
    def _convolve(flux, flux_input, flux_func, func_params, egrid, emid):
        pass


class PhFlux(NormalizationConvolution):
    _default = (
        ('flux', r'\mathcal{F}_{\rm ph}', 1, 0.01, 1e10, False, False),
    )

    @staticmethod
    def _convolve(flux, flux_input, flux_func, func_params, egrid, emid):
        mflux = jnp.sum(flux_func(func_params, egrid))
        return flux / mflux * flux_input


class EnFlux(NormalizationConvolution):
    _default = (
        ('eflux', r'\mathcal{F}_{\rm en}', 1e-12, 1e-30, 1e30, False, True),
    )

    @staticmethod
    def _convolve(eflux, flux_input, flux_func, func_params, egrid, emid):
        mflux = jnp.sum(1.6022e-9 * emid * flux_func(func_params, egrid))
        return eflux / mflux * flux_input
