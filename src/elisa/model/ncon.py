"""Models of normalization convolution type."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import jax.numpy as jnp

from .base import Component, ParamConfig

__all__ = ['EnFlux', 'PhFlux']


class NormalizationConvolution(Component, ABC):
    """Calculate flux of an additive model between `emin` and `emax`.

    Parameters
    ----------
    emin : float or int
        Minimum energy, in unit keV.
    emax : float or int
        Maximum energy, in unit keV.
    F : Parameter, optional
        The flux parameter.
    ngrid : int or None, optional
        The energy grid number to create.
    elog : bool, optional
        Whether to use regular energy grid in log scale. The default is True.

    """

    _extra_kw = (
        ('emin',),
        ('emax',),
        ('ngrid', 1000),
        ('elog', True)
    )

    def __init__(
        self,
        emin: float | int,
        emax: float | int,
        ngrid: int = 1000,
        elog: bool = True,
        **kwargs
    ):
        if emin >= emax:
            raise ValueError('emin must be less than emax')

        self._emin = emin
        self._emax = emax
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
        value = float(value)
        if value >= self.emax:
            raise ValueError('emin must be less than emax')
        self._emin = value

    @property
    def emax(self) -> float:
        """Maximum value of photon energy grid"""
        return self._emax

    @emax.setter
    def emax(self, value: float | int):
        """Maximum value of photon energy grid"""
        value = float(value)
        if value <= self._emin:
            raise ValueError('emax must be larger than emin')
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
        """Whether to use regular energy grid in log scale."""
        return self._elog

    @elog.setter
    def elog(self, value: bool):
        """Whether to use regular energy grid in log scale."""
        self._elog = bool(value)

    @property
    def _func(self) -> Callable:
        if self.elog:
            egrid = jnp.geomspace(self.emin, self.emax, self.ngrid)
        else:
            egrid = jnp.linspace(self.emin, self.emax, self.ngrid)

        # TODO: a faster calculation, flux_input can be reused
        def fn(F, flux_input, flux_func, func_params):
            """The convolution function."""
            return self._convolve(F, flux_input, flux_func, func_params, egrid)

        return fn

    @staticmethod
    @abstractmethod
    def _convolve(egrid, F, flux_input, flux_func, func_params):
        pass


class PhFlux(NormalizationConvolution):
    _config = (
        ParamConfig(
            'F', r'\mathcal{F}_\mathrm{ph}', 1.0, 0.01, 1e10, False, False
        ),
    )

    @staticmethod
    def _convolve(F, flux_input, flux_func, func_params, egrid):
        mflux = jnp.sum(flux_func(func_params, egrid))
        return F / mflux * flux_input


class EnFlux(NormalizationConvolution):
    _config = (
        ParamConfig(
            'F', r'\mathcal{F}_\mathrm{en}', 1e-12, 1e-30, 1e30, False, True
        ),
    )

    @staticmethod
    def _convolve(F, flux_input, flux_func, func_params, egrid):
        mid = jnp.sqrt(egrid[:-1] * egrid[1:])
        mflux = jnp.sum(1.602176634e-9 * mid * flux_func(func_params, egrid))
        return F / mflux * flux_input
