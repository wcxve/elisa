"""Convolution models."""
from __future__ import annotations

from abc import abstractmethod
from typing import Callable

import jax
import jax.numpy as jnp

from elisa.model.model import ConvolutionComponent, ParamConfig
from elisa.util.typing import ConvolveEval, JAXArray, NameValMapping

__all__ = ['EnFlux', 'PhFlux', 'RedShift', 'VelocityShift']


class NormConvolution(ConvolutionComponent):
    _args = ('emin', 'emax')
    _kwargs = ('ngrid', 'elog')
    _supported = frozenset({'add'})

    def __init__(
        self,
        emin: float | int,
        emax: float | int,
        params: dict,
        latex: str | None,
        ngrid: int | None,
        elog: bool | None,
    ):
        if emin >= emax:
            raise ValueError('emin must be less than emax')

        self._emin = float(emin)
        self._emax = float(emax)

        self.ngrid = 1000 if ngrid is None else ngrid
        self.elog = True if elog is None else bool(elog)

        self._prev_config: tuple | None = None

        super().__init__(params, latex)

    @staticmethod
    @abstractmethod
    def convolve(
        egrid: JAXArray,
        params: NameValMapping,
        model_fn: Callable[[JAXArray], JAXArray],
        flux_egrid: JAXArray,
    ) -> JAXArray:
        """Convolve a model function.

        Parameters
        ----------
        egrid : ndarray
            Photon energy grid in units of keV.
        params : dict
            Parameter dict for the convolution model.
        model_fn : callable
            The model function to be convolved, which takes the energy grid as
            input and returns the model flux over the grid.
        flux_egrid : ndarray
            Photon energy grid used to calculate flux in units of keV.

        Returns
        -------
        value : ndarray
            The re-normalized model over `egrid`, in units of cm⁻² s⁻¹ keV⁻¹.

        """
        pass

    @property
    def eval(self) -> ConvolveEval:
        if self._prev_config == (self.emin, self.emax, self.ngrid, self.elog):
            return self._convolve_jit

        else:
            if self.elog:
                flux_egrid = jnp.geomspace(self.emin, self.emax, self.ngrid)
            else:
                flux_egrid = jnp.linspace(self.emin, self.emax, self.ngrid)

            fn = self.convolve

            def convolve(
                egrid: JAXArray,
                params: NameValMapping,
                model_fn: Callable[[JAXArray], JAXArray],
            ) -> JAXArray:
                # TODO: egrid can be reused to reduce computation
                return fn(egrid, params, model_fn, flux_egrid)

            self._convolve_jit = jax.jit(convolve, static_argnums=2)

        return convolve

    @property
    def emin(self) -> float:
        """Minimum value of photon energy grid"""
        return self._emin

    @emin.setter
    def emin(self, value: float | int):
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
        value = float(value)
        if value <= self._emin:
            raise ValueError('emax must be larger than emin')
        self._emax = float(value)

    @property
    def ngrid(self) -> int:
        """Photon energy grid number."""
        return self._ngrid

    @ngrid.setter
    def ngrid(self, value: int):
        self._ngrid = int(value)

    @property
    def elog(self) -> bool:
        """Whether to use logarithmically regular energy grids."""
        return self._elog

    @elog.setter
    def elog(self, value: bool):
        self._elog = bool(value)


class PhFlux(NormConvolution):
    r"""Normalize an additive model by photon flux between `emin` and `emax`.

    Warnings
    --------
    The normalization of one of the additive components **must** be fixed to a
    positive value.

    Parameters
    ----------
    emin : float or int
        Minimum energy, in units of keV.
    emax : float or int
        Maximum energy, in units of keV.
    F : ParameterBase, optional
        Flux parameter, in units of cm⁻² s⁻¹.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    ngrid : int, optional
        The energy grid number to use. The default is 1000.
    elog : bool, optional
        Whether to use logarithmically regular energy grids.
        The default is True.

    """

    _config = (
        ParamConfig(
            'F', r'\mathcal{F}_\mathrm{ph}', 'cm^-2 s^-1', 1.0, 0.01, 1e10
        ),
    )

    @staticmethod
    def convolve(
        egrid: JAXArray,
        params: NameValMapping,
        model_fn: Callable[[JAXArray], JAXArray],
        flux_egrid: JAXArray,
    ) -> JAXArray:
        F = params['F']
        mflux = jnp.sum(model_fn(flux_egrid))
        flux = model_fn(egrid)
        return F / mflux * flux


class EnFlux(NormConvolution):
    r"""Normalize an additive model by energy flux between `emin` and `emax`.

    Warnings
    --------
    The normalization of one of the additive components **must** be fixed to a
    positive value.

    Parameters
    ----------
    emin : float or int
        Minimum energy, in units of keV.
    emax : float or int
        Maximum energy, in units of keV.
    F : ParameterBase, optional
        Flux parameter, in units of erg cm⁻² s⁻¹.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    ngrid : int, optional
        The energy grid number to use. The default is 1000.
    elog : bool, optional
        Whether to use logarithmically regular energy grids.
        The default is True.

    """

    _config = (
        ParamConfig(
            'F',
            r'\mathcal{F}_\mathrm{en}',
            'erg cm^-2 s^-1',
            1e-12,
            1e-30,
            1e30,
            log=True,
        ),
    )

    @staticmethod
    def convolve(
        egrid: JAXArray,
        params: NameValMapping,
        model_fn: Callable[[JAXArray], JAXArray],
        flux_egrid: JAXArray,
    ) -> JAXArray:
        F = params['F']
        keV_to_erg = 1.602176634e-9
        mid = jnp.sqrt(flux_egrid[:-1] * flux_egrid[1:])
        _flux = model_fn(flux_egrid)
        mflux = jnp.sum(keV_to_erg * mid * _flux)
        flux = model_fn(egrid)
        return F / mflux * flux


class RedShift(ConvolutionComponent):
    r"""Redshifts a model.

    It shifts energies by 1 / (1 + :math:`z`) and then calculates the model.

    Parameters
    ----------
    z : ParameterBase, optional
        Redshift :math:`z`, dimensionless.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.

    """

    _config = (ParamConfig('z', 'z', '', 0.0, -0.999, 15.0, fixed=True),)

    @staticmethod
    def convolve(
        egrid: JAXArray,
        params: NameValMapping,
        model_fn: Callable[[JAXArray], JAXArray],
    ) -> JAXArray:
        factor = 1.0 + params['z']
        return model_fn(egrid * factor)


class VelocityShift(ConvolutionComponent):
    r"""Velocity shifts a model.

    It shifts energies :math:`-Ev/c` and then calculates the model.

    Parameters
    ----------
    v : ParameterBase, optional
        Velocity :math:`v`, in units of km s⁻¹.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.

    """

    _config = (ParamConfig('v', 'v', 'km s^-1', 0.0, -1e4, 1e4, fixed=True),)

    @staticmethod
    def convolve(
        egrid: JAXArray,
        params: NameValMapping,
        model_fn: Callable[[JAXArray], JAXArray],
    ) -> JAXArray:
        v = params['v']  # unit: km/s
        c = 299792.458  # unit: km/s
        factor = 1.0 - v / c
        return model_fn(egrid * factor)
