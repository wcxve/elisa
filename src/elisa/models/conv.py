"""Convolution models."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from elisa.models.model import ConvolutionComponent, ParamConfig

if TYPE_CHECKING:
    from typing import Callable

    from elisa.util.typing import ConvolveEval, JAXArray, NameValMapping

__all__ = ['EnFlux', 'PhFlux', 'ZAShift', 'ZMShift', 'VAShift', 'VMShift']


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
        self._emin = float(emin)
        self.emax = emax

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

            self._prev_config = (self.emin, self.emax, self.ngrid, self.elog)
            self._convolve_jit = jax.jit(convolve, static_argnums=2)

            return self._convolve_jit

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

    .. math::
        N'(E) =
            \mathcal{F}_\mathrm{ph}
            \left[\int_{E_\mathrm{min}}^{E_\mathrm{max}} N(E) \, dE\right]^{-1}
            N(E)

    .. warning::
        The normalization of one of the additive components **must** be fixed
        to a positive value.

    .. warning::
        The flux is calculated by trapezoidal rule, and is accurate only if
        enough numbers of energy grids are used.

    Parameters
    ----------
    emin : float or int
        Minimum energy of the band to calculate the flux, in units of keV.
    emax : float or int
        Maximum energy of the band to calculate the flux, in units of keV.
    F : Parameter, optional
        Photon flux :math:`\mathcal{F}_\mathrm{ph}`, in units of cm⁻² s⁻¹.
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

    .. math::
        N'(E) =
            \mathcal{F}_\mathrm{en}
            \left[\int_{E_\mathrm{min}}^{E_\mathrm{max}} EN(E)\,dE\right]^{-1}
            N(E)

    .. warning::
        The normalization of one of the additive components **must** be fixed
        to a positive value.

    .. warning::
        The flux is calculated by trapezoidal rule, and is accurate only if
        enough numbers of energy grids are used.

    Parameters
    ----------
    emin : float or int
        Minimum energy of the band to calculate the flux, in units of keV.
    emax : float or int
        Maximum energy of the band to calculate the flux, in units of keV.
    F : Parameter, optional
        Energy flux :math:`\mathcal{F}_\mathrm{en}`, in units of erg cm⁻² s⁻¹.
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


class ZAShift(ConvolutionComponent):
    r"""Redshifts an additive model.

    Given flux function :math:`N(E)` [s⁻¹ cm⁻² keV⁻¹] of a source at redshift
    :math:`z`, the observed photon number :math:`n` between observed energy
    range :math:`e_1` [keV] and :math:`e_2` [keV] during exposure
    :math:`\Delta t` [s] is

    .. math::
        n &= \frac{\Delta t}{1+z} \int_{e_1}^{e_2} N(e (1+z)) \, \mathrm{d}e
          \\\\
          &= \frac{\Delta t}{(1+z)^2} \int_{E_1}^{E_2} N(E) \, \mathrm{d}E,

    where :math:`E_1 = e_1 (1+z)` [keV] and :math:`E_2 = e_2 (1+z)` [keV].

    Parameters
    ----------
    z : Parameter, optional
        Redshift :math:`z`, dimensionless.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    """

    _supported = frozenset({'add'})
    _config = (ParamConfig('z', 'z', '', 0.0, -0.999, 15.0, fixed=True),)

    @staticmethod
    def convolve(
        egrid: JAXArray,
        params: NameValMapping,
        model_fn: Callable[[JAXArray], JAXArray],
    ) -> JAXArray:
        factor = 1.0 + params['z']
        return model_fn(egrid * factor) / (factor * factor)


class ZMShift(ConvolutionComponent):
    r"""Redshifts a multiplicative model.

    Given model function :math:`M(E)` of a source at redshift :math:`z`, the
    average observed value between observed energy range :math:`e_1` [keV] and
    :math:`e_2` [keV] is

    .. math::
        m &= \frac{1}{e_2 - e_1} \int_{e_1}^{e_2} M(e (1+z)) \, \mathrm{d}e
          \\\\
          &= \frac{1}{E_2 - E_1} \int_{E_1}^{E_2} M(E) \, \mathrm{d}E,

    where :math:`E_1 = e_1 (1+z)` [keV] and :math:`E_2 = e_2 (1+z)` [keV].

    Parameters
    ----------
    z : Parameter, optional
        Redshift :math:`z`, dimensionless.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    """

    _supported = frozenset({'mul'})
    _config = (ParamConfig('z', 'z', '', 0.0, -0.999, 15.0, fixed=True),)

    @staticmethod
    def convolve(
        egrid: JAXArray,
        params: NameValMapping,
        model_fn: Callable[[JAXArray], JAXArray],
    ) -> JAXArray:
        factor = 1.0 + params['z']
        return model_fn(egrid * factor)


class VAShift(ConvolutionComponent):
    r"""Velocity shifts an additive model.

    Given flux function :math:`N(E)` [s⁻¹ cm⁻² keV⁻¹] of a source moving with
    speed :math:`v` along line of sight, the observed photon rate :math:`n`
    between observed energy range :math:`e_1` [keV] and :math:`e_2` [keV]
    during exposure :math:`\Delta t` [s] is

    .. math::
        n &= \frac{\Delta t}{\gamma} \int_{e_1}^{e_2} N(f e) \, \mathrm{d}e
          \\\\
          &= \frac{\Delta t}{\gamma f} \int_{E_1}^{E_2} N(E) \, \mathrm{d}E,

    where :math:`E_1 = f e_1` [keV], :math:`E_2 = f e_2` [keV],
    :math:`f = \sqrt{1-2\beta}`, :math:`\gamma = \sqrt{1-\beta^2}`, and
    :math:`\beta=v/c`.

    Parameters
    ----------
    v : Parameter, optional
        Velocity :math:`v`, in units of km s⁻¹.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    """

    _supported = frozenset({'add'})
    _config = (ParamConfig('v', 'v', 'km s^-1', 0.0, -1e4, 1e4, fixed=True),)

    @staticmethod
    def convolve(
        egrid: JAXArray,
        params: NameValMapping,
        model_fn: Callable[[JAXArray], JAXArray],
    ) -> JAXArray:
        v = params['v']  # unit: km/s
        c = 299792.458  # unit: km/s
        beta = v / c
        f = jnp.sqrt(1.0 - 2.0 * beta)
        gamma = jnp.sqrt(1.0 - beta * beta)
        return model_fn(egrid * f) / (f * gamma)


class VMShift(ConvolutionComponent):
    r"""Velocity shifts a multiplicative model.

    Given model function :math:`M(E)` of a source moving with speed :math:`v`
    along line of sight, the average observed value :math:`m` between observed
    energy range :math:`e_1` [keV] and :math:`e_2` [keV] is

    .. math::
        m &= \frac{1}{e_2 - e_1} \int_{e_1}^{e_2} M(f e) \, \mathrm{d}e
          \\\\
          &= \frac{1}{E_2 - E_1} \int_{E_1}^{E_2} M(E) \, \mathrm{d}E,

    where :math:`E_1 = f e_1` [keV], :math:`E_2 = f e_2` [keV],
    :math:`f = \sqrt{1-2\beta}`, and :math:`\beta=v/c`.

    Parameters
    ----------
    v : Parameter, optional
        Velocity :math:`v`, in units of km s⁻¹.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    """

    _supported = frozenset({'mul'})
    _config = (ParamConfig('v', 'v', 'km s^-1', 0.0, -1e4, 1e4, fixed=True),)

    @staticmethod
    def convolve(
        egrid: JAXArray,
        params: NameValMapping,
        model_fn: Callable[[JAXArray], JAXArray],
    ) -> JAXArray:
        v = params['v']  # unit: km/s
        c = 299792.458  # unit: km/s
        f = jnp.sqrt(1.0 - 2.0 * v / c)
        return model_fn(egrid * f)
