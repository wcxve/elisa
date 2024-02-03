"""Additive models."""
from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Callable, Literal

import jax
import jax.numpy as jnp

from elisa.model.core.model import ComponentBase, ParamConfig
from elisa.util.typing import Array, JAXArray, JAXFloat

__all__ = ['AdditiveComponent', 'NumIntAdditive', 'PowerLaw', 'CutoffPL']


class AdditiveComponent(ComponentBase):
    """Prototype class to define additive component."""

    def __init__(self, params: dict, latex: str | None):
        super().__init__(params, latex)

    @property
    def type(self) -> Literal['add']:
        """Model type is additive."""
        return 'add'


class NumIntAdditive(AdditiveComponent):
    """Prototype additive component with numerical integration method."""

    _kwargs = ('method',)
    _continnum_jit: Callable[[JAXArray, dict[str, JAXFloat]], JAXArray] | None

    def __init__(
        self, params: dict, latex: str | None, method: str | None = None
    ):
        self.method = 'trapz' if method is None else method
        self._continnum_jit = None
        super().__init__(params, latex)

    def _eval(self, egrid: JAXArray, params: dict[str, JAXFloat]) -> JAXArray:
        if self._continnum_jit is None:
            f_jit = jax.jit(self._continnum)
            f_vmap = jax.vmap(f_jit, in_axes=(0, None))
            self._continnum_jit = jax.jit(f_vmap)

        if self.method == 'trapz':
            de = egrid[1:] - egrid[:-1]
            f_grid = self._continnum_jit(egrid, params)
            return 0.5 * (f_grid[:-1] + f_grid[1:]) * de

        elif self.method == 'simpson':
            de = egrid[1:] - egrid[:-1]
            e_mid = (egrid[:-1] + egrid[1:]) / 2.0
            f_grid = self._continnum_jit(egrid, params)
            f_mid = self._continnum_jit(e_mid, params)
            return de / 6.0 * (f_grid[:-1] + 4.0 * f_mid + f_grid[1:])

        else:
            raise NotImplementedError(f"integration method '{self.method}'")

    if TYPE_CHECKING:

        @abstractmethod
        def _continnum(self, *args, **kwargs) -> JAXFloat:
            pass

    else:

        @abstractmethod
        def _continnum(
            self, egrid: JAXFloat, params: dict[str, JAXFloat]
        ) -> JAXFloat:
            pass

    @property
    def method(self) -> str:
        """Numerical integration method."""
        return self._method

    @method.setter
    def method(self, method: str):
        method = str(method)

        if method not in ('trapz', 'simpson'):
            raise ValueError(
                f"available integration methods are 'trapz' and 'simpson', "
                f"but got '{method}'"
            )

        self._method = method


class PowerLaw(AdditiveComponent):
    r"""The power law function.

    .. math::
        N_E = K \left(\frac{E}{E_0}\right)^{-\alpha},

    where :math:`E_0` is the pivot energy fixed at 1 keV.

    Parameters
    ----------
    alpha : ParameterBase
        The power law photon index :math:`alpha`, dimensionless.
    K : ParameterBase
        The amplitude :math:`K`, in units of :math:`\mathrm{cm}^{-2} \,
        \mathrm{s}^{-1} \, \mathrm{keV}^{-1}`.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.

    """

    _config = (
        ParamConfig('alpha', r'\alpha', '', 1.01, -3.0, 10.0),
        ParamConfig('K', 'K', 'cm^-2 s^-1 keV^-1', 1.0, 1e-10, 1e10),
    )

    @staticmethod
    def _eval(egrid: Array, params: dict[str, float | JAXFloat]) -> JAXArray:
        # ignore the case of alpha = 1.0
        one_minus_alpha = 1.0 - params['alpha']
        f = params['K'] / one_minus_alpha * jnp.power(egrid, one_minus_alpha)
        return f[1:] - f[:-1]


class CutoffPL(NumIntAdditive):
    r"""The power law with high-energy exponential cutoff.

    .. math::
        N(E) = K \left(\frac{E}{E_0}\right)^{-\alpha}
                \exp \left(-\frac{E}{E_\mathrm{c}}\right),

    where :math:`E_0` is the pivot energy fixed at 1 keV.

    Parameters
    ----------
    alpha : ParameterBase, optional
        The power law photon index :math:`\alpha`, dimensionless.
    Ec : ParameterBase, optional
        The :math:`e`-folding energy of exponential cutoff
        :math:`E_\mathrm{c}`, in units of keV.
    K : ParameterBase, optional
        The amplitude :math:`K`, in units of :math:`\mathrm{cm}^{-2} \,
        \mathrm{s}^{-1} \, \mathrm{keV}^{-1}`.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    method : {'trapz', 'simpson'}, optional
        Numerical integration method. Defaults to 'trapz'.

    """

    _config = (
        ParamConfig('alpha', r'\alpha', '', 1.0, -3.0, 10.0),
        ParamConfig('Ec', r'E_\mathrm{c}', 'keV', 15.0, 0.01, 1e4),
        ParamConfig('K', 'K', 'cm^-2 s^-1 keV^-1', 1.0, 1e-10, 1e10),
    )

    @staticmethod
    def _continnum(e: JAXFloat, params: dict[str, JAXFloat]) -> JAXFloat:
        alpha = params['alpha']
        Ec = params['Ec']
        K = params['K']
        return K * jnp.power(e, -alpha) * jnp.exp(-e / Ec)
