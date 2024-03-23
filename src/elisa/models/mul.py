"""Multiplicative models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from elisa.models.model import (
    AnaIntMultiplicative,
    NumIntMultiplicative,
    ParamConfig,
)

if TYPE_CHECKING:
    from elisa.util.typing import JAXArray, NameValMapping

__all__ = ['Constant', 'Edge', 'ExpAbs', 'ExpFac', 'GAbs', 'HighECut', 'PLAbs']


class Constant(AnaIntMultiplicative):
    r"""Energy-independent multiplicative factor.

    .. math::
        M(E) = f.

    Parameters
    ----------
    f : Parameter, optional
        The multiplicative factor :math:`f`, dimensionless.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    """

    _config = (ParamConfig('f', 'f', '', 1.0, 1e-5, 1e5),)

    @staticmethod
    def integral(egrid: JAXArray, params: NameValMapping) -> JAXArray:
        return jnp.full(egrid.size - 1, params['f'])


class Edge(NumIntMultiplicative):
    r"""Absorption edge.

    .. math::
        M(E) =
        \begin{cases}
        \exp\left[-D \bigl(\frac{E}{E_\mathrm{c}}\bigr)^3\right],
            &\text{if } E \ge E_\mathrm{c},
        \\\\
        1, &\text{otherwise.}
        \end{cases}

    Parameters
    ----------
    Ec : Parameter, optional
        The threshold energy :math:`E_\mathrm{c}`, in units of keV.
    D : Parameter, optional
        The absorption depth :math:`D` at the threshold energy, dimensionless.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    method : {'trapz', 'simpson'}, optional
        Numerical integration method. Defaults to 'trapz'.
    """

    _config = (
        ParamConfig('Ec', r'E_\mathrm{c}', 'keV', 7.0, 0.0, 100.0),
        ParamConfig('D', 'D', '', 1.0, 0.0, 10),
    )

    @staticmethod
    def continnum(egrid: JAXArray, params: NameValMapping) -> JAXArray:
        Ec = params['Ec']
        D = params['D']
        return jnp.where(
            egrid >= Ec, jnp.exp(-D * jnp.power(egrid / Ec, 3.0)), 1.0
        )


class ExpAbs(NumIntMultiplicative):
    r"""Low-energy exponential rolloff.

    .. math::
        M(E) = \exp\left(-\frac{E_\mathrm{c}}{E}\right).

    Parameters
    ----------
    Ec : Parameter, optional
        The e-folding energy :math:`E_\mathrm{c}` for the absorption,
        in units of keV.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    method : {'trapz', 'simpson'}, optional
        Numerical integration method. Defaults to 'trapz'.
    """

    _config = (ParamConfig('Ec', r'E_\mathrm{c}', 'keV', 2.0, 0.0, 200.0),)

    @staticmethod
    def continnum(egrid: JAXArray, params: NameValMapping) -> JAXArray:
        return jnp.exp(-params['Ec'] / egrid)


class ExpFac(NumIntMultiplicative):
    r"""Exponential modification.

    .. math::
        M(E) =
        \begin{cases}
        1 + A \exp\bigl(-\frac{f E}{E_0}\bigr), &\text{if } E \ge E_\mathrm{c},
        \\\\
        1, &\text{otherwise,}
        \end{cases}

    where :math:`E_0` is the pivot energy fixed at 1 keV.

    Parameters
    ----------
    A : Parameter, optional
        The amplitude of effect :math:`A`, dimensionless.
    f : Parameter, optional
        The exponential factor :math:`f`, dimensionless.
    Ec : Parameter, optional
        The start energy of modification :math:`E_\mathrm{c}`, in units of keV.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    method : {'trapz', 'simpson'}, optional
        Numerical integration method. Defaults to 'trapz'.
    """

    _config = (
        ParamConfig('A', 'A', '', 1.0, 0.0, 1e6),
        ParamConfig('f', 'f', '', 1.0, 0.0, 1e6),
        ParamConfig('Ec', r'E_\mathrm{c}', 'keV', 0.5, 0.0, 1e6),
    )

    @staticmethod
    def continnum(egrid: JAXArray, params: NameValMapping) -> JAXArray:
        A = params['A']
        f = params['f']
        Ec = params['Ec']
        return jnp.where(egrid >= Ec, 1.0 + A * jnp.exp(-f * egrid), 1.0)


class GAbs(NumIntMultiplicative):
    r"""Gaussian absorption line.

    .. math::
        M(E) = \exp\left[
                    -\frac{\tau}{\sqrt{2\pi} \sigma}
                    \exp\left[
                        -\frac{\left(E - E_\mathrm{l}\right)^2}{2 \sigma^2}
                    \right]
                \right].

    The optical depth at line center is :math:`\frac{\tau}{\sqrt{2\pi}\sigma}`.

    Parameters
    ----------
    El : Parameter, optional
        The line energy :math:`E_\mathrm{l}`, in units of keV.
    sigma : Parameter, optional
        The line width :math:`\sigma`, in units of keV.
    tau : Parameter, optional
        The line depth :math:`\tau`, in units of keV.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    method : {'trapz', 'simpson'}, optional
        Numerical integration method. Defaults to 'trapz'.
    """

    _config = (
        ParamConfig('El', r'E_\mathrm{l}', 'keV', 1, 0.0, 1e6),
        ParamConfig('sigma', r'\sigma', 'keV', 0.01, 0.0, 20),
        ParamConfig('tau', r'\tau', '', 1.0, 0.0, 1e6),
    )

    @staticmethod
    def continnum(egrid: JAXArray, params: NameValMapping) -> JAXArray:
        El = params['El']
        sigma = params['sigma']
        tau = params['tau']
        return jnp.exp(
            -tau
            / (jnp.sqrt(2 * jnp.pi) * sigma)
            * jnp.exp(-0.5 * jnp.power((egrid - El) / sigma, 2))
        )


class HighECut(NumIntMultiplicative):
    r"""High-energy cutoff.

    .. math::
        M(E) =
        \begin{cases}
        \exp\bigl(\frac{E_\mathrm{c}-E}{E_\mathrm{f}}\bigr),
            &\text{if } E \ge E_\mathrm{c},
        \\\\
        1, &\text{otherwise.}
        \end{cases}

    Parameters
    ----------
    Ec : Parameter, optional
        The cutoff energy :math:`E_\mathrm{c}`, in units of keV.
    Ef : Parameter, optional
        The e-folding energy :math:`E_\mathrm{f}`, in units of keV.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    method : {'trapz', 'simpson'}, optional
        Numerical integration method. Defaults to 'trapz'.
    """

    _config = (
        ParamConfig('Ec', r'E_\mathrm{c}', 'keV', 10.0, 1e-4, 1e6),
        ParamConfig('Ef', r'E_\mathrm{f}', 'keV', 15.0, 1e-4, 1e6),
    )

    @staticmethod
    def continnum(egrid: JAXArray, params: NameValMapping) -> JAXArray:
        Ec = params['Ec']
        Ef = params['Ef']
        return jnp.where(egrid >= Ec, jnp.exp((Ec - egrid) / Ef), 1.0)


class PLAbs(AnaIntMultiplicative):
    r"""Absorption as a power-law in energy. Useful for things like dust.

    .. math::
        M(E) = K \left(\frac{E}{E_0}\right)^{-\alpha},

    where :math:`E_0` is the pivot energy fixed at 1 keV.

    Parameters
    ----------
    alpha : Parameter, optional
        The power law index :math:`\alpha`, dimensionless.
    K : Parameter, optional
        The coefficient :math:`K`, dimensionless.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    """

    _config = (
        ParamConfig('alpha', r'\alpha', '', 2.0, 0.0, 5.0),
        ParamConfig('K', 'K', '', 1.0, 0.0, 100.0),
    )

    @staticmethod
    def integral(egrid: JAXArray, params: NameValMapping) -> JAXArray:
        # ignore the case of alpha = 1.0
        one_minus_alpha = 1.0 - params['alpha']
        f = params['K'] / one_minus_alpha * jnp.power(egrid, one_minus_alpha)
        return (f[1:] - f[:-1]) / (egrid[1:] - egrid[:-1])


class Absorption(NumIntMultiplicative):
    pass


class WAbs(Absorption):
    pass


class TBAbs(Absorption):
    pass


class PhAbs(Absorption):
    pass
