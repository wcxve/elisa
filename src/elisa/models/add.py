"""Additive models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax.scipy import stats

from elisa.models.model import (
    AnaIntAdditive,
    NumIntAdditive,
    ParamConfig,
)

if TYPE_CHECKING:
    from elisa.util.typing import JAXArray, NameValMapping

__all__ = [
    'Band',
    'BandEp',
    'Blackbody',
    'BlackbodyRad',
    'Compt',
    'CutoffPL',
    'Gauss',
    'OTTB',
    'OTTS',
    'PowerLaw',
]


class Band(NumIntAdditive):
    r"""Gamma-ray burst continuum developed by Band et al. (1993) [1]_.

    .. math::
        N(E) = K
        \begin{cases}
        \bigl(\frac{E}{E_0}\bigr)^\alpha
            \exp\bigl(-\frac{E}{E_\mathrm{c}}\bigr),
            &\text{if } E < (\alpha-\beta) E_\mathrm{c},
        \\\\
        \left[\frac{(\alpha-\beta)E_\mathrm{c}}{E_0}\right]^{\alpha-\beta}
            \exp(\beta-\alpha) \bigl(\frac{E}{E_0}\bigr)^\beta,
            &\text{otherwise,}
        \end{cases}

    where :math:`E_0` is the pivot energy fixed at 100 keV.

    Parameters
    ----------
    alpha : Parameter, optional
        The low-energy power law index :math:`\alpha`, dimensionless.
    beta : Parameter, optional
        The high-energy power law index :math:`\beta`, dimensionless.
    Ec : Parameter, optional
        The characteristic energy :math:`E_\mathrm{c}`, in units of keV.
    K : Parameter, optional
        The amplitude :math:`K`, in units of cm⁻² s⁻¹ keV⁻¹.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    method : {'trapz', 'simpson'}, optional
        Numerical integration method. Defaults to 'trapz'.

    References
    ----------
    .. [1] `Band, D., et al. 1993, ApJ, 413, 281
           <https://adsabs.harvard.edu/full/1993ApJ...413..281B>`__
    """

    _config = (
        ParamConfig('alpha', r'\alpha', '', -1.0, -10.0, 5.0),
        ParamConfig('beta', r'\beta', '', -2.0, -10.0, 10.0),
        ParamConfig('Ec', r'E_\mathrm{c}', 'keV', 300.0, 10.0, 1e4),
        ParamConfig('K', 'K', 'cm^-2 s^-1 keV^-1', 1.0, 1e-10, 1e10),
    )

    @staticmethod
    def continnum(egrid: JAXArray, params: NameValMapping) -> JAXArray:
        alpha = params['alpha']
        beta = params['beta']
        Ec = params['Ec']
        K = params['K']

        e0 = 100.0

        # workaround for beta > alpha, as in XSPEC
        amb_ = alpha - beta
        inv_Ec = 1.0 / Ec
        amb = jnp.where(jnp.less(amb_, inv_Ec), inv_Ec, amb_)
        Ebreak = Ec * amb

        log = jnp.where(
            jnp.less(egrid, Ebreak),
            alpha * jnp.log(egrid / e0) - egrid / Ec,
            amb * jnp.log(amb * Ec / e0) - amb + beta * jnp.log(egrid / e0),
        )

        return K * jnp.exp(log)


class BandEp(NumIntAdditive):
    r"""Gamma-ray burst continuum developed by Band et al. (1993) [1]_,
    parametrized by the peak of :math:`\nu F_\nu`.

    .. math::
        N(E) = K
        \begin{cases}
        \bigl(\frac{E}{E_0}\bigr)^\alpha
            \exp\left[-\frac{(2+\alpha)E}{E_\mathrm{p}}\right],
            &\text{if } E < \frac{(\alpha-\beta)E_\mathrm{p}}{2+\alpha},
        \\\\
        \left[\frac{(\alpha-\beta)E_\mathrm{p}}{(2+\alpha)E_0}\right]
            ^{\alpha-\beta}
            \exp(\beta-\alpha)\bigl(\frac{E}{E_0}\bigr)^\beta,
            &\text{otherwise},
        \end{cases}

    where :math:`E_0` is the pivot energy fixed at 100 keV.

    Parameters
    ----------
    alpha : Parameter, optional
        The low-energy power law index :math:`\alpha`, dimensionless.
    beta : Parameter, optional
        The high-energy power law index :math:`\beta`, dimensionless.
    Ep : Parameter, optional
        The peak energy :math:`E_\mathrm{p}` of :math:`\nu F_\nu`,
        in units of keV.
    K : Parameter, optional
        The amplitude :math:`K`, in units of cm⁻² s⁻¹ keV⁻¹.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    method : {'trapz', 'simpson'}, optional
        Numerical integration method. Defaults to 'trapz'.

    References
    ----------
    .. [1] `Band, D., et al. 1993, ApJ, 413, 281
           <https://adsabs.harvard.edu/full/1993ApJ...413..281B>`__
    """

    _config = (
        ParamConfig('alpha', r'\alpha', '', -1.0, -10.0, 5.0),
        ParamConfig('beta', r'\beta', '', -2.0, -10.0, 10.0),
        ParamConfig('Ep', r'E_\mathrm{p}', 'keV', 300.0, 10.0, 1e4),
        ParamConfig('K', 'K', 'cm^-2 s^-1 keV^-1', 1.0, 1e-10, 1e10),
    )

    @staticmethod
    def continnum(egrid: JAXArray, params: NameValMapping) -> JAXArray:
        alpha = params['alpha']
        beta = params['beta']
        Ep = params['Ep']
        K = params['K']

        e0 = 100.0

        Ec = Ep / (2.0 + alpha)
        Ebreak = (alpha - beta) * Ec

        # workaround for beta > alpha, as in XSPEC
        amb_ = alpha - beta
        inv_Ec = 1.0 / Ec
        amb = jnp.where(jnp.less(amb_, inv_Ec), inv_Ec, amb_)

        log = jnp.where(
            jnp.less(egrid, Ebreak),
            alpha * jnp.log(egrid / e0) - egrid / Ec,
            amb * jnp.log(amb * Ec / e0) - amb + beta * jnp.log(egrid / e0),
        )

        return K * jnp.exp(log)


class Blackbody(NumIntAdditive):
    r"""Blackbody function.

    .. math::
        N(E) = \frac{C K E^2}{(kT)^4 [\exp(E/kT)-1]},

    where :math:`C=8.0525`.

    Parameters
    ----------
    kT : Parameter, optional
        The temperature :math:`kT`, in units of keV.
    K : Parameter, optional
        The amplitude :math:`K = L_{39}/D_{10}^2`, where :math:`L_{39}` is the
        source luminosity in units of 10³⁹ erg s⁻¹ and :math:`D_{10}` is the
        distance to the source in units of 10 kpc.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    method : {'trapz', 'simpson'}, optional
        Numerical integration method. Defaults to 'trapz'.
    """

    _config = (
        ParamConfig('kT', 'kT', 'keV', 3.0, 1e-4, 200.0),
        ParamConfig('K', 'K', '10^37 erg s^-1 kpc^-2', 1.0, 1e-10, 1e10),
    )

    @staticmethod
    def continnum(egrid: JAXArray, params: NameValMapping) -> JAXArray:
        kT = params['kT']
        K = params['K']
        x = egrid / kT
        tmp = 8.0525 * K * egrid / (kT * kT * kT)
        x_ = jnp.where(
            jnp.greater_equal(x, 50.0),
            1.0,  # avoid exponential overflow
            x,
        )

        return jnp.where(
            jnp.less_equal(x, 1e-4),
            tmp,
            jnp.where(
                jnp.greater_equal(x, 50.0),
                0.0,  # avoid exponential overflow
                tmp * x / jnp.expm1(x_),
            ),
        )
        # return 8.0525 * K * e*e / (kT*kT*kT*kT * jnp.expm1(e / kT))


class BlackbodyRad(NumIntAdditive):
    r"""Blackbody function with normalization proportional to the surface area.

    .. math::
        N(E) = \frac{C K E^2}{\exp(E/kT)-1},

    where :math:`C=1.0344 \times 10^{-3}` cm⁻² s⁻¹ keV⁻³.

    Parameters
    ----------
    kT : Parameter, optional
        The temperature :math:`kT`, in units of keV.
    K : Parameter, optional
        The amplitude :math:`K = R_\mathrm{km}^2/D_{10}^2`, where
        :math:`R_\mathrm{km}` is the source radius in km and :math:`D_{10}` is
        the distance to the source in units of 10 kpc.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    method : {'trapz', 'simpson'}, optional
        Numerical integration method. Defaults to 'trapz'.
    """

    _config = (
        ParamConfig('kT', 'kT', 'keV', 3.0, 1e-4, 200.0),
        ParamConfig('K', 'K', '', 1.0, 1e-10, 1e10),
    )

    @staticmethod
    def continnum(egrid: JAXArray, params: NameValMapping) -> JAXArray:
        kT = params['kT']
        K = params['K']

        x = egrid / kT
        tmp = 1.0344e-3 * K * egrid
        x_ = jnp.where(
            jnp.greater_equal(x, 50.0),
            1.0,  # avoid exponential overflow
            x,
        )

        return jnp.where(
            jnp.less_equal(x, 1e-4),
            tmp * kT,
            jnp.where(
                jnp.greater_equal(x, 50.0),
                0.0,  # avoid exponential overflow
                tmp * egrid / jnp.expm1(x_),
            ),
        )
        # return 1.0344e-3 * K * e*e / jnp.expm1(e / kT)


class BrokenPL(AnaIntAdditive):
    pass


class DoubleBrokenPL(AnaIntAdditive):
    pass


class SmoothlyBrokenPL(NumIntAdditive):
    pass


class DoubleSmoothlyBrokenPL(NumIntAdditive):
    pass


class CutoffPL(NumIntAdditive):
    r"""Power law with high-energy exponential cutoff.

    .. math::
        N(E) = K \left(\frac{E}{E_0}\right)^{-\alpha}
                \exp \left(-\frac{E}{E_\mathrm{c}}\right),

    where :math:`E_0` is the pivot energy fixed at 1 keV.

    Parameters
    ----------
    alpha : Parameter, optional
        The power law photon index :math:`\alpha`, dimensionless.
    Ec : Parameter, optional
        The e-folding energy of exponential cutoff :math:`E_\mathrm{c}`,
        in units of keV.
    K : Parameter, optional
        The amplitude :math:`K`, in units of cm⁻² s⁻¹ keV⁻¹.
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
    def continnum(egrid: JAXArray, params: NameValMapping) -> JAXArray:
        alpha = params['alpha']
        Ec = params['Ec']
        K = params['K']
        return K * jnp.power(egrid, -alpha) * jnp.exp(-egrid / Ec)


class Compt(NumIntAdditive):
    r"""Power law with high-energy exponential cutoff, parametrized by the peak
    of :math:`\nu F_\nu`.

    .. math::
        N(E) = K \left(\frac{E}{E_0}\right)^{-\alpha}
                \exp \left[-\frac{(2-\alpha)E}{E_\mathrm{p}}\right],

    where :math:`E_0` is the pivot energy fixed at 1 keV.

    Parameters
    ----------
    alpha : Parameter, optional
        The power law photon index :math:`\alpha`, dimensionless.
    Ep : Parameter, optional
        The peak energy :math:`E_\mathrm{p}` of :math:`\nu F_\nu`,
        in units of keV.
    K : Parameter, optional
        The amplitude :math:`K`, in units of cm⁻² s⁻¹ keV⁻¹.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    method : {'trapz', 'simpson'}, optional
        Numerical integration method. Defaults to 'trapz'.
    """

    _config = (
        ParamConfig('alpha', r'\alpha', '', 1.0, -3.0, 10.0),
        ParamConfig('Ep', r'E_\mathrm{p}', 'keV', 15.0, 0.01, 1e4),
        ParamConfig('K', 'K', 'cm^-2 s^-1 keV^-1', 1.0, 1e-10, 1e10),
    )

    @staticmethod
    def continnum(egrid: JAXArray, params: NameValMapping) -> JAXArray:
        alpha = params['alpha']
        Ep = params['Ep']
        K = params['K']
        neg_inv_Ec = (alpha - 2.0) / Ep
        return K * jnp.power(egrid, -alpha) * jnp.exp(egrid * neg_inv_Ec)


class Gauss(NumIntAdditive):
    r"""Gaussian line profile.

    .. math::
        N(E) = \frac{K}{\sqrt{2\pi} \sigma}
                \exp\left[
                    -\frac{\left(E - E_\mathrm{l}\right)^2}{2 \sigma^2}
                \right].

    Parameters
    ----------
    El : Parameter, optional
        The line energy :math:`E_\mathrm{l}`, in units of keV.
    sigma : Parameter, optional
        The line width :math:`\sigma`, in units of keV.
    K : Parameter, optional
        The total photon flux :math:`K` of the line, in units of cm⁻² s⁻¹.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    method : {'trapz', 'simpson'}, optional
        Numerical integration method. Defaults to 'trapz'.
    """

    _config = (
        ParamConfig('El', r'E_\mathrm{l}', 'keV', 6.5, 0.0, 1e6),
        ParamConfig('sigma', r'\sigma', 'keV', 0.1, 0.0, 20),
        ParamConfig('K', 'K', 'cm^-2 s^-1', 1.0, 1e-10, 1e10),
    )

    @staticmethod
    def continnum(egrid: JAXArray, params: NameValMapping) -> JAXArray:
        El = params['El']
        sigma = params['sigma']
        K = params['K']
        return K * stats.norm.pdf((egrid - El) / sigma)


class LogParabola(NumIntAdditive):
    pass


class Lorentz(NumIntAdditive):
    pass


class OTTB(NumIntAdditive):
    r"""Optically-thin thermal bremsstrahlung.

    .. math::
        N(E) = K \left(\frac{E}{E_0}\right)^{-1} \exp\left(-\frac{E}{kT}\right)
                \exp\left(\frac{E_0}{kT}\right),

    where :math:`E_0` is the pivot energy fixed at 1 keV.

    Parameters
    ----------
    kT : Parameter, optional
        The electron energy :math:`kT`, in units of keV.
    K : Parameter, optional
        The amplitude :math:`K`, in units of cm⁻² s⁻¹ keV⁻¹.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    method : {'trapz', 'simpson'}, optional
        Numerical integration method. Defaults to 'trapz'.
    """

    _config = (
        ParamConfig('kT', 'kT', 'keV', 30.0, 0.1, 1e3),
        ParamConfig('K', 'K', 'cm^-2 s^-1 keV^-1', 1.0, 1e-10, 1e10),
    )

    @staticmethod
    def continnum(egrid: JAXArray, params: NameValMapping) -> JAXArray:
        kT = params['kT']
        K = params['K']
        return K * jnp.exp((1.0 - egrid) / kT) / egrid


class OTTS(NumIntAdditive):
    r"""Optically-thin thermal synchrotron [1]_.

    .. math::
        N(E) = K \exp\left[-\left(\frac{E}{E_\mathrm{c}}\right)^{1/3}\right].

    Parameters
    ----------
    Ec : Parameter, optional
        The energy scale :math:`E_\mathrm{c}`, in units of keV.
    K : Parameter, optional
        The amplitude :math:`K`, in units of cm⁻² s⁻¹ keV⁻¹.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    method : {'trapz', 'simpson'}, optional
        Numerical integration method. Defaults to 'trapz'.

    References
    ----------
    .. [1] `Liang, E. P., et al., 1983, ApJ, 271, 776
           <https://adsabs.harvard.edu/full/1983ApJ...271..766L>`__
    """

    _config = (
        ParamConfig('Ec', r'E_\mathrm{c}', 'keV', 100.0, 1e-3, 1e3),
        ParamConfig('K', 'K', 'cm^-2 s^-1 keV^-1', 1.0, 1e-10, 1e10),
    )

    @staticmethod
    def continnum(egrid: JAXArray, params: NameValMapping) -> JAXArray:
        Ec = params['Ec']
        K = params['K']
        return K * jnp.exp(-jnp.power(egrid / Ec, 1.0 / 3.0))


class PowerLaw(AnaIntAdditive):
    r"""Power law function.

    .. math::
        N(E) = K \left(\frac{E}{E_0}\right)^{-\alpha},

    where :math:`E_0` is the pivot energy fixed at 1 keV.

    Parameters
    ----------
    alpha : Parameter, optional
        The power law photon index :math:`\alpha`, dimensionless.
    K : Parameter, optional
        The amplitude :math:`K`, in units of cm⁻² s⁻¹ keV⁻¹.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    """

    _config = (
        ParamConfig('alpha', r'\alpha', '', 1.01, -3.0, 10.0),
        ParamConfig('K', 'K', 'cm^-2 s^-1 keV^-1', 1.0, 1e-10, 1e10),
    )

    @staticmethod
    def integral(egrid: JAXArray, params: NameValMapping) -> JAXArray:
        # ignore the case of alpha = 1.0
        one_minus_alpha = 1.0 - params['alpha']
        f = params['K'] / one_minus_alpha * jnp.power(egrid, one_minus_alpha)
        return f[1:] - f[:-1]
