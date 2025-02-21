"""Additive models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.scipy import stats

from elisa.models.model import (
    AnaIntAdditive,
    NumIntAdditive,
    ParamConfig,
)

if TYPE_CHECKING:
    from elisa.util.typing import CompEval, JAXArray, NameValMapping

__all__ = [
    'Band',
    'BandEp',
    'BentPL',
    'Blackbody',
    'BlackbodyRad',
    'BrokenPL',
    'Compt',
    'CutoffPL',
    'Gauss',
    'Lorentz',
    'OTTB',
    'OTTS',
    'PLEnFlux',
    'PLPhFlux',
    'PowerLaw',
    'SmoothlyBrokenPL',
]


def _powerlaw_integral(egrid: JAXArray, alpha: JAXArray) -> JAXArray:
    cond = jnp.full(len(egrid), jnp.not_equal(alpha, 1.0))

    one_minus_alpha = jnp.where(cond, 1.0 - alpha, 1.0)
    f1 = jnp.power(egrid, one_minus_alpha) / one_minus_alpha
    f1 = f1[1:] - f1[:-1]

    f2 = jnp.log(egrid)
    f2 = f2[1:] - f2[:-1]

    return jnp.where(cond[:-1], f1, f2)


class Band(NumIntAdditive):
    r"""Gamma-ray burst continuum developed by Band et al. (1993) [1]_.

    .. math::
        N(E) = K
        \begin{cases}
        \bigl(\frac{E}{E_0}\bigr)^\alpha
            \exp\bigl(-\frac{E}{E_\mathrm{c}}\bigr),
            &\text{if } E < (\alpha-\beta) E_\mathrm{c},
        \\\\
        \bigl(\frac{E}{E_0}\bigr)^\beta \exp(\beta-\alpha)
            \left[
                \frac{(\alpha-\beta)E_\mathrm{c}}{E_0}
            \right]^{\alpha-\beta},
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
        The amplitude :math:`K`, in units of ph cm⁻² s⁻¹ keV⁻¹.
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
        ParamConfig('K', 'K', 'ph cm^-2 s^-1 keV^-1', 1.0, 1e-10, 1e10),
    )

    @staticmethod
    def continuum(egrid: JAXArray, params: NameValMapping) -> JAXArray:
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
        \bigl(\frac{E}{E_0}\bigr)^\beta \exp(\beta-\alpha)
            \left[
                \frac{(\alpha-\beta)E_\mathrm{p}}{(2+\alpha)E_0}
            \right]^{\alpha-\beta},
            &\text{otherwise},
        \end{cases}

    where :math:`E_0` is the pivot energy fixed at 100 keV.

    .. warning::
        This model requires the low-energy power law index :math:`\alpha`
        to be greater than -2.0.

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
        The amplitude :math:`K`, in units of ph cm⁻² s⁻¹ keV⁻¹.
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
        ParamConfig('alpha', r'\alpha', '', -1.0, -1.99, 10.0),
        ParamConfig('beta', r'\beta', '', -2.0, -10.0, 10.0),
        ParamConfig('Ep', r'E_\mathrm{p}', 'keV', 300.0, 10.0, 1e4),
        ParamConfig('K', 'K', 'ph cm^-2 s^-1 keV^-1', 1.0, 1e-10, 1e10),
    )

    @staticmethod
    def continuum(egrid: JAXArray, params: NameValMapping) -> JAXArray:
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


class BentPL(NumIntAdditive):
    r"""Bent power law.

    .. math::
        N(E) = 2K \left[
            1 + \left(\frac{E}{E_\mathrm{b}}\right)^\alpha
        \right]^{-1}

    Parameters
    ----------
    alpha : Parameter, optional
        The power law photon index :math:`\alpha`, dimensionless.
    Eb : Parameter, optional
        The break energy :math:`E_\mathrm{b}`, in units of keV.
    K : Parameter, optional
        The amplitude :math:`K`, in units of ph cm⁻² s⁻¹ keV⁻¹.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    method : {'trapz', 'simpson'}, optional
        Numerical integration method. Defaults to 'trapz'.
    """

    _config = (
        ParamConfig('alpha', 'alpha', '', 1.5, -3.0, 10.0),
        ParamConfig('Eb', r'E_\mathrm{b}', 'keV', 5.0, 1e-2, 1e6),
        ParamConfig('K', 'K', 'ph cm^-2 s^-1 keV^-1', 1.0, 1e-10, 1e10),
    )

    @staticmethod
    def continuum(egrid: JAXArray, params: NameValMapping) -> JAXArray:
        alpha = params['alpha']
        Eb = params['Eb']
        K = params['K']
        return K * 2.0 / (1.0 + jnp.power(egrid / Eb, alpha))


class Blackbody(NumIntAdditive):
    r"""Blackbody function.

    .. math::
        N(E) = \frac{C K E^2}{(kT)^4 [\exp(E/kT)-1]},

    where :math:`C=8.0525` ph.

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
    def continuum(egrid: JAXArray, params: NameValMapping) -> JAXArray:
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

    where :math:`C=1.0344 \times 10^{-3}` ph cm⁻² s⁻¹ keV⁻³.

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
    def continuum(egrid: JAXArray, params: NameValMapping) -> JAXArray:
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
    r"""Broken power law.

    .. math::
        N(E) = K
        \begin{cases}
        \left(\frac{E}{E_\mathrm{b}}\right)^{-\alpha_1},
            &\text{if } E \le E_\mathrm{b},
        \\\\
        \left(\frac{E}{E_\mathrm{b}}\right)^{-\alpha_2},
            &\text{otherwise}.
        \end{cases}

    Parameters
    ----------
    alpha1 : Parameter, optional
        The low-energy power law photon index :math:`\alpha_1`, dimensionless.
    alpha2 : Parameter, optional
        The high-energy power law photon index :math:`\alpha_2`, dimensionless.
    Eb : Parameter, optional
        The break energy :math:`E_\mathrm{b}`, in units of keV.
    K : Parameter, optional
        The amplitude :math:`K`, in units of ph cm⁻² s⁻¹ keV⁻¹.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    method : {'trapz', 'simpson'}, optional
        Numerical integration method. Defaults to 'trapz'.
    """

    _config = (
        ParamConfig('alpha1', 'alpha_1', '', 1.0, -3.0, 10.0),
        ParamConfig('Eb', r'E_\mathrm{b}', 'keV', 5.0, 1e-2, 1e6),
        ParamConfig('alpha2', 'alpha_2', '', 2.0, -3.0, 10.0),
        ParamConfig('K', 'K', 'ph cm^-2 s^-1 keV^-1', 1.0, 1e-10, 1e10),
    )

    @staticmethod
    def integral(egrid: JAXArray, params: NameValMapping) -> JAXArray:
        alpha1 = params['alpha1']
        Eb = params['Eb']
        alpha2 = params['alpha2']
        K = params['K']
        mask = egrid[:-1] <= Eb
        egrid_ = egrid / Eb
        p1 = _powerlaw_integral(egrid_, alpha1)
        p2 = _powerlaw_integral(egrid_, alpha2)
        f = jnp.where(mask, p1, p2)
        idx = jnp.flatnonzero(mask, size=egrid.size - 1)[-1]
        pb1 = _powerlaw_integral(jnp.hstack([egrid_[idx], Eb]), alpha1)
        pb2 = _powerlaw_integral(jnp.hstack([Eb, egrid_[idx + 1]]), alpha2)
        f.at[idx].set(pb1[0] + pb2[0])
        return K * f


class DoubleBrokenPL(AnaIntAdditive):
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
        The amplitude :math:`K`, in units of ph cm⁻² s⁻¹ keV⁻¹.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    method : {'trapz', 'simpson'}, optional
        Numerical integration method. Defaults to 'trapz'.
    """

    _config = (
        ParamConfig('alpha', r'\alpha', '', 1.0, -3.0, 10.0),
        ParamConfig('Ec', r'E_\mathrm{c}', 'keV', 15.0, 0.01, 1e4),
        ParamConfig('K', 'K', 'ph cm^-2 s^-1 keV^-1', 1.0, 1e-10, 1e10),
    )

    @staticmethod
    def continuum(egrid: JAXArray, params: NameValMapping) -> JAXArray:
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
        The amplitude :math:`K`, in units of ph cm⁻² s⁻¹ keV⁻¹.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    method : {'trapz', 'simpson'}, optional
        Numerical integration method. Defaults to 'trapz'.
    """

    _config = (
        ParamConfig('alpha', r'\alpha', '', 1.0, -3.0, 10.0),
        ParamConfig('Ep', r'E_\mathrm{p}', 'keV', 15.0, 0.01, 1e4),
        ParamConfig('K', 'K', 'ph cm^-2 s^-1 keV^-1', 1.0, 1e-10, 1e10),
    )

    @staticmethod
    def continuum(egrid: JAXArray, params: NameValMapping) -> JAXArray:
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
        The total photon flux :math:`K` of the line, in units of ph cm⁻² s⁻¹.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    method : {'trapz', 'simpson'}, optional
        Numerical integration method. Defaults to 'trapz'.
    """

    _config = (
        ParamConfig('El', r'E_\mathrm{l}', 'keV', 6.5, 0.0, 1e6),
        ParamConfig('sigma', r'\sigma', 'keV', 0.1, 0.0, 20),
        ParamConfig('K', 'K', 'ph cm^-2 s^-1', 1.0, 1e-10, 1e10),
    )

    @staticmethod
    def continuum(egrid: JAXArray, params: NameValMapping) -> JAXArray:
        El = params['El']
        sigma = params['sigma']
        K = params['K']
        return K * stats.norm.pdf(x=egrid, loc=El, scale=sigma)


class LogParabola(NumIntAdditive):
    pass


class Lorentz(AnaIntAdditive):
    r"""Lorentzian line profile.

    .. math::
        N(E) = \frac{K/\mathcal{F}}{(E - E_\mathrm{l})^2 + (\sigma/2)^2},

    where

    .. math::
        \mathcal{F}=\int_0^{+\infty}
        \frac{1}{(E - E_\mathrm{l})^2 + (\sigma/2)^2} \ \mathrm{d}E.

    Parameters
    ----------
    El : Parameter, optional
        The line energy :math:`E_\mathrm{l}`, in units of keV.
    sigma : Parameter, optional
        The FWHM line width :math:`\sigma`, in units of keV.
    K : Parameter, optional
        The total photon flux :math:`K` of the line, in units of ph cm⁻² s⁻¹.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    """

    _config = (
        ParamConfig('El', r'E_\mathrm{l}', 'keV', 6.5, 1e-6, 1e6),
        ParamConfig('sigma', r'\sigma', 'keV', 0.1, 1e-3, 20),
        ParamConfig('K', 'K', 'ph cm^-2 s^-1', 1.0, 1e-10, 1e10),
    )

    @staticmethod
    def integral(egrid: JAXArray, params: NameValMapping) -> JAXArray:
        El = params['El']
        sigma = params['sigma']
        K = params['K']
        x = 2.0 * (egrid - El) / sigma
        integral = jnp.arctan(x)
        norm = 2.0 * K / (jnp.pi + 2 * jnp.arctan(2 * El / sigma))
        return norm * (integral[1:] - integral[:-1])


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
        The amplitude :math:`K`, in units of ph cm⁻² s⁻¹ keV⁻¹.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    method : {'trapz', 'simpson'}, optional
        Numerical integration method. Defaults to 'trapz'.
    """

    _config = (
        ParamConfig('kT', 'kT', 'keV', 30.0, 0.1, 1e3),
        ParamConfig('K', 'K', 'ph cm^-2 s^-1 keV^-1', 1.0, 1e-10, 1e10),
    )

    @staticmethod
    def continuum(egrid: JAXArray, params: NameValMapping) -> JAXArray:
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
        The amplitude :math:`K`, in units of ph cm⁻² s⁻¹ keV⁻¹.
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
        ParamConfig('K', 'K', 'ph cm^-2 s^-1 keV^-1', 1.0, 1e-10, 1e10),
    )

    @staticmethod
    def continuum(egrid: JAXArray, params: NameValMapping) -> JAXArray:
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
        Power law photon index :math:`\alpha`, dimensionless.
    K : Parameter, optional
        Amplitude :math:`K`, in units of ph cm⁻² s⁻¹ keV⁻¹.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    """

    _config = (
        ParamConfig('alpha', r'\alpha', '', 1.01, -3.0, 10.0),
        ParamConfig('K', 'K', 'ph cm^-2 s^-1 keV^-1', 1.0, 1e-10, 1e10),
    )

    @staticmethod
    def integral(egrid: JAXArray, params: NameValMapping) -> JAXArray:
        return params['K'] * _powerlaw_integral(egrid, params['alpha'])


class PLFluxNorm(AnaIntAdditive):
    _args = ('emin', 'emax')
    _energy: bool

    def __init__(
        self,
        emin: float | int,
        emax: float | int,
        params: dict,
        latex: str | None,
    ):
        emin = float(emin)
        emax = float(emax)

        if emin >= emax:
            raise ValueError('emin must be less than emax')

        self._emin = emin
        self._emax = emax

        super().__init__(params, latex)

    @property
    def eval(self) -> CompEval:
        if self._integral_jit is None:
            emin = self._emin
            emax = self._emax
            energy = self._energy
            fn = jax.jit(self.integral)

            def integral(egrid: JAXArray, params: NameValMapping) -> JAXArray:
                if energy:
                    keV_to_erg = 1.602176634e-9
                    params_ = {'alpha': params['alpha'] - 1.0}
                    f = fn(jnp.array([emin, emax]), params_)
                    factor = params['F'] / (f * keV_to_erg)
                else:
                    f = fn(jnp.array([emin, emax]), params)
                    factor = params['F'] / f

                return factor * fn(egrid, params)

            self._integral_jit = jax.jit(integral)

        return self._integral_jit

    @staticmethod
    def integral(egrid: JAXArray, params: NameValMapping) -> JAXArray:
        return _powerlaw_integral(egrid, params['alpha'])


class PLPhFlux(PLFluxNorm):
    r"""Power law function with photon flux used as normalization.

    .. math::
        N(E) &=
            \mathcal{F}_\mathrm{ph}
            \left[
                \int_{E_\mathrm{min}}^{E_\mathrm{max}}
                \left(\frac{E}{E_0}\right)^{-\alpha} \, \mathrm{d}E
            \right]^{-1}
            \left(\frac{E}{E_0}\right)^{-\alpha}\\
            &=
            \mathcal{F}_\mathrm{ph} (1 - \alpha)
            \left[
                \left(\frac{E_\mathrm{max}}{E_0}\right)^{1 - \alpha}
                - \left(\frac{E_\mathrm{min}}{E_0}\right)^{1 - \alpha}
            \right]^{-1}
            \left(\frac{E}{E_0}\right)^{-\alpha},

    where :math:`E_0` is the pivot energy fixed at 1 keV.

    Parameters
    ----------
    emin : float or int
        Minimum energy of the band to calculate the flux, in units of keV.
    emax : float or int
        Maximum energy of the band to calculate the flux, in units of keV.
    alpha : Parameter, optional
        Power law photon index :math:`\alpha`, dimensionless.
    F : Parameter, optional
        Photon flux :math:`\mathcal{F}_\mathrm{ph}`, in units of ph cm⁻² s⁻¹.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    """

    _energy: bool = False
    _config = (
        ParamConfig('alpha', r'\alpha', '', 1.01, -3.0, 10.0),
        ParamConfig(
            'F', r'\mathcal{F}_\mathrm{ph}', 'ph cm^-2 s^-1', 1.0, 0.01, 1e10
        ),
    )


class PLEnFlux(PLFluxNorm):
    r"""Power law function with energy flux used as normalization.

    .. math::
        N(E) &=
            \mathcal{F}_\mathrm{en}
            \left[
                \int_{E_\mathrm{min}}^{E_\mathrm{max}}
                \left(\frac{E}{E_0}\right)^{-\alpha} \, E \, \mathrm{d}E
            \right]^{-1}
            \left(\frac{E}{E_0}\right)^{-\alpha}\\
            &=
            \mathcal{F}_\mathrm{en} (2 - \alpha)
            \left[
                \left(\frac{E_\mathrm{max}}{E_0}\right)^{2 - \alpha}
                - \left(\frac{E_\mathrm{min}}{E_0}\right)^{2 - \alpha}
            \right]^{-1}
            \left(\frac{E}{E_0}\right)^{-\alpha},

    where :math:`E_0` is the pivot energy fixed at 1 keV.

    Parameters
    ----------
    emin : float or int
        Minimum energy of the band to calculate the flux, in units of keV.
    emax : float or int
        Maximum energy of the band to calculate the flux, in units of keV.
    alpha : Parameter, optional
        Power law photon index :math:`\alpha`, dimensionless.
    F : Parameter, optional
        Energy flux :math:`\mathcal{F}_\mathrm{en}`, in units of erg cm⁻² s⁻¹.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    """

    _energy: bool = True
    _config = (
        ParamConfig('alpha', r'\alpha', '', 1.01, -3.0, 10.0),
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


class SmoothlyBrokenPL(NumIntAdditive):
    r"""Smoothly broken power law.

    .. math::
        N(E) = K
            \left( \frac{E}{E_0} \right) ^ {-\alpha_1}
            \left\{
                2\left[
                    1 + \left(\frac{E}{E_\mathrm{b}}
                        \right)^{\left( \alpha_2 - \alpha_1 \right) / \rho}
                \right]
            \right\}^{-\rho},

    where :math:`E_0` is the pivot energy fixed at 1 keV.

    Parameters
    ----------
    alpha1 : Parameter, optional
        The low-energy power law index :math:`\alpha_1`, dimensionless.
    Eb : Parameter, optional
        The break energy :math:`E_\mathrm{b}`, in units of keV.
    alpha2 : Parameter, optional
        The high-energy power law index :math:`\alpha_2`, dimensionless.
    rho : Parameter, optional
        The smoothness parameter :math:`\rho`, dimensionless.
    K : Parameter, optional
        The amplitude :math:`K`, in units of ph cm⁻² s⁻¹ keV⁻¹.
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    method : {'trapz', 'simpson'}, optional
        Numerical integration method. Defaults to 'trapz'.
    """

    _config = (
        ParamConfig('alpha1', 'alpha_1', '', 1.0, -3.0, 10.0),
        ParamConfig('Eb', r'E_\mathrm{b}', 'keV', 5.0, 1e-2, 1e6),
        ParamConfig('alpha2', 'alpha_2', '', 2.0, -3.0, 10.0),
        ParamConfig('rho', r'\rho', '', 1.0, 1e-2, 1e2, fixed=True),
        ParamConfig('K', 'K', 'ph cm^-2 s^-1 keV^-1', 1.0, 1e-10, 1e10),
    )

    @staticmethod
    def continuum(egrid: JAXArray, params: NameValMapping) -> JAXArray:
        alpha1 = params['alpha1']
        Eb = params['Eb']
        alpha2 = params['alpha2']
        rho = params['rho']
        K = params['K']

        e = egrid / Eb
        x = (alpha2 - alpha1) / rho * jnp.log(e)

        threshold = 30
        alpha = jnp.where(x > threshold, alpha2, alpha1)
        r = jnp.where(jnp.abs(x) > threshold, 0.5, 0.5 * (1.0 + jnp.exp(x)))

        return K * jnp.power(e, -alpha) * jnp.power(r, -rho)
