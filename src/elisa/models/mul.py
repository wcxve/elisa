"""Multiplicative models."""

from __future__ import annotations

import warnings
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from elisa.models.model import (
    AnaIntMultiplicative,
    NumIntMultiplicative,
    ParamConfig,
)

if TYPE_CHECKING:
    from elisa.util.typing import CompEval, JAXArray, NameValMapping

__all__ = [
    'Constant',
    'Edge',
    'ExpAbs',
    'ExpFac',
    'GAbs',
    'HighECut',
    'PLAbs',
    'PhAbs',
    'TBAbs',
    'WAbs',
]


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
    def continuum(egrid: JAXArray, params: NameValMapping) -> JAXArray:
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
    def continuum(egrid: JAXArray, params: NameValMapping) -> JAXArray:
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
    def continuum(egrid: JAXArray, params: NameValMapping) -> JAXArray:
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
    def continuum(egrid: JAXArray, params: NameValMapping) -> JAXArray:
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
    def continuum(egrid: JAXArray, params: NameValMapping) -> JAXArray:
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


def _make_interp(egrid, xsect):
    xp = np.log(egrid)
    fp = np.log(xsect)
    return jax.jit(
        lambda e: jnp.exp(
            jnp.interp(
                jnp.log(e), xp, fp, left='extrapolate', right='extrapolate'
            )
        )
    )


with h5py.File(Path(__file__).parent / 'tables' / 'xsect.hdf5') as f:
    _XSECT_INTERP = {
        mabs: {
            xsect: {
                abund: _make_interp(
                    f['energy'][:], f[f'{mabs}/{xsect}/{abund}'][:]
                )
                for abund in f[f'{mabs}/{xsect}'].keys()
            }
            for xsect in f[mabs].keys()
        }
        for mabs in [k for k in f.keys() if k != 'energy']
    }
del f


class PhotonAbsorption(NumIntMultiplicative):
    r"""Photon absorption model.

    .. note::
        The photon cross-sections are obtained by interpolating cross-sections
        table, which is the same as threeml does, except that if input energy
        is outside the table energy range, extrapolated value is used.
    """

    _kwargs = ('abund', 'xsect', 'method')
    _default_abund: str
    _default_xsect: str

    def __init__(
        self,
        params: dict,
        latex: str | None,
        method: str | None,
        abund: str | None,
        xsect: str | None,
    ):
        if self.__class__.__name__ == 'WAbs':
            warnings.warn(
                'The WAbs model is obsolete and is only included for '
                'comparison with historical results. The TBAbs model '
                'should be used for the ISM or PhAbs for general '
                'photoelectric absorption.',
                DeprecationWarning,
                stacklevel=4,
            )

        if abund is None:
            abund = self._default_abund
        self.abund = abund
        if xsect is None:
            xsect = self._default_xsect
        self.xsect = xsect
        super().__init__(params, latex, method)

    @property
    def eval(self) -> CompEval:
        """Get photon absorption model function."""
        if self._continuum_jit is None:
            self._continuum_jit = jax.jit(
                self.continuum, static_argnums=(2, 3, 4)
            )

        abs_model = self.__class__.__name__.lower()
        abund = self.abund
        xsect = self.xsect
        continuum = jax.jit(
            lambda egrid, params: self._continuum_jit(
                egrid, params, abs_model, abund, xsect
            )
        )
        return self._make_integral(continuum)

    @staticmethod
    def continuum(
        egrid: JAXArray,
        params: NameValMapping,
        abs_model: str,
        abund: str,
        xsect: str,
    ) -> JAXArray:
        """Photon absorption model.

        Parameters
        ----------
        egrid : ndarray
            Photon energy grid in units of keV.
        params : dict
            Parameter dict for the convolution model.
        abs_model : str
            Photon absorption model name.
        abund : str
            Abundance table name.
        xsect : str
            Photon cross-sections.

        Returns
        -------
        jax.Array
            The model value at `egrid`, dimensionless.
        """
        sigma = _XSECT_INTERP[abs_model][xsect][abund](egrid)
        return jnp.exp(-params['nH'] * sigma)

    @property
    def abund(self) -> str:
        """Current abundance table."""
        return self._abund

    @abund.setter
    def abund(self, abund: str):
        abund = str(abund)
        if abund not in self.abund_list():
            available = ', '.join(self.abund_list())
            raise ValueError(f'available abundance: {available}')
        self._abund = abund

    @staticmethod
    @abstractmethod
    def abund_list() -> list[str]:
        """Get available abundance list."""
        pass

    @property
    def xsect(self) -> str:
        """Current photon cross-section."""
        return self._xsect

    @xsect.setter
    def xsect(self, xsect: str | None):
        xsect = str(xsect)
        if xsect not in self.xsect_list():
            available = ', '.join(self.xsect_list())
            raise ValueError(f'available photon cross-section: {available}')
        self._xsect = xsect

    @staticmethod
    @abstractmethod
    def xsect_list() -> list[str]:
        """Get available photon cross-section list."""
        pass


class PhAbs(PhotonAbsorption):
    r"""Photoelectric absorption.

    .. math ::
        M(E) = \exp \left[ -\eta_\mathrm{H}\ \sigma(E) \right],

    where :math:`\sigma(E)` is the photo-electric cross-section, **NOT**
    including Thomson scattering.

    Parameters
    ----------
    nH : Parameter, optional
        The equivalent hydrogen column density :math:`\eta_\mathrm{H}`,
        in units of 10²² cm⁻².
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    abund : str, optional
        Abundance table to use. Available options are:

            * ``'angr'`` [1]_ (Photospheric, using Table 2)
            * ``'aspl'`` [2]_ (Photospheric, using Table 1)
            * ``'feld'`` [3]_
            * ``'aneb'`` [4]_
            * ``'grsa'`` [5]_
            * ``'wilm'`` [6]_
            * ``'lodd'`` [7]_ (Photospheric, using Table 1)
            * ``'lpgp'`` [8]_ (Photospheric, using Table 4)
            * ``'lpgs'`` [8]_ (Proto-solar, using Table 10)

        The default is ``'angr'``.
    xsect : str, optional
        Photon cross-section to use. Available options are:

            * ``'bcmc'`` [9]_ with a new He cross-section [10]_
            * ``'obcm'`` [9]_
            * ``'vern'`` [11]_

        The default is ``'vern'``.
    method : {'trapz', 'simpson'}, optional
        Numerical integration method. Defaults to 'trapz'.

    References
    ----------
    .. [1] `Anders & Grevesse 1989, Geochimica et Cosmochimica Acta, 53, 1,
            197-214 <https://doi.org/10.1016/0016-7037(89)90286-X>`__
    .. [2] `Asplund et al. 2009 ARAA, 47, 481 <https://doi.org/10.1146/annurev.astro.46.060407.145222>`__
    .. [3] `Feldman 1992, Phys. Scr. 46, 202 <https://doi.org/10.1088/0031-8949/46/3/002>`__
    .. [4] `Anders & Ebihara 1982, Geochimica et Cosmochimica Acta, 46, 11,
            2363-2380 <https://doi.org/10.1016/0016-7037(82)90208-3>`__
    .. [5] `Grevesse & Sauval 1998, Space Science Reviews, 85, 161–174 <https://doi.org/10.1023/A:1005161325181>`__
    .. [6] `Wilms et al 2000, ApJ, 542, 914 <https://doi.org/10.1086/317016>`__
    .. [7] `Lodders 2003, ApJ, 591, 1220 <https://doi.org/10.1086/375492>`__
    .. [8] `Lodders et al. 2009 <https://doi.org/10.1007/978-3-540-88055-4_34>`__
    .. [9] `Balucinska-Church & McCammon 1992, ApJ, 400, 699 <https://doi.org/10.1086/172032>`__
    .. [10] `Yan et al. 1998, ApJ, 496, 1044 <https://doi.org/10.1086/305420>`__
    .. [11] `Verner et al. 1996, ApJ, 465, 487 <https://doi.org/10.1086/177435>`__
    """

    _default_abund: str = 'angr'
    _default_xsect: str = 'vern'
    _config = (
        ParamConfig('nH', r'\eta_\mathrm{H}', '10^22 cm^-2', 1.0, 0.0, 1e6),
    )

    @staticmethod
    def abund_list() -> list[str]:
        """Get available abundance list."""
        return [
            'angr',
            'aspl',
            'feld',
            'aneb',
            'grsa',
            'wilm',
            'lodd',
            'lpgp',
            'lpgs',
        ]

    @staticmethod
    def xsect_list() -> list[str]:
        """Get available photon cross-section list."""
        return ['bcmc', 'obcm', 'vern']


class TBAbs(PhotonAbsorption):
    r"""The Tuebingen-Boulder ISM absorption model.

    This model calculates the cross-sections for X-ray absorption by the ISM as
    the sum of the cross-sections for X-ray absorption due to the gas-phase
    ISM, the grain-phase ISM, and the molecules in the ISM.

    Parameters
    ----------
    nH : Parameter, optional
        The equivalent hydrogen column density :math:`\eta_\mathrm{H}`,
        in units of 10²² cm⁻².
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    abund : str, optional
        Abundance table to use. Available options are:

            * ``'angr'`` [1]_ (Photospheric, using Table 2)
            * ``'aspl'`` [2]_ (Photospheric, using Table 1)
            * ``'feld'`` [3]_
            * ``'aneb'`` [4]_
            * ``'grsa'`` [5]_
            * ``'wilm'`` [6]_
            * ``'lodd'`` [7]_ (Photospheric, using Table 1)
            * ``'lpgp'`` [8]_ (Photospheric, using Table 4)
            * ``'lpgs'`` [8]_ (Proto-solar, using Table 10)

        The default is ``'wilm'``.
    xsect : str, optional
        Always use cross-section ``'vern'`` [9]_ as baseline.
    method : {'trapz', 'simpson'}, optional
        Numerical integration method. Defaults to 'trapz'.

    References
    ----------
    .. [1] `Anders & Grevesse 1989, Geochimica et Cosmochimica Acta, 53, 1,
            197-214 <https://doi.org/10.1016/0016-7037(89)90286-X>`__
    .. [2] `Asplund et al. 2009 ARAA, 47, 481 <https://doi.org/10.1146/annurev.astro.46.060407.145222>`__
    .. [3] `Feldman 1992, Phys. Scr. 46, 202 <https://doi.org/10.1088/0031-8949/46/3/002>`__
    .. [4] `Anders & Ebihara 1982, Geochimica et Cosmochimica Acta, 46, 11,
            2363-2380 <https://doi.org/10.1016/0016-7037(82)90208-3>`__
    .. [5] `Grevesse & Sauval 1998, Space Science Reviews, 85, 161–174 <https://doi.org/10.1023/A:1005161325181>`__
    .. [6] `Wilms et al 2000, ApJ, 542, 914 <https://doi.org/10.1086/317016>`__
    .. [7] `Lodders 2003, ApJ, 591, 1220 <https://doi.org/10.1086/375492>`__
    .. [8] `Lodders et al. 2009 <https://doi.org/10.1007/978-3-540-88055-4_34>`__
    .. [9] `Verner et al. 1996, ApJ, 465, 487 <https://doi.org/10.1086/177435>`__
    """

    _default_abund: str = 'angr'
    _default_xsect: str = 'vern'
    _config = (
        ParamConfig('nH', r'\eta_\mathrm{H}', '10^22 cm^-2', 1.0, 0.0, 1e6),
    )

    @staticmethod
    def abund_list() -> list[str]:
        """Get available abundance list."""
        return [
            'angr',
            'aspl',
            'feld',
            'aneb',
            'grsa',
            'wilm',
            'lodd',
            'lpgp',
            'lpgs',
        ]

    @staticmethod
    def xsect_list() -> list[str]:
        return ['vern']


class WAbs(PhotonAbsorption):
    r"""A photo-electric absorption using Wisconsin cross-sections [1]_.

    .. math ::
        M(E) = \exp \left[ -\eta_\mathrm{H}\ \sigma(E) \right],

    where :math:`\sigma(E)` is the photo-electric cross-section, **NOT**
    including Thomson scattering.

    .. warning ::
        The :class:`WAbs` model is obsolete and is only included for comparison
        with historical results. The :class:`TBAbs` model should be used for
        the ISM or :class:`PhAbs` for general photoelectric absorption.

    Parameters
    ----------
    nH : Parameter, optional
        The equivalent hydrogen column density :math:`\eta_\mathrm{H}`,
        in units of 10²² cm⁻².
    latex : str, optional
        :math:`\LaTeX` format of the component. Defaults to class name.
    abund : str, optional
        Always use abundance table ``'aneb'`` [2]_.
    xsect : str, optional
        Always use Wisconsin cross-sections [1]_.
    method : {'trapz', 'simpson'}, optional
        Numerical integration method. Defaults to 'trapz'.

    References
    ----------
    .. [1] `Morrison & McCammon 1983, ApJ, 270, 119 <https://doi.org/10.1086/161102>`__
    .. [2] `Anders & Ebihara 1982, Geochimica et Cosmochimica Acta, 46, 11,
            2363-2380 <https://doi.org/10.1016/0016-7037(82)90208-3>`__
    """

    _default_abund: str = 'aneb'
    _default_xsect: str = 'wabs'
    _config = (
        ParamConfig('nH', r'\eta_\mathrm{H}', '10^22 cm^-2', 1.0, 0.0, 1e6),
    )

    @staticmethod
    def abund_list() -> list[str]:
        return ['aneb']

    @staticmethod
    def xsect_list() -> list[str]:
        return ['wabs']
