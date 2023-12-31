"""Models of additive type."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import jax.numpy as jnp

from .base import Component, ParamConfig
from .integral import integral, list_methods


__all__ = [
    'Band', 'BandEp',
    'Bbody', 'Bbodyrad',
    'Compt', 'Cutoffpl',
    'OTTB',
    'Powerlaw',
]


class AdditiveComponent(Component, ABC):
    """Prototype class to define additive component."""

    @property
    def type(self) -> str:
        """Model type is additive."""
        return 'add'

    @property
    def _func(self) -> Callable:
        """Return function that integrates continnum over energy grid."""
        return self._integral

    @abstractmethod
    def _integral(self, *args) -> Callable:
        """Return integral function, overriden by subclass."""
        pass


class NumIntAdditive(AdditiveComponent, ABC):
    """Prototype class with numerical integral to define additive model."""

    _extra_kw = (('method', 'default'),)

    def __init__(self, method='default', **kwargs):
        self.method = str(method)
        super().__init__(**kwargs)

    @property
    def method(self) -> str:
        """Numerical integral method."""
        return self._method

    @method.setter
    def method(self, value: str):
        """Numerical integral method."""
        value = str(value)

        methods = list_methods()
        if value not in methods:
            methods = '"' + '", "'.join(methods) + '"'
            raise ValueError(
                f'available numerical integral options are: {methods}, '
                f'but got "{value}"'
            )

        self._method = value

    @property
    def _integral(self) -> Callable:
        """Wrap continnum function with numerical integral method."""
        name = self.__class__.__name__.lower()
        return integral(self._continnum, name, self._method)

    @staticmethod
    @abstractmethod
    def _continnum(*args, **kwargs):
        """Continnum to be integrated over energy grids."""
        pass


class Bbody(NumIntAdditive):
    """TODO"""

    _config = (
        ParamConfig('kT', 'kT', 3.0, 0.0001, 200.0, False, False),
        ParamConfig('K', 'K', 1.0, 1e-10, 1e10, False, False),
    )

    @staticmethod
    def _continnum(egrid, kT, K):
        e = egrid
        x = e / kT
        tmp = 8.0525 * K * e / (kT * kT * kT)
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
                tmp * x / jnp.expm1(x_)
            )
        )
        # return 8.0525 * K * e*e / (kT*kT*kT*kT * jnp.expm1(energy / kT))


class Bbodyrad(NumIntAdditive):
    """TODO"""

    _config = (
        ParamConfig('kT', 'kT', 3.0, 0.0001, 200.0, False, False),
        ParamConfig('K', 'K', 1.0, 1e-10, 1e10, False, False),
    )

    @staticmethod
    def _continnum(egrid, kT, K):
        e = egrid
        x = e / kT
        tmp = 1.0344e-3 * K * e
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
                tmp * e / jnp.expm1(x_)
            )
        )
        # return 1.0344e-3 * K * e*e / jnp.expm1(e / kT)


class Band(NumIntAdditive):
    """TODO"""

    _config = (
        ParamConfig('alpha', r'\alpha', -1.0, -10.0, 5.0, False, False),
        ParamConfig('beta', r'\beta', -2.0, -10.0, 10.0, False, False),
        ParamConfig('Ec', r'E_\mathrm{c}', 300.0, 10.0, 10000.0, False, False),
        ParamConfig('K', 'K', 1.0, 1e-10, 1e10, False, False),
    )

    @staticmethod
    def _continnum(egrid, alpha, beta, Ec, K):
        Epiv = 100.0
        # workaround for beta > alpha, as in xspec
        amb_ = alpha - beta
        inv_Ec = 1.0 / Ec
        amb = jnp.where(jnp.less(amb_, inv_Ec), inv_Ec, amb_)
        Ebreak = Ec*amb

        log_func = jnp.where(
            jnp.less(egrid, Ebreak),
            alpha * jnp.log(egrid / Epiv) - egrid / Ec,
            amb * jnp.log(amb * Ec / Epiv) - amb + beta * jnp.log(egrid / Epiv)
        )
        return K * jnp.exp(log_func)


class BandEp(NumIntAdditive):
    """TODO"""

    _config = (
        ParamConfig('alpha', r'\alpha', -1.0, -10.0, 5.0, False, False),
        ParamConfig('beta', r'\beta', -2.0, -10.0, 10.0, False, False),
        ParamConfig('Ep', r'E_\mathrm{p}', 300.0, 10.0, 10000.0, False, False),
        ParamConfig('K', 'K', 1.0, 1e-10, 1e10, False, False),
    )

    @staticmethod
    def _continnum(egrid, alpha, beta, Ep, K):
        e = egrid
        Epiv = 100.0
        Ec = Ep / (2.0 + alpha)
        Ebreak = (alpha - beta) * Ec

        # workaround for beta > alpha, as in xspec
        amb_ = alpha - beta
        inv_Ec = 1.0 / Ec
        amb = jnp.where(jnp.less(amb_, inv_Ec), inv_Ec, amb_)

        log_func = jnp.where(
            jnp.less(e, Ebreak),
            alpha * jnp.log(e / Epiv) - e / Ec,
            amb * jnp.log(amb * Ec / Epiv) - amb + beta * jnp.log(e / Epiv)
        )
        return K * jnp.exp(log_func)


class Compt(NumIntAdditive):
    """TODO"""

    _config = (
        ParamConfig('alpha', r'\alpha', -1.0, -10.0, 3.0, False, False),
        ParamConfig('Ep', r'E_\mathrm{p}', 15.0, 0.01, 10000.0, False, False),
        ParamConfig('K', 'K', 1.0, 1e-10, 1e10, False, False),
    )

    @staticmethod
    def _continnum(egrid, alpha, Ep, K):
        e = egrid
        return K * jnp.power(e, alpha) * jnp.exp(-e * (2.0 + alpha) / Ep)


class Cutoffpl(NumIntAdditive):
    """TODO"""

    _config = (
        ParamConfig('alpha', r'\alpha', -1.0, -10.0, 3.0, False, False),
        ParamConfig('Ec', r'E_\mathrm{c}', 15.0, 0.01, 10000.0, False, False),
        ParamConfig('K', 'K', 1.0, 1e-10, 1e10, False, False),
    )

    @staticmethod
    def _continnum(egrid, alpha, Ec, K):
        e = egrid
        return K * jnp.power(e, alpha) * jnp.exp(-e / Ec)


class OTTB(NumIntAdditive):
    """TODO"""

    _config = (
        ParamConfig('kT', 'kT', 30.0, 0.1, 1000.0, False, False),
        ParamConfig('K', 'K', 1.0, 1e-10, 1e10, False, False),
    )

    @staticmethod
    def _continnum(egrid, kT, K):
        e = egrid
        Epiv = 1.0
        return K * jnp.exp((Epiv - e) / kT) * Epiv / e


class Powerlaw(AdditiveComponent):
    """TODO"""

    _config = (
        ParamConfig('alpha', r'\alpha', 1.01, -3.0, 10.0, False, False),
        ParamConfig('K', 'K', 1.0, 1e-10, 1e10, False, False),
    )

    @staticmethod
    def _integral(egrid, alpha, K):
        # we ignore the case of alpha = 1.0
        tmp = 1.0 - alpha
        f = K / tmp * jnp.power(egrid, tmp)
        return f[1:] - f[:-1]
