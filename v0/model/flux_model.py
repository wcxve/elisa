import numpy as np
import pytensor.tensor as pt

from .base import SpectralComponent, SpectralModel


__all__ = ['EnergyFlux', 'PhotonFlux']


class FluxComponents(SpectralComponent):
    mtype = 'con'
    _config = {
        'log10Flux': (-12.0, -30, 30, False, False),
        'Flux': (1, 0.01, 1e10, False, False)
    }
    def __init__(self, name, par, is_energy=True):
        self.name = name
        self._comp_name = name

        if is_energy:
            par_name = 'log10Flux'
            self._config = {
                'log10Flux': (-12.0, -30, 30, False, False),
            }
        else:
            par_name = 'Flux'
            self._config = {
                'Flux': (1, 0.01, 1e10, False, False)
            }

        self._pars_dict = {}
        self._set_par(par_name, par)

        self._pars_tensor = {
            name: pt.scalar(par.name)
            for name, par in self._pars_dict.items()
        }

class FluxModel(SpectralModel):
    @property
    def Emin(self):
        return self._Emin

    @Emin.setter
    def Emin(self, value):
        if type(value) in [float, int]:
            self._Emin = float(value)
        else:
            raise TypeError(
                'float type is required for `Emin`'
            )

    @property
    def Emax(self):
        return self._Emax

    @Emax.setter
    def Emax(self, value):
        if type(value) in [float, int]:
            self._Emax = float(value)
        else:
            raise TypeError(
                'float type is required for `Emax`'
            )

    @property
    def _eval_tensor(self):
        raise NotImplementedError

    def eval(self, *args, **kwargs):
        raise NotImplementedError


class EnergyFlux(FluxModel):
    def __init__(self, Emin: float, Emax: float, log10Flux=None, ngrid=1000, elog=True):
        components = [FluxComponents('EnFlux', log10Flux, True)]
        self._components = components
        self.Emin = Emin
        self.Emax = Emax
        self.ngrid = ngrid
        self.elog = elog
        super(SpectralModel, self).__setattr__('EnFlux', components[0])

    def __call__(self, flux, model, fit_call=True):
        if fit_call:
            log10Flux = self.EnFlux._pars_dict['log10Flux'].rv
        else:
            log10Flux = self.EnFlux._pars_tensor['log10Flux']

        if self.elog:  # evenly-spaced energies in log space
            ebins = np.geomspace(self.Emin, self.Emax, self.ngrid)
            emid = np.sqrt(ebins[:-1] * ebins[1:])
        else:  # evenly-spaced energies in linear space
            ebins = np.linspace(self.Emin, self.Emax, self.ngrid)
            emid = (ebins[:-1] + ebins[1:]) / 2.0

        mflux = pt.sum(1.6022e-9 * emid * model(ebins, fit_call=fit_call))

        return pt.pow(10, log10Flux) / mflux * flux


class PhotonFlux(FluxModel):
    def __init__(self, Emin: float, Emax: float, Flux=None, ngrid=1000, elog=True):
        components = [FluxComponents('PhFlux', Flux, False)]
        self._components = components
        self.Emin = Emin
        self.Emax = Emax
        self.ngrid = ngrid
        self.elog = elog
        super(SpectralModel, self).__setattr__('PhFlux', components[0])

    def __call__(self, flux, model, fit_call=True):
        if fit_call:
            Flux = self.PhFlux._pars_dict['Flux'].rv
        else:
            Flux = self.PhFlux._pars_tensor['Flux']

        if self.elog:  # evenly-spaced energies in log space
            ebins = np.geomspace(self.Emin, self.Emax, self.ngrid)
        else:  # evenly-spaced energies in linear space
            ebins = np.linspace(self.Emin, self.Emax, self.ngrid)

        mflux = pt.sum(model(ebins, fit_call=fit_call))

        return Flux / mflux * flux
