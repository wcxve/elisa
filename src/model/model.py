import pytensor.tensor as pt
from .base import AutoGradOp, SpectralComponent, SpectralModel

__all__ = [
    'Band', 'BandEp',
    'BlackBody', 'BlackBodyRad',
    'Comptonized', 'CutoffPowerlaw',
    'OTTB',
    'Powerlaw',
    'Constant',
]


class BandOp(AutoGradOp):
    optype = 'add'
    def __init__(self, alpha, beta, Ec, integral_method='simpson'):
        super().__init__([alpha, beta, Ec], integral_method)

    def _NE(self, E):
        alpha, beta, Ec = self._pars
        Epiv = 100.0

        # workaround for beta > alpha, as in xspec
        amb_ = alpha - beta
        inv_Ec = 1.0 / Ec
        amb = pt.switch(pt.lt(amb_, inv_Ec), inv_Ec, amb_)

        Ebreak = Ec*amb
        return pt.exp(pt.switch(
            pt.lt(E, Ebreak),
            alpha * pt.log(E / Epiv) - E / Ec,
            amb * pt.log(amb * Ec / Epiv) - amb + beta * pt.log(E / Epiv)
        ))

class BandEpOp(AutoGradOp):
    optype = 'add'
    def __init__(self, alpha, beta, Ep, integral_method='simpson'):
        super().__init__([alpha, beta, Ep], integral_method)

    def _NE(self, E):
        alpha, beta, Ep = self._pars
        Epiv = 100.0
        Ec = Ep / (2.0 + alpha)
        Ebreak = (alpha - beta) * Ec

        # workaround for beta > alpha, as in xspec
        amb_ = alpha - beta
        inv_Ec = 1.0 / Ec
        amb = pt.switch(pt.lt(amb_, inv_Ec), inv_Ec, amb_)

        return pt.exp(pt.switch(
            pt.lt(E, Ebreak),
            alpha * pt.log(E / Epiv) - E / Ec,
            amb * pt.log(amb * Ec / Epiv) - amb + beta * pt.log(E / Epiv)
        ))

class BlackBodyOp(AutoGradOp):
    optype = 'add'
    def __init__(self, kT, integral_method='simpson'):
        super().__init__([kT], integral_method)

    def _NE(self, E):
        kT, = self._pars
        return 8.0525 * E * E / (kT * kT * kT * kT * pt.expm1(E / kT))

class BlackBodyRadOp(AutoGradOp):
    optype = 'add'
    def __init__(self, kT, integral_method='simpson'):
        super().__init__([kT], integral_method)

    def _NE(self, E):
        kT, = self._pars
        return 1.0344e-3 * E * E / pt.expm1(E / kT)

class ComptonizedOp(AutoGradOp):
    optype = 'add'
    def __init__(self, alpha, Ep, integral_method='simpson'):
        super().__init__([alpha, Ep], integral_method)

    def _NE(self, E):
        alpha, Ep = self._pars
        # return pt.exp(-E * (2.0 + alpha) / Ep + alpha * pt.log(E))
        return pt.pow(E, alpha) * pt.exp(-E * (2.0 + alpha) / Ep)

class CutoffPowerlawOp(AutoGradOp):
    optype = 'add'
    def __init__(self, PhoIndex, Ec, integral_method='simpson'):
        super().__init__([PhoIndex, Ec], integral_method)

    def _NE(self, E):
        PhoIndex, Ec = self._pars
        return pt.pow(E, -PhoIndex) * pt.exp(-E / Ec)

class OTTBOp(AutoGradOp):
    optype = 'add'
    def __init__(self, kT, integral_method='simpson'):
        super().__init__([kT], integral_method)

    def _NE(self, E):
        kT, = self._pars
        Epiv = 1.0
        return pt.exp((Epiv - E) / kT) * Epiv / E

class PowerlawOp(AutoGradOp):
    optype = 'add'
    def __init__(self, PhoIndex):
        super().__init__([PhoIndex])

    def _eval_flux(self, ebins):
        PhoIndex, = self._pars
        alpha = 1.0 - PhoIndex
        # integral = ifelse(
        #     pt.eq(alpha, 0.0),
        #     pt.log(ebins),
        #     pt.pow(ebins, alpha) / alpha,
        # )
        integral = pt.pow(ebins, alpha) / alpha

        return integral[1:] - integral[:-1]

class ConstantOp(AutoGradOp):
    optype = 'mul'
    def __init__(self, factor):
        super().__init__([factor])

    def _eval(self, ebins, flux=None):
        return self._pars[0]


class BandComponent(SpectralComponent):
    _comp_name = 'Band'
    _config = {
        'alpha': [-1.0, -10.0, 5.0, False, False],
        'beta': [-2.0, -10.0, 10.0, False, False],
        'Ec': [300.0, 10.0, 10000.0, False, False],
        'norm': [1, 1e-10, 1e10, False, False],
    }
    _op_class = BandOp
    def __init__(self, alpha, beta, Ec, norm, name=None, integral_method='simpson'):
        kwargs = {
            k: v for k, v in locals().items()
            if k not in ('self', '__class__')
        }
        super().__init__(**kwargs)

class BandEpComponent(SpectralComponent):
    _comp_name = 'BandEp'
    _config = {
        'alpha': [-1.0, -10.0, 5.0, False, False],
        'beta': [-2.0, -10.0, 10.0, False, False],
        'Ep': [300.0, 10.0, 10000.0, False, False],
        'norm': [1, 1e-10, 1e10, False, False],
    }
    _op_class = BandEpOp
    def __init__(self, alpha, beta, Ep, norm, name=None, integral_method='simpson'):
        kwargs = {
            k: v for k, v in locals().items()
            if k not in ('self', '__class__')
        }
        super().__init__(**kwargs)

class BlackBodyComponent(SpectralComponent):
    _comp_name = 'BB'
    _config = {
        'kT': [3.0, 0.0001, 200.0, False, False],
        'norm': [1, 1e-10, 1e10, False, False],
    }
    _op_class = BlackBodyOp
    def __init__(self, kT, norm, name=None, integral_method='simpson'):
        kwargs = {
            k: v for k, v in locals().items()
            if k not in ('self', '__class__')
        }
        super().__init__(**kwargs)

class BlackBodyRadComponent(SpectralComponent):
    _comp_name = 'BBrad'
    _config = {
        'kT': [3.0, 0.0001, 200.0, False, False],
        'norm': [1, 1e-10, 1e10, False, False],
    }
    _op_class = BlackBodyRadOp
    def __init__(self, kT, norm, name=None, integral_method='simpson'):
        kwargs = {
            k: v for k, v in locals().items()
            if k not in ('self', '__class__')
        }
        super().__init__(**kwargs)

class ComptonizedComponent(SpectralComponent):
    _comp_name = 'Compt'
    _config = {
        'alpha': [-1.0, -10.0, 3.0, False, False],
        'Ep': [15.0, 0.01, 10000.0, False, False],
        'norm': [1, 1e-10, 1e10, False, False],
    }
    _op_class = ComptonizedOp
    def __init__(self, alpha, Ep, norm, name=None, integral_method='simpson'):
        kwargs = {
            k: v for k, v in locals().items()
            if k not in ('self', '__class__')
        }
        super().__init__(**kwargs)

class CutoffPowerlawComponent(SpectralComponent):
    _comp_name = 'CPL'
    _config = {
        'PhoIndex': [1.0, -3.0, 10.0, False, False],
        'Ec': [15.0, 0.01, 10000.0, False, False],
        'norm': [1, 1e-10, 1e10, False, False],
    }
    _op_class = CutoffPowerlawOp
    def __init__(self, PhoIndex, Ec, norm, name=None, integral_method='simpson'):
        kwargs = {
            k: v for k, v in locals().items()
            if k not in ('self', '__class__')
        }
        super().__init__(**kwargs)

class OTTBComponent(SpectralComponent):
    _comp_name = 'OTTB'
    _config = {
        'kT': [30.0, 0.1, 1000.0, False, False],
        'norm': [1, 1e-10, 1e10, False, False],
    }
    _op_class = OTTBOp
    def __init__(self, kT, norm, name=None, integral_method='simpson'):
        kwargs = {
            k: v for k, v in locals().items()
            if k not in ('self', '__class__')
        }
        super().__init__(**kwargs)

class PowerlawComponent(SpectralComponent):
    _comp_name = 'PL'
    _config = {
        'PhoIndex': [1.01, -3.0, 10.0, False, False],
        'norm': [1, 1e-10, 1e10, False, False]
    }
    _op_class = PowerlawOp
    def __init__(self, PhoIndex, norm, name=None):
        kwargs = {
            k: v for k, v in locals().items()
            if k not in ('self', '__class__')
        }
        super().__init__(**kwargs)

class ConstantComponent(SpectralComponent):
    _comp_name = 'constant'
    _config = {
        'factor': [1.0, 1e-5, 1e5, False, False],
    }
    _op_class = ConstantOp
    def __init__(self, factor, name=None):
        kwargs = {
            k: v for k, v in locals().items()
            if k not in ('self', '__class__')
        }
        super().__init__(**kwargs)


class Band(SpectralModel):
    def __init__(self, alpha=None, beta=None, Ec=None, norm=None, name=None, integral_method='simpson'):
        super().__init__([BandComponent(alpha, beta, Ec, norm, name, integral_method)])

class BandEp(SpectralModel):
    def __init__(self, alpha=None, beta=None, Ep=None, norm=None, name=None, integral_method='simpson'):
        super().__init__([BandEpComponent(alpha, beta, Ep, norm, name, integral_method)])

class BlackBody(SpectralModel):
    def __init__(self, kT=None, norm=None, name=None, integral_method='simpson'):
        super().__init__([BlackBodyComponent(kT, norm, name, integral_method)])

class BlackBodyRad(SpectralModel):
    def __init__(self, kT=None, norm=None, name=None, integral_method='simpson'):
        super().__init__([BlackBodyRadComponent(kT, norm, name, integral_method)])

class Comptonized(SpectralModel):
    def __init__(self, alpha=None, Ep=None, norm=None, name=None, integral_method='simpson'):
        super().__init__([ComptonizedComponent(alpha, Ep, norm, name, integral_method)])

class CutoffPowerlaw(SpectralModel):
    def __init__(self, PhoIndex=None, Ec=None, norm=None, name=None, integral_method='simpson'):
        super().__init__([CutoffPowerlawComponent(PhoIndex, Ec, norm, name, integral_method)])

class OTTB(SpectralModel):
    def __init__(self, kT=None, norm=None, name=None, integral_method='simpson'):
        super().__init__([OTTBComponent(kT, norm, name, integral_method)])

class Powerlaw(SpectralModel):
    def __init__(self, PhoIndex=None, norm=None, name=None):
        super().__init__([PowerlawComponent(PhoIndex, norm, name)])

class Constant(SpectralModel):
    def __init__(self, factor=None, name=None):
        super().__init__([ConstantComponent(factor, name)])
