import numpy as np
import pytensor
import pytensor.tensor as pt
import xspec_models_cxc as xspec_models
from ..base import NumericGradOp


class XspecNumericGradOp(NumericGradOp):
    modname = None
    optype = None
    def __init__(self, grad_method='f', eps=1e-7, **par_kwargs):
        if self.modname is None:
            raise ValueError('`modname` should be specified')

        if self.optype is None:
            raise ValueError('`optype` should be specified')

        if self.modname not in xspec_models.list_models():
            raise ValueError(f'Model "{self.modname}" not found')

        par_name = [p.name for p in xspec_models.info(self.modname).parameters]
        pars = [par_kwargs[p] for p in par_name]
        super().__init__(pars, grad_method, eps)

        xs_func = getattr(xspec_models, self.modname)
        language = xspec_models.info(self.modname).language.name

        if pytensor.config.floatX == 'float64' and language == 'F77Style4':
            def _xs_func(*args):
                return np.float64(xs_func(*args))
        elif pytensor.config.floatX == 'float32' and language != 'F77Style4':
            def _xs_func(*args):
                return np.float32(xs_func(*args))
        else:
            _xs_func = xs_func

        self._xs_func = _xs_func

        self.itypes = [
            pt.TensorType('floatX', shape=())
            for _ in pars
        ]
        self.itypes.append(
            pt.TensorType('floatX', shape=(None,))  # ebins
        )

        if self.optype == 'con':
            self.itypes.append(
                pt.TensorType('floatX', shape=(None,))  # flux
            )

    def _perform(self, *inputs):
        pars = np.array(inputs[:self.npars])
        return self._xs_func(pars, *inputs[self.npars:])
