"""Xspec model wrappers."""

from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp

from elisa.models.model import (
    Component,
    ComponentMeta,
    ConvolutionModel,
    ConvolvedModel,
    ParamConfig,
)
from elisa.util.misc import define_fdjvp

try:
    from xspex import (
        abundance as abundance,
        chatter as chatter,
        clear_db as clear_db,
        clear_model_string as clear_model_string,
        clear_xflt as clear_xflt,
        cosmology as cosmology,
        cross_section as cross_section,
        element_abundance as element_abundance,
        element_name as element_name,
        get_db as get_db,
        get_model_string as get_model_string,
        get_number_xflt as get_number_xflt,
        get_xflt as get_xflt,
        in_xflt as in_xflt,
        list_models as list_models,
        number_elements as number_elements,
        set_db as set_db,
        set_xflt as set_xflt,
        version as version,
    )
    from xspex.primitive import XSModel as _XSMODEL

    _HAS_XSPEC = True
except ImportError as e:
    _XSMODEL = {}
    _HAS_XSPEC = False
    warnings.warn(f'Xspec model library is not available: {e}', ImportWarning)

if TYPE_CHECKING:
    from elisa.models.model import Model, UniComponentModel
    from elisa.util.typing import (
        CompEval,
        CompIDParamValMapping,
        JAXArray,
        ModelEval,
    )

__all__ = []


class XspecComponentMeta(ComponentMeta):
    def __call__(
        cls, *args, **kwargs
    ) -> UniComponentModel | XspecConvolutionModel:
        if issubclass(cls, XspecConvolution):
            component = super(ComponentMeta, cls).__call__(*args, **kwargs)
            return XspecConvolutionModel(component)
        else:
            return super().__call__(*args, **kwargs)


class XspecComponent(Component, metaclass=XspecComponentMeta):
    """Xspec model wrapper."""

    _kwargs: tuple[str, ...] = ('grad_method', 'spec_num')
    _eval: CompEval | None = None

    def __init__(
        self,
        params: dict,
        latex: str | None,
        grad_method: Literal['central', 'forward'] | None,
        spec_num: int | None,
    ):
        self.grad_method = grad_method

        if spec_num is None:
            spec_num = 1
        self._spec_num = int(spec_num)

        super().__init__(params, latex)

    @property
    def grad_method(self) -> Literal['central', 'forward']:
        """Numerical differentiation method."""
        return self._grad_method

    @grad_method.setter
    def grad_method(self, value: Literal['central', 'forward'] | None):
        if value is None:
            value: Literal['central'] = 'central'

        if value not in {'central', 'forward'}:
            raise ValueError(
                f"supported methods are 'central' and 'forward', but got "
                f"'{value}'"
            )
        self._grad_method = value

    @property
    def spec_num(self) -> int:
        """Spectrum number."""
        return self._spec_num


class XspecAdditive(XspecComponent):
    @property
    def type(self) -> Literal['add']:
        return 'add'

    @property
    def eval(self) -> CompEval:
        if self._eval is not None:
            return self._eval

        _integral = jax.jit(define_fdjvp(self._integral, self.grad_method))

        def integral(egrid, params):
            return params['norm'] * _integral(egrid, params)

        self._eval = jax.jit(integral)

        return self._eval

    @property
    @abstractmethod
    def _integral(self):
        pass


class XspecMultiplicative(XspecComponent):
    @property
    def type(self) -> Literal['mul']:
        return 'mul'

    @property
    def eval(self) -> CompEval:
        if self._eval is not None:
            return self._eval

        self._eval = jax.jit(define_fdjvp(self._integral, self.grad_method))
        return self._eval

    @property
    @abstractmethod
    def _integral(self):
        pass


class XspecConvolutionModel(ConvolutionModel):
    def __call__(self, model: Model) -> XspecConvolvedModel:
        if model.type not in self._component._supported:
            accepted = [f"'{i}'" for i in self._component._supported]
            raise TypeError(
                f'{self.name} convolution model supports models with type: '
                f"{', '.join(accepted)}; got '{model.type}' type model {model}"
            )

        return XspecConvolvedModel(self._component, model)


class XspecConvolvedModel(ConvolvedModel):
    _op: XspecConvolution

    @property
    def eval(self) -> ModelEval:
        comp_id = self._op._id
        convolve = self._op.eval
        model = self._model.eval

        def fn(egrid: JAXArray, params: CompIDParamValMapping) -> JAXArray:
            """The convolved model evaluation function."""
            return convolve(egrid, params[comp_id], model(egrid, params))

        return jax.jit(fn)


class XspecConvolution(XspecComponent):
    _supported: frozenset[Literal['add', 'mul']]
    _convolve_jit = None
    _kwargs = (
        'low_energy',
        'low_ngrid',
        'low_log',
        'high_energy',
        'high_ngrid',
        'high_log',
        'grad_method',
        'spec_num',
    )

    def __init__(
        self,
        params: dict,
        latex: str | None,
        low_energy: float | int | None,
        low_ngrid: int | None,
        low_log: bool | None,
        high_energy: float | int | None,
        high_ngrid: int | None,
        high_log: bool | None,
        grad_method: Literal['central', 'forward'] | None,
        spec_num: int | None,
    ):
        self.grad_method = grad_method
        self._spec_num = spec_num

        if spec_num is None:
            spec_num = 1
        else:
            spec_num = int(spec_num)
        self._spec_num = spec_num

        if low_energy is None:
            low_energy = 0.01
        self._low_energy = float(low_energy)

        if low_ngrid is None:
            low_ngrid = 100
        self._low_ngrid = int(low_ngrid)

        if low_log is None:
            low_log = True
        self._low_log = bool(low_log)

        if high_energy is None:
            high_energy = 100.0
        self._high_energy = float(high_energy)

        if high_ngrid is None:
            high_ngrid = 100
        self._high_ngrid = int(high_ngrid)

        if high_log is None:
            high_log = True
        self._high_log = bool(high_log)

        super().__init__(params, latex, grad_method, spec_num)

    @property
    def type(self) -> Literal['conv']:
        return 'conv'

    @property
    def eval(self):
        if self._convolve_jit is not None:
            self._convolve_jit = jax.jit(self._convolve)
        return self._convolve_jit

    @property
    @abstractmethod
    def _convolve(self):
        pass


class cflux(XspecConvolution):
    _config = (ParamConfig('F', r'\mathrm{F}', 'keV', 1.0, 1e-10, 1e10),)

    @property
    def _convolve(self):
        spec_num = self.spec_num
        params_names = [p.name for p in self._config]
        fn = _XSMODEL['primitive']['cflux']

        def cflux(egrid, params, flux):
            params = jnp.stack([params[p] for p in params_names])
            return fn(params, egrid, flux, spec_num)

        return cflux


def create_xspec_components():
    """Create Xspec model classes."""
    if not _HAS_XSPEC:
        return {}

    template = '''
class {name}({component_class}):
    """Xspec {name} model."""

    _config = (
        {params_config},
    )

    @property
    def _integral(self):
        spec_num = self.spec_num
        params_names = [p.name for p in self._config]
        if self.type == 'add':
            params_names.remove('norm')
        fn = XSMODEL['primitive']['{name}']

        def {name}(egrid, params):
            params = jnp.stack([params[p] for p in params_names])
            return fn(params, egrid, spec_num)

        return {name}
'''
    env = {
        'ParamConfig': ParamConfig,
        'XSMODEL': _XSMODEL,
        'XspecAdditive': XspecAdditive,
        'XspecMultiplicative': XspecMultiplicative,
        'jnp': jnp,
    }
    models = _XSMODEL['add'] | _XSMODEL['mul']
    model_classes = {}
    for name, info in models.items():
        params_config = []
        for p in info.parameters:
            if p.units is None:
                unit = ''
            else:
                unit = p.units

            pname = p.name
            if pname == 'del':
                pname = 'delta'

            pmin = p.hardmin
            pmax = p.hardmax
            default = p.default
            if p.paramtype.name == 'Default' and (not pmin < default < pmax):
                delta = 1e-3 * (pmax - pmin)
                if pmin == default:
                    default += delta
                else:
                    default -= delta

            params_config.append(
                rf"ParamConfig('{pname}', r'\mathrm{{{pname}}}', '{unit}', "
                f'{default}, {pmin}, {pmax}, fixed={p.frozen})'
            )

        if info.modeltype.name == 'Add':
            component_class = 'XspecAdditive'
            params_config.append(
                r"ParamConfig('norm', r'\mathrm{{norm}}', '', "
                '1.0, 1e-10, 1e10)'
            )
        else:
            component_class = 'XspecMultiplicative'

        params_config = ',\n        '.join(params_config)

        str_map = {
            'name': name,
            'component_class': component_class,
            'params_config': params_config,
        }
        exec(template.format_map(str_map), env, model_classes)

    return model_classes


_xs_comps = create_xspec_components()
locals().update(_xs_comps)
__all__.append(_xs_comps.keys())
del _xs_comps
