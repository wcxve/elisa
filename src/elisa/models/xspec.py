"""Xspec model library API."""

from __future__ import annotations

import os
import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
from bs4 import BeautifulSoup

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
        ModelType as _XspecModelType,
        abundance,
        chatter,
        clear_db,
        clear_model_string,
        clear_xflt,
        cosmology,
        cross_section,
        element_abundance,
        element_name,
        get_db,
        get_model_string,
        get_number_xflt,
        get_xflt,
        in_xflt,
        number_elements,
        set_db,
        set_model_string,
        set_xflt,
        version,
    )
    from xspex.primitive import XSModel as _XSMODEL

    __all__ = [
        'abundance',
        'chatter',
        'clear_db',
        'clear_model_string',
        'clear_xflt',
        'cosmology',
        'cross_section',
        'element_abundance',
        'element_name',
        'get_db',
        'get_model_string',
        'get_number_xflt',
        'get_xflt',
        'in_xflt',
        'number_elements',
        'set_db',
        'set_model_string',
        'set_xflt',
        'version',
    ]
    _HAS_XSPEC = True
except ImportError as e:
    __all__ = []
    _XSMODEL = {}
    _HAS_XSPEC = False
    warnings.warn(f'Xspec model library is not available: {e}', ImportWarning)

if TYPE_CHECKING:
    from xspex import (
        XspecModel as XspecModelInfo,
        XspecParameter as XspecParameterInfo,
    )

    from elisa.models.model import Model, UniComponentModel
    from elisa.util.typing import (
        Callable,
        CompEval,
        CompIDParamValMapping,
        JAXArray,
        ModelEval,
        ParamNameValMapping,
    )

    XspecConvolveEval = Callable[
        [JAXArray, ParamNameValMapping, JAXArray],
        JAXArray,
    ]


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
        model = self._model.eval
        comp_id = self._op._id
        convolve = self._op.eval
        elow = self._op._low_energy
        nlow = self._op._low_ngrid
        loglow = self._op._low_log
        ehigh = self._op._high_energy
        nhigh = self._op._high_ngrid
        loghigh = self._op._high_log

        def extend_low(egrid):
            if egrid[0] < elow:
                raise RuntimeError(
                    f'for Xspec convolution model {self}, the lower limit '
                    f'of the energy extension ({elow}) must be less than '
                    f'the minimum energy grid ({egrid[0]})'
                )
            if loglow:
                low_extension = np.geomspace(elow, egrid[0], nlow + 1)[:-1]
            else:
                low_extension = np.linspace(elow, egrid[0], nlow + 1)[:-1]
            return np.concatenate((low_extension, egrid)).astype(egrid.dtype)

        def extend_high(egrid):
            if egrid[-1] > ehigh:
                raise RuntimeError(
                    f'for Xspec convolution model {self}, the upper limit '
                    f'of the energy extension ({ehigh}) must be greater than '
                    f'the maximum energy grid ({egrid[-1]})'
                )
            if loghigh:
                high_extension = np.geomspace(egrid[-1], ehigh, nhigh + 1)[1:]
            else:
                high_extension = np.linspace(egrid[-1], ehigh, nhigh + 1)[1:]
            return np.concatenate((egrid, high_extension)).astype(egrid.dtype)

        def fn(egrid: JAXArray, params: CompIDParamValMapping) -> JAXArray:
            """The convolved model evaluation function."""
            rtype = jax.ShapeDtypeStruct((egrid.size + nlow,), egrid.dtype)
            egrid = jax.pure_callback(extend_low, rtype, egrid)
            rtype = jax.ShapeDtypeStruct((egrid.size + nhigh,), egrid.dtype)
            egrid = jax.pure_callback(extend_high, rtype, egrid)
            conv_params = params[comp_id]
            flux = model(egrid, params)
            result = convolve(egrid, conv_params, flux)
            return result[nlow:-nhigh]

        fn = define_fdjvp(jax.jit(fn), self._op.grad_method)
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
    def eval(self) -> XspecConvolveEval:
        if self._convolve_jit is None:
            self._convolve_jit = jax.jit(self._convolve)
        return self._convolve_jit

    @property
    @abstractmethod
    def _convolve(self) -> XspecConvolveEval:
        pass


_TEMPLATE_ADD = '''
class XS{name}(XspecAdditive):
    """Xspec additive model `{name} <{link}>`_: {desc}."""

    _config = (
        {params_config},
    )

    @property
    def _integral(self):
        spec_num = self.spec_num
        params_names = [p.name for p in self._config if p.name != 'norm']

        def integral(egrid, params):
            params = [params[p] for p in params_names]
            if params:
                params = jnp.stack(params)
            else:
                params = jnp.empty(0)
            return {name}(params, egrid, spec_num)

        return integral
'''

_TEMPLATE_MUL = '''
class XS{name}(XspecMultiplicative):
    """Xspec multiplicative model `{name} <{link}>`_: {desc}."""

    _config = (
        {params_config},
    )

    @property
    def _integral(self):
        spec_num = self.spec_num
        params_names = [p.name for p in self._config]

        def integral(egrid, params):
            params = [params[p] for p in params_names]
            if params:
                params = jnp.stack(params)
            else:
                params = jnp.empty(0)
            return {name}(params, egrid, spec_num)

        return integral
'''

_TEMPLATE_CON = '''
class XS{name}(XspecConvolution):
    """Xspec convolution model `{name} <{link}>`_: {desc}."""

    _supported = frozenset(['{supported}'])
    _config = (
        {params_config},
    )

    @property
    def _convolve(self):
        spec_num = self.spec_num
        params_names = [p.name for p in self._config]

        def convolve(egrid, params, flux):
            params = [params[p] for p in params_names]
            if params:
                params = jnp.stack(params)
            else:
                params = jnp.empty(0)
            return {name}(params, egrid, flux, spec_num)

        return convolve
'''

_PARAM_CONFIG_TEMPLATE = (
    r'ParamConfig(name="{name}", latex=r"\mathrm{{{name}}}", unit="{unit}", '
    'default={default}, min={min}, max={max}, fixed={fixed})'
)


def generate_xspec_models():
    """Generate Xspec model classes."""
    models = {}

    if not _HAS_XSPEC:
        return models

    # Models' short description and URL link to official manual
    model_desc_link = xspec_model_desc_and_link()

    # Convolution models that should be applied for multiplicative models
    conv_for_mul = ('partcov', 'vmshift', 'zmshift')

    # For compile model function
    env = {
        'ParamConfig': ParamConfig,
        'XspecAdditive': XspecAdditive,
        'XspecMultiplicative': XspecMultiplicative,
        'XspecConvolution': XspecConvolution,
        'jnp': jnp,
    }

    # For additive models
    norm_config = _PARAM_CONFIG_TEMPLATE.format(
        name='norm',
        latex=r'\mathrm{{norm}}',
        unit='',
        default=1.0,
        min=1e-10,
        max=1e10,
        fixed=False,
    )

    def gen_param_config(param_info: XspecParameterInfo) -> str:
        """Generate parameter configuration string given parameter info."""
        name = param_info.name
        default = param_info.default
        unit = param_info.units or ''
        pmin = param_info.hardmin
        pmax = param_info.hardmax

        # avoid conflict to del in Python
        if name == 'del':
            name = 'delta'

        # min and max is None for switch and scale type parameters
        if pmin is None:
            pmin = -1
        if pmax is None:
            pmax = 1e10

        return _PARAM_CONFIG_TEMPLATE.format(
            name=name,
            unit=unit,
            default=default,
            min=pmin,
            max=pmax,
            fixed=param_info.frozen,
        )

    def make_xspec_model(model_info: XspecModelInfo) -> XspecComponentMeta:
        """Make Xspec model class."""
        name = model_info.name
        vars_map = {
            'name': name,
            'desc': model_desc_link[name]['desc'],
            'link': model_desc_link[name]['link'],
        }
        params_config = list(map(gen_param_config, model_info.parameters))
        model_type = model_info.modeltype
        if model_type == _XspecModelType.Add:
            params_config.append(norm_config)
            template = _TEMPLATE_ADD
        elif model_type == _XspecModelType.Mul:
            template = _TEMPLATE_MUL
        elif model_type == _XspecModelType.Con:
            vars_map['supported'] = 'mul' if name in conv_for_mul else 'add'
            template = _TEMPLATE_CON
        else:
            raise ValueError(f'Unsupported model type: {model_type.name}')
        vars_map['params_config'] = ',\n        '.join(params_config)
        model_code = template.format_map(vars_map)
        primitive = _XSMODEL['primitive'][name]
        exec(model_code, env | {name: primitive}, models)

    list(map(make_xspec_model, _XSMODEL['add'].values()))
    list(map(make_xspec_model, _XSMODEL['mul'].values()))
    list(map(make_xspec_model, _XSMODEL['con'].values()))

    return models


def xspec_model_desc_and_link():
    HEADAS = os.environ.get('HEADAS', '')
    model_info = {}

    if not HEADAS:
        return model_info

    spectral_path = os.path.abspath(f'{HEADAS}/../spectral')
    if not os.path.exists(spectral_path):
        spectral_path = os.path.abspath(f'{HEADAS}/spectral')
        if not os.path.exists(spectral_path):
            spectral_path = ''

    if not spectral_path:
        return model_info

    url = 'https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSmodel{}.html'
    html_path = f'{spectral_path}/help/html'
    for mtype in ['Additive', 'Multiplicative', 'Convolution']:
        with open(f'{html_path}/{mtype}.html', encoding='utf-8') as f:
            s = BeautifulSoup(f.read(), 'html.parser')

        for a in s.find_all('ul', class_='ChildLinks')[0].find_all('a'):
            text = a.text

            # there is no ':' in agnslim model desc
            if mtype == 'Additive' and text.startswith('agnslim, AGN'):
                text = text.replace('agnslim, AGN', 'agnslim: AGN')

            models, desc = text.split(':')
            models = [m.strip() for m in models.split(',')]
            desc = desc.strip()
            link = url.format(models[0].title())
            for m in models:
                model_info[m.lower()] = {'desc': desc, 'link': link}

    # There are some typos in the model name
    if 'bvvcie' not in model_info and 'bbvcie' in model_info:
        model_info['bvvcie'] = model_info.pop('bbvcie')
    if 'bvwdem' not in model_info and 'bwwdem' in model_info:
        model_info['bvwdem'] = model_info.pop('bwwdem')

    return model_info


_xs_comps = generate_xspec_models()
for c in _xs_comps.values():
    c.__module__ = __name__
locals().update(_xs_comps)
__all__.extend(_xs_comps.keys())
del _xs_comps
