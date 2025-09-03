"""XSPEC model library API."""

from __future__ import annotations

import keyword
import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from elisa.models.model import (
    Component,
    ComponentMeta,
    ConvolutionModel,
    ConvolvedModel,
    ParamConfig,
)
from elisa.util.misc import define_fdjvp

try:
    import xspex as _xx
    from xspex import (
        abund,
        abund_file,
        chatter,
        clear_mstr,
        clear_xflt,
        cosmo,
        list_models,
        mstr,
        xflt,
        xsect,
        xspec_version,
    )
    from xspex._xspec.types import (
        XspecModelType as _XspecModelType,
    )

    __all__ = [
        'xspec_version',
        'list_models',
        'abund',
        'abund_file',
        'xsect',
        'cosmo',
        'mstr',
        'clear_mstr',
        'xflt',
        'clear_xflt',
        'chatter',
        *_xx.list_models(),
    ]
    _HAS_XSPEC = True
except ImportError as e:
    __all__ = []
    _HAS_XSPEC = False
    warnings.warn(f'XSPEC model library is not available: {e}', ImportWarning)

if TYPE_CHECKING:
    from typing import Literal

    from xspex._xspec.types import (
        XspecParam as XspecParamInfo,
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
            value = 'central'

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
            return params.pop('norm') * _integral(egrid, params)

        self._eval = jax.jit(integral)

        return self._eval

    @property
    @abstractmethod
    def _integral(self) -> CompEval:
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
    def _integral(self) -> CompEval:
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


_XSPEC_MODEL_TEMPLATE_ADD = '''
class {name}(XspecAdditive):
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

_XSPEC_MODEL_TEMPLATE_MUL = '''
class {name}(XspecMultiplicative):
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

_XSPEC_MODEL_TEMPLATE_CON = '''
class {name}(XspecConvolution):
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

_XSPEC_MODEL_PARAM_CONFIG_TEMPLATE = (
    r'ParamConfig(name="{name}", latex=r"\mathrm{{{name}}}", unit="{unit}", '
    'default={default}, min={min}, max={max}, fixed={fixed})'
)


def _generate_xspec_models():
    """Generate Xspec model classes."""
    models = {}

    if not _HAS_XSPEC:
        return models

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
    norm_config = _XSPEC_MODEL_PARAM_CONFIG_TEMPLATE.format(
        name='norm',
        latex=r'\mathrm{{norm}}',
        unit='',
        default=1.0,
        min=1e-10,
        max=1e10,
        fixed=False,
    )

    def gen_param_config(param_info: XspecParamInfo) -> str:
        """Generate parameter configuration string given parameter info."""
        name = param_info.name
        default = param_info.default
        unit = param_info.unit or ''
        pmin = param_info.min
        pmax = param_info.max

        # smaug model's params names have dots
        name = name.replace('.', '_')

        # avoid keyword conflict
        if keyword.iskeyword(name):
            name = f'{name}_'

        # min and max is None for switch and scale type parameters
        if pmin is None:
            pmin = -1
        if pmax is None:
            pmax = 1e10

        return _XSPEC_MODEL_PARAM_CONFIG_TEMPLATE.format(
            name=name,
            unit=unit,
            default=default,
            min=pmin,
            max=pmax,
            fixed=param_info.fixed,
        )

    def make_xspec_model(name: str):
        """Make Xspec model class."""
        fn, model_info = _xx.get_model(name)
        vars_map = {
            'name': name,
            'desc': model_info.desc,
            'link': model_info.link,
        }
        params_config = list(map(gen_param_config, model_info.parameters))
        model_type = model_info.type
        if model_type == _XspecModelType.Add:
            params_config.append(norm_config)
            template = _XSPEC_MODEL_TEMPLATE_ADD
        elif model_type == _XspecModelType.Mul:
            template = _XSPEC_MODEL_TEMPLATE_MUL
        elif model_type == _XspecModelType.Con:
            vars_map['supported'] = 'mul' if name in conv_for_mul else 'add'
            template = _XSPEC_MODEL_TEMPLATE_CON
        else:
            raise ValueError(f'Unsupported model type: {model_type.name}')
        vars_map['params_config'] = ',\n        '.join(params_config)
        model_code = template.format_map(vars_map)
        exec(model_code, env | {name: fn}, models)

    list(map(make_xspec_model, _xx.list_models()))

    for m in models.values():
        m.__module__ = __name__

    return models


locals().update(_generate_xspec_models())
