"""Xspec model wrappers."""

from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp

from elisa.models.model import Component, ParamConfig
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
    from elisa.util.typing import CompEval

__all__ = []


class XspecComponent(Component):
    """Xspec model wrapper."""

    _kwargs = ('grad_method', 'spec_num')
    _eval: CompEval | None = None

    def __init__(
        self,
        params: dict,
        latex: str | None,
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


class XspecConvolution(XspecComponent):
    @property
    def type(self) -> Literal['conv']:
        return 'conv'


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
