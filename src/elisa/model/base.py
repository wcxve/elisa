"""Classes to handle model construction."""
from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from typing import Callable, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
from numpyro.distributions import Distribution, LogUniform, Uniform

from .node import (
    LabelSpace,
    ModelNode,
    ModelOperationNode,
    ParameterNode,
    ParameterOperationNode,
)

ModelNodeType = Union[ModelNode, ModelOperationNode]
ParameterNodeType = Union[ParameterNode, ParameterOperationNode]

__all__ = ['UniformParameter', 'generate_parameter', 'generate_model'][:1]

# TODO: time dependent model


class Parameter:
    """Prototype parameter class.

    Parameters
    ----------
    node : ParameterNode, or ParameterOperationNode
        The base parameter node.

    """

    def __init__(self, node: ParameterNodeType):
        if not isinstance(node, (ParameterNode, ParameterOperationNode)):
            raise ValueError(
                "node must be ParameterNode or ParameterOperationNode"
            )

        self._node = node

    def __repr__(self):
        return self._node.attrs['name']

    def __add__(self, other: Parameter) -> SuperParameter:
        return SuperParameter(self, other, '+')

    def __mul__(self, other: Parameter) -> SuperParameter:
        return SuperParameter(self, other, '*')

    @property
    def default(self) -> float:
        """Parameter default value."""
        return self._node.default

    @default.setter
    def default(self, value: float):
        """Parameter default value."""
        self._node.default = value


class SuperParameter(Parameter):
    """Class to handle operation on parameters.

    Parameters
    ----------
    lh: Parameter
        Left-hand parameter.
    rh: Parameter
        Right-hand parameter.
    op : {'+', '*'}
        Operation to perform.

    """

    def __init__(self, lh: Parameter, rh: Parameter, op: str):
        if op not in {'+', '*'}:
            raise TypeError(f'operator "{op}" is not supported')

        if not isinstance(lh, Parameter):
            raise TypeError(f'{lh} is not a valid parameter')

        if not isinstance(rh, Parameter):
            raise TypeError(f'{rh} is not a valid parameter')

        self._op = op
        self._lh = lh
        self._rh = rh

        if op == '+':
            node = lh._node + rh._node
        else:  # op == '*'
            node = lh._node * rh._node

        super().__init__(node)

    def __repr__(self):
        if isinstance(self._lh, UniformParameter):
            lh = self._lh._node.attrs['name']
        else:
            lh = repr(self._lh)

        if isinstance(self._rh, UniformParameter):
            rh = self._rh._node.attrs['name']
        else:
            rh = repr(self._rh)

        op = self._op

        if op == '*':
            if getattr(self._lh, '_op', '') == '+':
                lh = f'({lh})'

            if getattr(self._rh, '_op', '') == '+':
                rh = f'({rh})'

        return f'{lh} {op} {rh}'

    @property
    def default(self) -> float:
        """Parameter default value, get only."""
        return self._node.default


class UniformParameter(Parameter):
    """The default class to handle parameter definition.

    Parameters
    ----------
    name : str
        Parameter name.
    fmt : str
        Parameter Tex.
    default : float
        Parameter default value.
    min : float
        Parameter minimum value.
    max : float
        Parameter maximum value.
    frozen : bool
        Whether parameter is fixed.
    log : bool
        Whether parameter is parameterized into log scale, only for positive
        valued parameter.

    """

    def __init__(
        self,
        name: str,
        fmt: str,
        default: float,
        min: float,
        max: float,
        frozen: bool = False,
        log: bool = False
    ):
        self._config = {
            'default': float(default),
            'min': float(min),
            'max': float(max),
            'frozen': bool(frozen),
            'log': bool(log)
        }

        self._check_and_set_values()

        node = ParameterNode(
            name=name,
            fmt=fmt,
            default=default,
            distribution=self._get_distribution(),
            min=self._config['min'],
            max=self._config['max'],
            dist_expr=self.get_expression()
        )
        super().__init__(node)

    def __repr__(self):
        name = self._node.attrs['name']
        expr = self.get_expression()

        if self._config['frozen']:
            s = f'{name} = {expr}'
        else:
            s = f'{name} ~ {expr}'

        return s

    @property
    def fmt(self) -> str:
        """Parameter Tex."""
        return self._node.attrs['fmt']

    @fmt.setter
    def fmt(self, value):
        """Parameter Tex."""
        self._node.attrs['fmt'] = str(value)

    @property
    def default(self) -> float:
        """Parameter default value."""
        return self._config['default']

    @default.setter
    def default(self, value):
        """Parameter default value."""
        self._check_and_set_values(default=value)

    @property
    def min(self) -> float:
        """Parameter minimum."""
        return self._config['min']

    @min.setter
    def min(self, value):
        """Parameter minimum."""
        self._check_and_set_values(min=value)

    @property
    def max(self) -> float:
        """Parameter maximum."""
        return self._config['max']

    @max.setter
    def max(self, value):
        """Parameter maximum."""
        self._check_and_set_values(max=value)

    @property
    def frozen(self) -> bool:
        """Parameter frozen status."""
        return self._config['frozen']

    @frozen.setter
    def frozen(self, value):
        """Parameter frozen status."""
        flag = bool(value)

        if self._config['frozen'] != flag:
            self._config['frozen'] = flag
            self._reset_distribution()

    @property
    def log(self) -> bool:
        """Whether parameter is in log scale."""
        return self._config['log']

    @log.setter
    def log(self, value):
        """Whether parameter is in log scale."""
        log_flag = bool(value)

        config = self._config
        if config['log'] != log_flag:
            if log_flag and config['min'] <= 0.0:
                raise ValueError(
                    'parameterization into log-uniform failed due to '
                    'non-positive minimum'
                )

            config['log'] = log_flag

            self._reset_distribution()

    def _check_and_set_values(self, default=None, min=None, max=None) -> None:
        """Check and set parameter configuration."""
        config = self._config

        if default is None:
            _default = config['default']
        else:
            _default = float(default)

        if min is None:
            _min = config['min']
        else:
            _min = float(min)

        if max is None:
            _max = config['max']
        else:
            _max = float(max)

        if _min <= 0.0 and config['log']:
            raise ValueError(
                f'min ({_min}) must be positive for LogUniform'
            )

        if _min > _max:
            raise ValueError(
                f'min ({_min}) must not larger than max ({_max})'
            )

        if _default < _min:
            raise ValueError(
                f'default value ({_default}) is smaller than min ({_min})'
            )

        if _default > _max:
            raise ValueError(
                f'default value ({_default}) is larger than max ({_max})'
            )

        if default is not None:
            config['default'] = float(default)
            self._node.attrs['default'] = float(default)

        if min is not None:
            config['min'] = float(min)

        if max is not None:
            config['max'] = float(max)

        if (min is not None) or (max is not None):
            self._reset_distribution()

    def _get_distribution(self) -> Distribution | float:
        """Get distribution for :class:`ParameterNode`."""
        config = self._config

        if config['frozen']:
            distribution = config['default']
        else:
            if config['log']:
                distribution = LogUniform(config['min'], config['max'])
            else:
                distribution = Uniform(config['min'], config['max'])

        return distribution

    def get_expression(self) -> str:
        """Get expression of distribution."""
        default = self._config["default"]

        if self._config['frozen']:
            expr = str(default)
        else:
            min = f'{self._config["min"]:.4g}'
            max = f'{self._config["max"]:.4g}'

            if self._config['log']:
                # expr = f'LogUniform(min={min}, max={max}, default={default})'
                expr = f'LogUniform(min={min}, max={max})'
            else:
                # expr = f'Uniform(min={min}, max={max}, default={default})'
                expr = f'Uniform(min={min}, max={max})'

        return expr

    def _reset_distribution(self) -> None:
        """Reset distribution after configuring parameter."""
        self._node.attrs['distribution'] = self._get_distribution()
        self._node.attrs['dist_expr'] = self.get_expression()


class ModelParameterFormat(dict):
    """Class to restore Tex format of model parameters."""

    def __init__(self, fmt_dict: dict) -> None:
        super().__init__()
        for k, v in fmt_dict.items():
            super().__setitem__(k, v)
        self._keys = tuple(fmt_dict.keys())

    def __setitem__(self, key: str, value: str) -> None:
        if key in self._keys:
            super().__setitem__(key, str(value))
        else:
            raise KeyError(key)


class Model:
    """Prototype model class.

    Parameters
    ----------
    node : ModelNode, or ModelOperationNode
        The base model node.
    params : dict, or None
        A :class:`str`-:class:`Parameter` mapping that contains the parameters
        of the model. It should be None when initializing :class:`SuperModel`.
    params_fmt : dict, or None

    """

    def __init__(
        self,
        node: ModelNodeType,
        params: Optional[dict[str, Parameter]] = None,
        params_fmt: Optional[dict[str, str]] = None
    ):
        if not isinstance(node, (ModelNode, ModelOperationNode)):
            raise ValueError(
                'node must be ModelNode or ModelOperationNode'
            )

        if isinstance(node, ModelNode) and not (
            isinstance(params, dict)
            and all(isinstance(k, str) for k in params.keys())
            and all(isinstance(v, Parameter) for v in params.values())
        ):
            raise ValueError(
                'params must be dict of str-Parameter mapping for model'
            )

        self._node = node
        self._label = LabelSpace(node)
        self._fn = None
        self._sub_comp = None
        self._comp_fn = None

        if params is not None:
            for key, value in params.items():
                setattr(self, key, value)

            self._comps = {node.name: self}  # use name_id to mark model
            self._params = params
            self._params_name = tuple(params.keys())

            if params_fmt is not None:
                if set(self._params_name) != set(params_fmt.keys()):
                    raise ValueError('`params_fmt` must match `params`')
            else:
                params_fmt = {i: r'\mathrm{%s}' % i for i in params.keys()}

            self._params_fmt = {
                node.name: ModelParameterFormat(params_fmt)
            }

    def __repr__(self):
        return self._label.name

    def __add__(self, other: Model) -> SuperModel:
        return SuperModel(self, other, '+')

    def __mul__(self, other: Model) -> SuperModel:
        return SuperModel(self, other, '*')

    def __setattr__(self, key, value):
        if hasattr(self, '_params_name') \
                and self._params_name is not None \
                and key in self._params_name:
            self._set_param(key, value)

        super().__setattr__(key, value)

    def __getitem__(self, name: str) -> Parameter:
        if name not in self._params:
            raise ValueError(f'{self} has no "{name}" parameter')

        return self._params[name]

    def __setitem__(self, name: str, param: Parameter):
        if name not in self._params:
            raise ValueError(f'{self} has no "{name}" parameter')

        setattr(self, name, param)

    @property
    def type(self) -> str:
        """Model type."""
        return self._node.attrs['mtype']

    @property
    def params_fmt(self) -> ModelParameterFormat:
        """Model parameter format configuration."""
        return self._params_fmt[self._node.name]

    def ne(
        self,
        egrid: jax.Array,
        params: dict[str, dict[str, float | jax.Array]],
        comps: bool = False
    ):
        r"""Calculate :math:`N_E` over `egrid`.

        Parameters
        ----------
        egrid : array_like
            Energy grid over which to calculate the :math:`N_E`.
        params : dict
            Parameter dict for the spectral model.
        comps : bool, optional
            Whether to return the result of each model component, instead of
            summing them up. The default is False.

        Returns
        -------
        N_E : jax.Array or dict[str, jax.Array]
            The :math:`N_E` over `egrid`, in unit of cm^-2 s^-1 keV^-1.

        """
        shapes = jax.tree_util.tree_flatten(
            tree=jax.tree_map(jnp.shape, params),
            is_leaf=lambda i: isinstance(i, tuple)
        )[0]

        if not shapes:
            print(shapes)
            raise ValueError('empty params')

        shape = shapes[0]
        if not all(shape == s for s in shapes[1:]):
            print(shapes)
            raise ValueError('all params must have the same shape')

        de = jnp.diff(egrid)

        if shape == ():
            eval_fn = lambda f: f(egrid, params) / de

        elif len(shape) == 1:
            eval_fn = lambda f: \
                jax.vmap(f, in_axes=(None, 0))(egrid, params) / de

        elif len(shape) == 2:
            eval_fn = lambda f: \
                jax.vmap(
                    jax.vmap(f, in_axes=(None, 0)),
                    in_axes=(None, 0)
                )(egrid, params) / de

        else:
            raise ValueError(f'params ndim should <= 2, got {len(shape)}')

        if not comps:
            return eval_fn(self._wrapped_fn)
        else:
            return jax.tree_map(eval_fn, self._wrapped_comp_fn)

    def ene(
        self,
        egrid: jax.Array,
        params: dict[str, dict[str, float | jax.Array]],
        comps: bool = False
    ):
        r"""Calculate :math:`E N_E` (:math:`F_\nu`) over `egrid`.

        Parameters
        ----------
        egrid : array_like
            Energy grid over which to calculate the :math:`E N_E`.
        params : dict
            Parameter dict for the spectral model.
        comps : bool, optional
            Whether to return the result of each model component, instead of
            summing them up. The default is False.

        Returns
        -------
        EN_E : jax.Array or dict[str, jax.Array]
            The :math:`E N_E` over `egrid`, in unit of erg cm^-2 s^-1 keV^-1.

        """
        ne = self.ne(egrid, params, comps)
        emid = jnp.sqrt(egrid[:-1] * egrid[1:])
        fn = lambda x: emid * x * 1.602176634e-9
        if comps:
            return jax.tree_map(fn, ne)
        else:
            return fn(ne)

    def eene(
        self,
        egrid: jax.Array,
        params: dict[str, dict[str, float | jax.Array]],
        comps: bool = False
    ):
        r"""Calculate :math:`E^2 N_E` (:math:`\nu F_\nu`) over `egrid`.

        Parameters
        ----------
        egrid : array_like
            Energy grid over which to calculate the :math:`E^2 N_E`.
        params : dict
            Parameter dict for the spectral model.
        comps : bool, optional
            Whether to return the result of each model component, instead of
            summing them up. The default is False.

        Returns
        -------
        EEN_E : jax.Array or dict[str, jax.Array]
            The :math:`E^2 N_E` over `egrid`, in unit of erg cm^-2 s^-1.

        """
        ne = self.ne(egrid, params, comps)
        e2 = egrid[:-1] * egrid[1:]
        fn = lambda x: e2 * x * 1.602176634e-9
        if comps:
            return jax.tree_map(fn, ne)
        else:
            return fn(ne)

    def folded(
        self,
        ph_egrid: jax.Array,
        params: dict[str, dict[str, float | jax.Array]],
        resp_matrix: jax.Array,
        ch_width: jax.Array,
        comps: bool = False
    ):
        """Calculate the folded spectral model (:math:`C_E`).

        Parameters
        ----------
        params : dict
            Parameter dict for the spectral model.
        resp_matrix : ndarray
            Instrumental response matrix used to fold the spectral model.
        ph_egrid : ndarray
            Photon energy grid of the `resp_matrix`.
        ch_width : ndarray
            Measured energy channel grid width of the `resp_matrix`.
        comps : bool, optional
            Whether to return the result of each model component, instead of
            summing them up. The default is False.

        Returns
        -------
        C_E : jax.Array or dict[str, jax.Array]
            The folded spectral model :math:`C_E`, in unit of s^-1 keV^-1.

        """
        ne = self.ne(ph_egrid, params, comps)
        de = jnp.diff(ph_egrid)
        fn = lambda x: (ne * de) @ resp_matrix / ch_width
        if comps:
            return jax.tree_map(fn, ne)
        else:
            return fn(ne)

    def flux(
        self,
        params: dict[str, dict[str, float | jax.Array]],
        emin: float | int,
        emax: float | int,
        energy: bool = True,
        comps: bool = False,
        ngrid: int = 1000,
        elog: bool = True
    ):
        """Calculate flux of the model between `emin` and `emax`.

        Parameters
        ----------
        params : dict
            Parameter dict for the spectral model.
        emin : float or int
            Minimum energy of the energy range to calculate the flux.
        emax : float or int
            Maximum energy of the energy range to calculate the flux.
        energy : bool, optional
            Whether to calculate energy flux of the model. Calculate photon
            flux when False. The default is True.
        comps : bool, optional
            Whether to return the result of each model component, instead of
            summing them up. The default is False.
        ngrid : int, optional
            The energy grid number to create. The default is 1000.
        elog : bool, optional
            Whether to use regular energy grid in log scale. The default is
            True.

        Returns
        -------
        flux : jax.Array or dict[str, jax.Array]
            The flux of the model, in unit of s^-1 keV^-1.

        """
        if self.type != 'add':
            msg = f'flux is undefined for {self.type} type model "{self}"'
            raise TypeError(msg)

        if elog:
            egrid = jnp.geomspace(emin, emax, ngrid)
        else:
            egrid = jnp.linspace(emin, emax, ngrid)

        if energy:
            f = self.ene(egrid, params, comps)
        else:
            f = self.ne(egrid, params, comps)

        delta = jnp.diff(egrid)
        fn = lambda x: jnp.sum(x * delta, axis=-1)

        if comps:
            return jax.tree_map(fn, f)
        else:
            return fn(f)

    def _set_param(self, name: str, param: Parameter) -> None:
        """Set parameter.

        Parameters
        ----------
        name : str
            Parameter name.
        param : Parameter
            Parameter instance.

        """
        if not isinstance(self._node, ModelNode):
            raise TypeError('SuperModel does not support setting parameter')

        type_error = False

        if isinstance(param, Parameter):
            self._params[name] = param
            idx = self._params_name.index(name)
            self._node.predecessor[idx] = param._node

        else:
            if isinstance(param, (float, int)):
                p = self._params[name]
                if isinstance(p, UniformParameter):
                    p.frozen = True
                    p.default = param
                else:
                    type_error = True
            else:
                type_error = True

        if type_error:  # other input types are not supported yet
            type_name = type(param).__name__
            raise TypeError(
                f'got {repr(param)} ({type_name}) for '
                f'{self}.{name}, which is not supported'
            )

    # def set_params(self, params: dict[str, Parameter]) -> None:
    #     """Set multiple parameters."""
    #
    #     params_in = set(params.keys())
    #     params_all = set(self._params_name)
    #
    #     if not params_in <= params_all:
    #         diff = params_all - params_in
    #         raise ValueError(f'input params {diff} not included in {self}')
    #
    #     if not all(isinstance(v, Parameter) for v in params.values())):
    #         raise ValueError('Parameter type is required')
    #
    #     params = self._params_names | params
    #     params_node = list(map(lambda v: v._node, params.values()))
    #     self._params_names = params
    #     self._node.predecessor = params_node

    def _fn_wrapper(self, name_mapping: dict) -> Callable:
        """Wrap function given model name mapping"""
        f = self._node.generate_func(name_mapping)

        # TODO: eliminate *args and **kwargs
        def fn(egrid, params, *args, **kwargs):
            """Model function wrapper."""
            return f(params, egrid, *args, **kwargs)

        return fn

    @property
    def _wrapped_fn(self) -> Callable:
        """Model evaluation function."""
        if self._fn is None:
            self._fn = jax.jit(self._fn_wrapper(self._label.mapping['name']))

        return self._fn

    def _get_comp(self) -> tuple[Model, ...]:
        """Get subcomponents."""
        if self._sub_comp is None:
            self._sub_comp = (self,)

        return self._sub_comp

    @property
    def _wrapped_comp_fn(self) -> dict[str, Callable]:
        """Sub-components evaluation function."""
        if self._comp_fn is None:
            mapping = self._label.mapping
            self._comp_fn = {
                m._label._label('name', mapping):
                jax.jit(m._fn_wrapper(mapping['name']))
                for m in self._get_comp()
            }

        return self._comp_fn

    @property
    def _model_info(self) -> dict:
        """Model information."""
        site = self._node.site
        mapping = self._label.mapping
        info = dict(
            sample=site['sample'],
            composite=site['composite'],
            pname=site['name'],
            pfmt=site['fmt'],
            default=site['default'],
            min=site['min'],
            max=site['max'],
            dist_expr=site['dist_expr'],
            params=self._node.params,
            mname=mapping['name'],
            mfmt=mapping['fmt'],
            mpfmt=self._params_fmt
        )
        return info


class SuperModel(Model):
    """Class to handle operation on models.

    Parameters
    ----------
    lh: Model
        Left-hand parameter.
    rh: Model
        Right-hand parameter.
    op : {'+', '*'}
        Operation to perform.

    """

    def __init__(self, lh: Model, rh: Model, op: str):
        if op not in {'+', '*'}:
            raise TypeError(f'operator "{op}" is not supported')

        if not isinstance(lh, Model):
            raise TypeError(f'{lh} is not a valid model')

        if not isinstance(rh, Model):
            raise TypeError(f'{rh} is not a valid model')

        self._op = op
        self._lh = lh
        self._rh = rh

        if op == '+':
            node = lh._node + rh._node
        else:  # op == '*'
            node = lh._node * rh._node

        super().__init__(node)

        comps = lh._comps | rh._comps
        names = self._label.mapping['name']
        _comps = {}
        _comps_name = []
        for k in names:
            setattr(self, names[k], comps[k])
            _comps[k] = comps[k]
            _comps_name.append(names[k])

        self._comps = _comps
        self._comps_name = tuple(_comps_name)
        self._params_fmt = lh._params_fmt | rh._params_fmt

    def __setattr__(self, key, value):
        if hasattr(self, '_comps_name') and key in self._comps_name:
            raise AttributeError(f"can't set attribute '{key}'")

        super().__setattr__(key, value)

    def __getitem__(self, name: str) -> Model:
        if name not in self._comps_name:
            raise ValueError(
                f'{self} has no "{name}" component'
            )

        return getattr(self, name)

    def __setitem__(self, name, value):
        raise TypeError('item assignment not supported')

    @property
    def params_fmt(self) -> dict[str, ModelParameterFormat]:
        """Model parameter format."""
        return {
            i: j._params_fmt[j._node.name]
            for i, j in zip(self._comps_name, self._comps.values())
        }

    def _get_comp(self) -> tuple[Model, ...]:
        """Get subcomponents."""
        if self._sub_comp is None:
            if self._op == '+':  # add + add
                subs = self._lh._get_comp() + self._rh._get_comp()
            elif self._lh.type == 'add':  # add * mul
                rh = self._rh
                subs = tuple(lh_i * rh for lh_i in self._lh._get_comp())
            elif self._rh.type == 'add':  # mul * add, con * add
                lh = self._lh
                if lh.type == 'mul':  # mul * add
                    subs = tuple(lh * rh_i for rh_i in self._rh._get_comp())
                else:  # con * add, note that con model isn't a linear operator
                    subs = (self,)
            else:  # mul * mul, con * mul, mul * con, con * con
                subs = (self,)

            self._sub_comp = subs

        return self._sub_comp


class ComponentMeta(ABCMeta):
    """Metaclass to avoid cumbersome code for ``__init__`` of subclass."""

    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # if subclass has ``config`` defined correctly, override its __init__
        if hasattr(cls, '_config') and isinstance(cls._config, tuple):
            name = cls.__name__.lower()

            # >>> construct __init__ function >>>
            config = cls._config
            init_def = 'self, '
            init_body = ''
            par_body = '('
            for cfg in config:
                init_def += cfg[0] + '=None, '
                init_body += f'{cfg[0]}={cfg[0]}, '
                par_body += f'{cfg[0]}, '
            par_body += ')'

            par_def = str(init_def)

            init_def += 'fmt=None'
            init_body += f'fmt=fmt'

            if hasattr(cls, '_extra_kw') and isinstance(cls._extra_kw, tuple):
                pos_args = []
                for kw in cls._extra_kw:
                    # FIXME: repr may fail!
                    if len(kw) == 2:
                        init_def += f', {kw[0]}={repr(kw[1])}'
                    else:
                        pos_args.append(kw[0])
                    init_body += f', {kw[0]}={kw[0]}'

                if pos_args:
                    s = init_def
                    init_def = s[:6] + ', '.join(pos_args) + ', ' + s[6:]

            func_code = f'def __init__({init_def}):\n    '
            func_code += f'super(type(self), type(self))'
            func_code += f'.__init__(self, {init_body})\n'
            func_code += f'def {name}({par_def}):\n    '
            func_code += f'return {par_body}'
            # <<< construct __init__ function <<<

            # now create and set __init__ function
            tmp = {}
            exec(func_code, tmp)
            init_func = tmp['__init__']
            init_func.__qualname__ = f'{name}.__init__'
            cls.__init__ = init_func
            cls._get_args = tmp[name]


class ParamConfig(NamedTuple):
    """Parameter configuration.

    Parameters
    ----------
    name : str
        Name of the parameter.
    fmt : str
        Tex format of the parameter.
    min : float, or int
        Minimum value of the parameter.
    max : float, or int
        Maximum value of the parameter.
    frozen : bool
        Whether the parameter is frozen.
    log : bool
        Whether the parameter is uniform in log scale.

    """

    name: str
    fmt: str
    default: float
    min: float
    max: float
    frozen: bool
    log: bool


class Component(Model, ABC, metaclass=ComponentMeta):
    """The abstract spectral component class."""

    def __init__(self, fmt=None, **params):
        if fmt is None:
            fmt = r'\mathrm{%s}' % self.__class__.__name__.lower()

        # parse parameters
        params_dict = {}
        for cfg in self._config:
            param_name = cfg[0]
            param = params[param_name]

            if param is None:
                # use default configuration
                params_dict[param_name] = UniformParameter(*cfg)

            elif isinstance(param, Parameter):
                # specified by user
                params_dict[param_name] = param

            elif isinstance(param, (float, int)):
                # frozen to the value given by user
                p = UniformParameter(*cfg)
                p.default = param
                p.frozen = True
                params_dict[param_name] = p

            else:
                # other input types are not supported yet
                cls_name = type(self).__name__
                type_name = type(param).__name__
                raise TypeError(
                    f'got {type_name} type {repr(param)} for '
                    f'{cls_name}.{param_name}, which is not supported'
                )

        if self.type == 'ncon':  # normalization convolution type
            mtype = 'con'
            is_ncon = True
        else:
            mtype = self.type
            is_ncon = False

        component = ModelNode(
            name=type(self).__name__.lower(),
            fmt=fmt,
            mtype=mtype,
            params={k: v._node for k, v in params_dict.items()},
            func=self._func,
            is_ncon=is_ncon
        )

        params_fmt = {cfg[0]: cfg[1] for cfg in self._config}

        super().__init__(component, params_dict, params_fmt)

    @property
    @abstractmethod
    def _func(self) -> Callable:
        """Return model evaluation function, overriden by subclass."""
        pass

    @property
    @abstractmethod
    def type(self) -> str:
        """Model type, overriden by subclass."""
        pass

    @property
    @abstractmethod
    def _config(self) -> tuple[ParamConfig, ...]:
        """Configuration of parameters, overriden by subclass."""
        pass

    @property
    def _extra_kw(self) -> tuple:
        """Extra keywords passed to ``__init__``, overriden by subclass.

        Note that element of inner tuple should respect :func:`repr`, or
        ``__init__`` of :class:`ComponentMeta` may fail.

        """
        return tuple()


def generate_parameter(
    name: str,
    fmt: str,
    default: float,
    distribution: Distribution,
    min: Optional[float] = None,
    max: Optional[float] = None,
    dist_expr: Optional[str] = None
) -> Parameter:
    """Create :class:`Parameter` instance.

    Parameters
    ----------
    name : str
        Name of the parameter.
    fmt : str
        Tex format of the parameter.
    default : float
        Default value of the parameter.
    distribution : Distribution
        Instance of :class:`numpyro.distributions.Distribution`.
    min : float, optional
        Minimum value of the parameter. Defaults to None.
    max : float, optional
        Maximum value of the parameter. Defaults to None.
    dist_expr : str, optional
        Expression of the parameter distribution. Defaults to 'CustomDist'.

    Returns
    -------
    Parameter
        The generated parameter.

    """
    dist_expr = str(dist_expr) if dist_expr is not None else 'CustomDist'
    node = ParameterNode(name, fmt, default, distribution, min, max, dist_expr)

    return Parameter(node)


def generate_model(
    name: str,
    fmt: str,
    mtype: str,
    params: dict[str, Parameter],
    func: Callable,
    is_ncon: bool
) -> Model:
    """Create :class:`Model` instance.

    Parameters
    ----------
    name : str
        Name of the model.
    fmt : str
        Tex format of the model.
    mtype : {'add', 'mul', 'con'}
        Model type.
    params : dict
        A :class:`str`-:class:`Parameter` mapping that defines the parameters
        of the model.
    func : callable
        Evaluation function of the model.
    is_ncon : bool
        Whether the model is normalization convolution type.

    Returns
    -------
    Model
        The generated model.

    Notes
    -----
    The signature of `func` should be ``func(egrid, par1, ...)``, where
    `func` is expected to return ndarray of length ``len(egrid) - 1`` (i.e.,
    integrating over `egrid`), and ``par1, ...`` matches `params`.

    If the function is convolution type, operating on flux (con) or norm
    (ncon), then corresponding signature of `func` should be
    ``func(egrid, flux, par1, ...)`` or ``func(flux_func, flux, par1, ...)``,
    where `flux` is ndarray of length ``len(egrid) - 1``, and `flux_func` has
    the same signature and returns as aforementioned.

    """
    params_node = {k: v._node for k, v in params.items()}
    model_node = ModelNode(name, fmt, mtype, params_node, func, is_ncon)

    return Model(model_node, params)
