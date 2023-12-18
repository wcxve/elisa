"""Module to handle model construction."""
# TODO:
#    - model construction and parameter binding should be separated
#    - transformable to other PPL libs
#    - time-dependent model
from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from typing import Callable, Optional, Union

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

__all__ = ['UniformParameter', 'generate_parameter', 'generate_model']


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
        """Check and set parameter configure."""
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
        """Get distribution expression for :class:`ParameterNode`."""
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
            raise ValueError(
                f'{self} has no "{name}" parameter'
            )

        return self._params[name]

    def __setitem__(self, name: str, param: Parameter):
        if name not in self._params:
            raise ValueError(
                f'{self} has no "{name}" parameter'
            )

        setattr(self, name, param)

    @property
    def type(self) -> str:
        """Model type."""
        return self._node.attrs['mtype']

    @property
    def params_fmt(self) -> ModelParameterFormat:
        """Model parameter format."""
        return self._params_fmt[self._node.name]

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

    @property
    def _wrapped_func(self) -> Callable:
        """Model evaluation function."""
        f = self._node.generate_func(self._label.mapping['name'])

        # TODO: eliminate *args and **kwargs
        def func(egrid, params, *args, **kwargs):
            """Model function wrapper."""
            return f(params, egrid, *args, **kwargs)

        return func

    @property
    def _model_info(self) -> dict:
        """Model information.

        Returns sample site information of :mod:`numpyro`,
        model parameter configure based on site, model function and id mapping.
        """
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
            func=self._wrapped_func,
            mname=mapping['name'],
            mfmt=mapping['fmt'],
            mpfmt=self._params_fmt
        )
        return info

    # def set_params(self, params: dict[str, Parameter]) -> None:
    #     """Set parameters."""
    #
    #     params_in = set(params.keys())
    #     params_all = set(self._params_name)
    #
    #     if not params_in <= params_all:
    #         diff = params_all - params_in
    #         raise ValueError(f'input params {diff} not included in {self}')
    #
    #     if not all(map(lambda v: isinstance(v, Parameter), params.values())):
    #         raise ValueError('Parameter type is required')
    #
    #     params = self._params | params
    #     params_node = list(map(lambda v: v._node, params.values()))
    #     self._params = params
    #     self._node.predecessor = params_node
    #
    # @property
    # def additives(self):
    #     return ...
    #
    # def flux(self):
    #     ...
    #
    # def ne(self):
    #     ...
    #
    # def ene(self):
    #     ...
    #
    # def eene(self):
    #     ...
    #
    # def ce(self):
    #     ...
    #
    # def counts(self):
    #     ...
    #
    # def fmt(self):
    #     ...


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


class ComponentMeta(ABCMeta):
    """Metaclass to avoid cumbersome code for ``__init__`` of subclass."""

    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # if subclass has ``default`` defined correctly, override its __init__
        if hasattr(cls, '_default') and isinstance(cls._default, tuple):
            name = cls.__name__.lower()

            # >>> construct __init__ function >>>
            default = cls._default
            init_def = 'self, '
            init_body = ''
            par_body = '('
            for cfg in default:
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
                    if kw[1] is not None:
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


class Component(Model, ABC, metaclass=ComponentMeta):
    """The abstract spectral component class."""

    def __init__(self, fmt=None, **params):
        if fmt is None:
            fmt = r'\mathrm{%s}' % self.__class__.__name__.lower()

        # parse parameters
        params_dict = {}
        for cfg in self._default:
            param_name = cfg[0]
            param = params[param_name]

            if param is None:
                # use default configure
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

        params_fmt = {cfg[0]: cfg[1] for cfg in self._default}

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
    def _default(self) -> tuple:
        """Default configuration of parameters, overriden by subclass.

        Configuration format should be ``((name: str, fmt: str, default: float,
        min: float, max: float, frozen: bool, log: bool), ...)``.

        """
        pass

    @property
    def _extra_kw(self) -> tuple:
        """Extra keywords passed to ``__init__``, overriden by subclass.

        Note that Element of inner tuple should respect :py:func:`repr`, or
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

#
#
# class Model(metaclass=ComponentMeta):
#     def _get_param(self, key):
#         """Check and return component and parameter key."""
#
#         if '.' not in key:
#             raise ValueError('key format should be "component.parameter"')
#
#         c, p = key.split('.')
#
#         name_map = {v: k for k, v in self._label._label_map['name'].items()}
#
#         if c not in name_map:
#             raise ValueError(f'No component "{c}" in model "{self}"')
#
#         return self._node.comps[name_map[c]], p
#
#     @property
#     def comps(self):
#         """Additive type of compositions of subcomponents."""
#
#         if self.type != 'add':
#             raise TypeError(
#                 f'"{self.type}" model has no additive sub-components'
#             )
#         else:
#             if not self._comps:
#                 raise NotImplementedError
#
#             return self._comps
#
#     # def flux(self, params, e_range, energy=True, ngrid=1000, log=True):
#     #     """Evaluate model by
#     #         * analytic expression
#     #         * trapezoid or Simpson's 1/3 rule given :math:`N_E`
#     #         * Xspec model library
#     #
#     #     TODO: docstring
#     #
#     #     """
#     #
#     #     if self.type != 'add':
#     #         raise TypeError(
#     #             f'flux is undefined for "{self.type}" type model "{self}"'
#     #         )
#     #
#     #     if log:  # evenly spaced grid in log space
#     #         egrid = np.geomspace(*e_range, ngrid)
#     #     else:  # evenly spaced grid in linear space
#     #         egrid = np.linspace(*e_range, ngrid)
#     #
#     #     if energy:  # energy flux
#     #         flux = np.sum(self.ENE(pars, ebins) * np.diff(ebins), axis=-1)
#     #     else:  # photon flux
#     #         flux = np.sum(self.NE(pars, ebins) * np.diff(ebins), axis=-1)
#     #
#     #     return flux
#
#     def _build_prior(self) -> Callable:
#         """Get the prior function which will be used in numpyro."""
#
#         def prior():
#             """This should be called inside numpyro."""
#             self
#             ...
#
#         return prior
#
#     def __call__(self, egrid, flux=None):
#         # TODO:
#         return self.eval(egrid, flux)
#
#     def eval(self, *args, **kwargs):
#         """TODO"""
#         if self.type != 'con':
#             ...
#         else:
#             ...
#
