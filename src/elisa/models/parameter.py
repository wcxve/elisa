"""The parameter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple, get_args

import jax.numpy as jnp
from numpyro.distributions import Distribution, LogUniform, Uniform

from elisa.util.integrate import AdaptQuadMethod, make_integral_factory
from elisa.util.misc import build_namespace

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any, Callable, Literal

    # from tinygp import kernels, means, noise
    from elisa.util.integrate import IntegralFactory
    from elisa.util.typing import (
        CompID,
        CompParamName,
        JAXArray,
        JAXFloat,
        ParamID,
        ParamIDStrMapping,
        ParamIDValMapping,
    )


class AssignmentTracker:
    """Track component assignment of a parameter."""

    def __init__(self):
        self._history = []

    def append(self, cid: CompID, pname: CompParamName) -> None:
        """Append a new assignment record."""
        record = (cid, pname)
        if record not in self._history:
            self._history.append((cid, pname))

    def remove(self, cid: CompID, pname: CompParamName) -> None:
        """Remove an assignment record."""
        self._history.remove((cid, pname))

    def get_comp_param(
        self,
        comp_ids: Iterable[CompID],
    ) -> tuple[CompID, CompParamName] | None:
        """Get the earliest component assignment record within comp_ids."""
        id_set = set(comp_ids)
        flag = [i[0] in id_set for i in self._history]
        if any(flag):
            return self._history[flag.index(True)]
        else:
            return None


class ParamInfo(NamedTuple):
    """Parameter information."""

    name: str | Callable[[ParamIDStrMapping], str]
    """Plain name of the parameter.

    It is a getter function for composite parameter.
    """

    latex: str | Callable[[ParamIDStrMapping], str]
    r""":math:`\LaTeX` format of the parameter.

    It is a getter function for composite parameter.
    """

    default: Any
    """Default value of the parameter."""

    bound: str
    """Value bound expression of the parameter."""

    prior: str
    """Prior distribution expression of the parameter."""

    log: bool
    """Whether the parameter is parameterized in a logarithmic scale."""

    fixed: bool
    """Whether the parameter is fixed."""

    tracker: AssignmentTracker
    """Component assignment tracker."""

    id_to_value: Callable[[ParamIDValMapping], JAXFloat]
    """Mapping function from id to value."""

    dist: Distribution | None = None
    """NumPyro distribution for the parameter."""

    composite: bool = False
    """Whether the parameter is composite."""

    integrate: IntegralFactory | bool = False
    """Integration factory for the parameter for interval parameter.

    For composite parameter composed by interval parameter, this is True.
    """


class Parameter(ABC):
    """Parameter base."""

    _id: ParamID
    _tracker: AssignmentTracker
    _nodes_id: tuple[ParamID, ...]

    def __init__(self):
        self._id = hex(id(self))[2:]
        self._tracker = AssignmentTracker()
        self._nodes_id = (self._id,)

    def _id_to_label(
        self,
        mapping: ParamIDStrMapping,
        label_type: Literal['name', 'latex'],
    ) -> str:
        """Get the label of the parameter."""
        if label_type not in {'name', 'latex'}:
            raise ValueError(f'unknown label type: {label_type}')

        return mapping[self._id]

    @property
    def _id_to_value(self) -> Callable[[ParamIDValMapping], JAXFloat]:
        """Gets the mapping function from id to value."""
        pid = self._id
        default = self.default

        def id_to_value(mapping: ParamIDValMapping) -> JAXFloat:
            """Get the value of the parameter from mapping."""
            return mapping.get(pid, default)

        return id_to_value

    @property
    @abstractmethod
    def name(self) -> str:
        """Plain name of the parameter."""
        pass

    @property
    @abstractmethod
    def latex(self) -> str:
        r""":math:`\LaTeX` format of the parameter."""
        pass

    @property
    @abstractmethod
    def default(self) -> JAXFloat:
        """Default value of the parameter."""
        pass

    @property
    @abstractmethod
    def log(self) -> bool:
        """Whether the parameter is parameterized in a logarithmic scale."""
        pass

    @property
    @abstractmethod
    def fixed(self) -> bool:
        """Whether the parameter is fixed."""
        pass

    @property
    @abstractmethod
    def _info(self) -> dict[ParamID, ParamInfo]:
        """Parameter information."""
        pass

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def __add__(self, other: Parameter) -> CompositeParameter:
        return self._make_composite_parameter(self, other, '+')

    def __radd__(self, other: Parameter) -> CompositeParameter:
        return self._make_composite_parameter(other, self, '+')

    def __sub__(self, other: Parameter) -> CompositeParameter:
        return self._make_composite_parameter(self, other, '-')

    def __rsub__(self, other: Parameter) -> CompositeParameter:
        return self._make_composite_parameter(other, self, '-')

    def __mul__(self, other: Parameter) -> CompositeParameter:
        return self._make_composite_parameter(self, other, '*')

    def __rmul__(self, other: Parameter) -> CompositeParameter:
        return self._make_composite_parameter(other, self, '*')

    def __truediv__(self, other: Parameter) -> CompositeParameter:
        return self._make_composite_parameter(self, other, '/')

    def __rtruediv__(self, other: Parameter) -> CompositeParameter:
        return self._make_composite_parameter(other, self, '/')

    def __pow__(self, other: Parameter) -> CompositeParameter:
        return self._make_composite_parameter(self, other, '^')

    def __rpow__(self, other: Parameter) -> CompositeParameter:
        return self._make_composite_parameter(other, self, '^')

    @staticmethod
    def _make_composite_parameter(
        lhs: Parameter,
        rhs: Parameter,
        op: Literal['+', '-', '*', '/', '^'],
    ) -> CompositeParameter:
        # check if the type of lhs and rhs are both parameter
        if not (isinstance(lhs, Parameter) and isinstance(rhs, Parameter)):
            raise TypeError(
                f'unsupported operand types for {op}: '
                f"'{type(lhs).__name__}' and '{type(rhs).__name__}'"
            )

        op_symbol = op

        if op == '+':
            op = jnp.add
            op_name = '{} + {}'
            op_latex = '{{{}}} + {{{}}}'
        elif op == '-':
            op = jnp.subtract
            op_name = '{} - {}'
            op_latex = '{{{}}} - {{{}}}'
        elif op == '*':
            op = jnp.multiply
            op_name = '{} * {}'
            op_latex = r'{{{}}} \times {{{}}}'
        elif op == '/':
            op = jnp.divide
            op_name = '{} / {}'
            op_latex = r'{{{}}} / {{{}}}'
        elif op == '^':
            op = jnp.power
            op_name = '{}^{}'
            op_latex = r'{{{}}}^{{{}}}'
        else:
            raise NotImplementedError(f'op {op}')

        return CompositeParameter(
            params=[lhs, rhs],
            op=op,
            op_name=op_name,
            op_latex=op_latex,
            op_symbol=op_symbol,
        )


class ParameterHelper(Parameter):
    """Handle name, latex, and default value of a parameter."""

    def __init__(
        self,
        name: str,
        latex: str | None,
        default: Any | None = None,
    ):
        if latex is None:
            latex = name

        self._name = name
        self.latex = latex

        if default is not None:
            self.default = default

        super().__init__()

    @property
    def name(self) -> str:
        return self._name

    @property
    def latex(self) -> str:
        return self._latex

    @latex.setter
    def latex(self, latex):
        self._latex = str(latex)

    @property
    @abstractmethod
    def default(self) -> Any:
        pass

    @default.setter
    @abstractmethod
    def default(self, value: Any):
        pass


class DistParameter(ParameterHelper):
    r"""Define a parameter by a distribution.

    Parameters
    ----------
    name : str
        Parameter name.
    dist : Distribution
        Numpyro distribution from which the parameter is sampled.
    default : float
        Parameter default value.
    min : float, optional
        Parameter minimum value, for display purpose only. The default is None.
    max : float, optional
        Parameter maximum value, for display purpose only. The default is None.
    log : bool, optional
        Whether the parameter is parameterized in a logarithmic scale.
        The default is False.
    fixed : bool, optional
        Whether the parameter is fixed. The default is False.
    latex : str, optional
        :math:`\LaTeX` format of the parameter. The default is as `name`.
    """

    def __init__(
        self,
        name: str,
        dist: Distribution,
        default: float,
        *,
        min: float | None = None,
        max: float | None = None,
        log: bool = False,
        fixed: bool = False,
        latex: str | None = None,
    ):
        if not isinstance(dist, Distribution):
            raise ValueError('prior must be a numpyro distribution')

        if jnp.shape(default) != ():
            raise ValueError('default must be a scalar')

        if not bool(dist._validate_sample(default)):
            raise ValueError('default should be within the prior support')

        self._dist = dist
        self._default = jnp.asarray(default, float)

        if min is None:
            self._min = None
        else:
            if jnp.shape(min) != ():
                raise ValueError('min must be a scalar')
            self._min = jnp.asarray(min, float)

        if max is None:
            self._max = None
        else:
            if jnp.shape(max) != ():
                raise ValueError('max must be a scalar')
            self._max = jnp.asarray(max, float)

        self._log = bool(log)
        self._fixed = bool(fixed)

        super().__init__(name, latex, default)

    @property
    def default(self) -> JAXFloat:
        return self._default

    @default.setter
    def default(self, default: float):
        if jnp.shape(default) != ():
            raise ValueError('default must be a scalar')

        if not bool(self._dist._validate_sample(default)):
            raise ValueError('default should be within the dist support')

        self._default = jnp.asarray(default, float)

    @property
    def log(self) -> bool:
        return self._log

    @property
    def fixed(self) -> bool:
        return self._fixed

    @property
    def _dist_expr(self) -> str:
        dist = self._dist
        name = dist.__class__.__name__
        args = [
            f'{arg}={getattr(dist, arg):.4g}'
            for arg in self._dist.arg_constraints
        ]
        args = ', '.join(args)
        return f'{name}({args})'

    @property
    def _info(self) -> dict[ParamID, ParamInfo]:
        vmin = f'{self._min:.4g}' if self._min is not None else '???'
        vmax = f'{self._max:.4g}' if self._max is not None else '???'
        if vmin == '???' or vmax == '???':
            bound_expr = str(self._dist.support)
        else:
            bound_expr = f'({vmin}, {vmax})'

        info = ParamInfo(
            name=self.name,
            latex=self.latex,
            default=self.default,
            bound='' if self.fixed else bound_expr,
            prior='' if self.fixed else self._dist_expr,
            log=self.log,
            fixed=self.fixed,
            tracker=self._tracker,
            id_to_value=self._id_to_value,
            dist=None if self.fixed else self._dist,
        )

        return {self._id: info}


class UniformParameter(DistParameter):
    r"""Define the parameter by a uniform distribution.

    Parameters
    ----------
    name : str
        Parameter name.
    default : float
        Parameter default value.
    min : float
        Parameter minimum value.
    max : float
        Parameter maximum value.
    log : bool, optional
        Whether the parameter is logarithmically uniform. The default is False.
    fixed : bool, optional
        Whether the parameter is fixed. The default is False.
    latex : str, optional
        :math:`\LaTeX` format of the parameter. The default is as `name`.
    """

    def __init__(
        self,
        name: str,
        default: float,
        min: float,
        max: float,
        *,
        log: bool = False,
        fixed: bool = False,
        latex: str | None = None,
    ):
        self._log = bool(log)

        self._check_and_set_values(default, min, max)

        super().__init__(
            name,
            self._dist,
            default,
            min=min,
            max=max,
            log=log,
            fixed=fixed,
            latex=latex,
        )

    def __repr__(self) -> str:
        if self._fixed:
            return f'{self.name} = {self.default:.4g}'
        elif self._log:
            return f'{self.name} ~ LogUniform({self.min:.4g}, {self.max:.4g})'
        else:
            return f'{self.name} ~ Uniform({self.min:.4g}, {self.max:.4g})'

    @property
    def name(self) -> str:
        return self._name

    @property
    def latex(self) -> str:
        return self._latex

    @latex.setter
    def latex(self, latex: str):
        self._latex = str(latex)

    @property
    def default(self) -> JAXFloat:
        return self._default

    @default.setter
    def default(self, default: float):
        self._check_and_set_values(default=default)

    @property
    def min(self) -> JAXFloat:
        """Parameter minimum value."""
        return self._min

    @min.setter
    def min(self, min: float):
        self._check_and_set_values(min=min)

    @property
    def max(self) -> JAXFloat:
        """Parameter maximum value."""
        return self._max

    @max.setter
    def max(self, max: float):
        self._check_and_set_values(max=max)

    @property
    def log(self) -> bool:
        """Whether the parameter is logarithmically uniform."""
        return self._log

    @log.setter
    def log(self, log: bool):
        log = bool(log)
        if self._log != log:
            self._log = log

            if log:
                self._dist = LogUniform(self._min, self._max)
            else:
                self._dist = Uniform(self._min, self._max)

    @property
    def fixed(self) -> bool:
        return self._fixed

    @fixed.setter
    def fixed(self, fixed: bool):
        self._fixed = bool(fixed)

    @property
    def _dist_expr(self) -> str:
        if self._log:
            return f'LogUniform({self.min:.4g}, {self.max:.4g})'
        else:
            return f'Uniform({self.min:.4g}, {self.max:.4g})'

    def _check_and_set_values(
        self,
        default: float | None = None,
        min: float | None = None,
        max: float | None = None,
    ) -> None:
        """Check and set parameter configuration."""
        if default is None:
            _default = self._default
        else:
            if jnp.shape(default) != ():
                raise ValueError('default must be a scalar')
            _default = jnp.asarray(default, float)

        if min is None:
            _min = self._min
        else:
            if jnp.shape(min) != ():
                raise ValueError('min must be a scalar')
            _min = jnp.asarray(min, float)

        if max is None:
            _max = self._max
        else:
            if jnp.shape(max) != ():
                raise ValueError('max must be a scalar')
            _max = jnp.asarray(max, float)

        if _min <= 0.0 and self._log:
            raise ValueError(f'min ({_min}) must be positive for log uniform')

        if _min >= _max:
            raise ValueError(f'min ({_min}) must be less than max ({_max})')

        if _default <= _min:
            raise ValueError(
                f'default ({_default}) must be greater than min ({_min})'
            )

        if _default >= _max:
            raise ValueError(
                f'default ({_default}) must be less than max ({_max})'
            )

        if default is not None:
            self._default = _default

        if min is not None:
            self._min = _min

        if max is not None:
            self._max = _max

        if min is not None or max is not None:
            if self.log:
                self._dist = LogUniform(self._min, self._max)
            else:
                self._dist = Uniform(self._min, self._max)


class ConstantParameter(ParameterHelper):
    r"""Constant parameter.

    Parameters
    ----------
    name : str
        Parameter name.
    value : array_like
        The constant value of parameter.
    latex : str, optional
        :math:`\LaTeX` format of the parameter. The default is as `name`.
    """

    def __init__(self, name: str, value: Any, latex: str | None = None):
        super().__init__(name, latex, value)

    @property
    def log(self) -> bool:
        """Constant parameter is not logarithmically parameterized."""
        return False

    @property
    def fixed(self) -> bool:
        """Constant parameter is fixed."""
        return True


class ConstantValue(ConstantParameter):
    r"""Parameter with a fixed value.

    Parameters
    ----------
    name: str
        Parameter name.
    value: float
        Parameter value.
    latex : str, optional
        :math:`\LaTeX` format of the parameter. The default is as `name`.
    """

    def __init__(self, name: str, value: float, latex: str | None = None):
        super().__init__(name, value, latex)

    def __repr__(self) -> str:
        return f'{self.name} = {self.default:.4g}'

    @property
    def default(self) -> JAXFloat:
        return self._default

    @default.setter
    def default(self, default: float):
        if jnp.shape(default) != ():
            raise ValueError('default must be a scalar')

        self._default = jnp.asarray(default, float)

    @property
    def _info(self) -> dict[ParamID, ParamInfo]:
        info = ParamInfo(
            name=self.name,
            latex=self.latex,
            default=self.default,
            bound='',  # this is not supposed to be used
            prior='',  # this is not supposed to be used
            log=self.log,
            fixed=self.fixed,
            tracker=self._tracker,
            id_to_value=self._id_to_value,
        )

        return {self._id: info}


class ConstantInterval(ConstantParameter):
    r"""Constant parameter to be integrated over a given interval.

    When assigning :class:`ConstantInterval` parameters to a model component,
    the model will be evaluated according to the following formula:

    .. math::
        \frac{1}{\prod_i (b_i - a_i)}
        \int f(E, \vec{\theta}(\vec{p}, \vec{q})) \, \mathrm{d} \vec{p}

    where :math:`f` is the model function, :math:`\vec{\theta}` is the
    parameter vector of the model, :math:`\vec{p}` is the
    :class:`ConstantInterval` parameters, :math:`\vec{q}` is the other
    parameters, and :math:`a_i` and :math:`b_i` are the intervals given
    by :math:`\vec{p}`.

    Parameters
    ----------
    name: str
        Parameter name.
    interval: array_like
        The interval, a 2-element sequence.
    method : {'quadgk', 'quadcc', 'quadts', 'romberg', 'rombergts'}, optional
        Numerical integration method used to integrate over the parameter.
        Available options are:

            * ``'quadgk'``: global adaptive quadrature by Gauss-Konrod rule
            * ``'quadcc'``: global adaptive quadrature by Clenshaw-Curtis rule
            * ``'quadts'``: global adaptive quadrature by trapz tanh-sinh rule
            * ``'romberg'``: Romberg integration
            * ``'rombergts'``: Romberg integration by tanh-sinh
              (a.k.a. double exponential) transformation

        The default is ``'quadgk'``.
    latex : str, optional
        :math:`\LaTeX` format of the parameter. The default is as `name`.
    kwargs : dict, optional
        Extra kwargs passed to integration methods. See [1]_ for details.

    References
    ----------
    .. [1] `quadax docs <https://quadax.readthedocs.io/en/latest/api.html#adaptive-integration-of-a-callable-function-or-method>`__
    """

    def __init__(
        self,
        name: str,
        interval: Sequence[float],
        method: AdaptQuadMethod = 'quadgk',
        latex: str | None = None,
        **kwargs,
    ):
        super().__init__(name, interval, latex)
        self.method = method
        self._integrate_kwargs = {'epsabs': 0.0, 'epsrel': 1.4e-8} | kwargs

    def __repr__(self) -> str:
        return f'{self.name} = [{self.default[0]:.4g}, {self.default[1]:.4g}]'

    @property
    def default(self) -> JAXArray:
        return self._default

    @default.setter
    def default(self, default):
        if jnp.shape(default) != (2,):
            raise ValueError('interval must be a 2-element sequence')

        self._default = jnp.asarray(default, float)

    @property
    def method(self) -> AdaptQuadMethod:
        """Numerical integration method."""
        return self._method

    @method.setter
    def method(self, value: AdaptQuadMethod):
        supported = get_args(AdaptQuadMethod)
        if value not in supported:
            raise ValueError(f'method must be one of {supported}')

        self._method = value

    @property
    def _info(self) -> dict[ParamID, ParamInfo]:
        factory = make_integral_factory(
            param_id=self._id,
            interval=self.default,
            method=self.method,
            kwargs=self._integrate_kwargs,
        )

        info = ParamInfo(
            name=self.name,
            latex=self.latex,
            default=self.default,
            bound='',  # this is not supposed to be used
            prior='',  # this is not supposed to be used
            log=self.log,
            fixed=self.fixed,
            tracker=self._tracker,
            integrate=factory,
            id_to_value=self._id_to_value,
        )

        return {self._id: info}


# class DependentInterval(Parameter):
#     """Interval whose bounds are defined by function of parameters."""


class CompositeParameter(Parameter):
    r"""Combine parameters to create a new parameter.

    Parameters
    ----------
    params : Parameter, or sequence of Parameter
        Parameters to be combined.
    op : callable
        Function applied to `params`. The function should take the same
        number and order of arguments as `params` and return a single value.
        The function must be compatible with :mod:`JAX`.
    op_name : str
        Name of the composition operator `op`.
    op_latex : str, optional
        :math:`\LaTeX` format of the composition operator `op`.
        The default is as `op_name`.
    """

    _params: tuple[Parameter, ...]
    _has_interval: bool

    def __init__(
        self,
        params: Parameter | Sequence[Parameter],
        op: Callable[..., JAXFloat],
        op_name: str,
        op_latex: str | None = None,
        *,
        op_symbol: Literal['+', '-', '*', '/', '^'] | None = None,
    ):
        # check if the type of params is parameter or sequence
        if not isinstance(params, (Parameter, Sequence)):
            raise TypeError(
                'parameters must be a Parameter or a sequence of Parameter'
            )

        # make params a list
        if isinstance(params, Parameter):
            params = [params]
        else:
            params = list(params)

        # check if params are all parameters
        if not all(isinstance(i, Parameter) for i in params):
            raise TypeError(
                'parameters must be a Parameter or a sequence of Parameter'
            )

        if op_symbol not in {'+', '-', '*', '/', '^', None}:
            raise ValueError('`op_symbol` is for internal use only')

        super().__init__()

        self._op = op
        self._op_name = str(op_name)
        self._op_latex = self._op_name if op_latex is None else str(op_latex)
        self._op_symbol = op_symbol
        self._params = tuple(params)

        for p in self._params:
            if isinstance(p, ConstantInterval) or (
                isinstance(p, CompositeParameter) and p._has_interval
            ):
                self._has_interval = True
                break
        else:
            self._has_interval = False

        # correct Parameter's _nodes_id attribute
        nodes = []
        for p in self._params:
            stack = [p]
            while stack:
                node = stack.pop(0)
                if isinstance(node, CompositeParameter):
                    stack = list(node._params) + stack
                elif node not in nodes:
                    nodes.append(node)
        self._nodes = tuple(nodes)
        self._nodes_id = tuple(p._id for p in self._nodes)

        pid_to_pname = dict(
            zip(
                self._nodes_id,
                build_namespace([p.name for p in self._nodes], prime=True)[
                    'namespace'
                ],
            )
        )
        self._name = self._id_to_label(pid_to_pname, 'name')

    def _id_to_label(
        self,
        mapping: dict[ParamID, str],
        label_type: Literal['name', 'latex'],
    ) -> str:
        if label_type not in {'name', 'latex'}:
            raise ValueError(f'unknown label type: {label_type}')

        if self._id in mapping:
            return mapping[self._id]
        else:
            if self._op_symbol:
                op = self._op_symbol
                lhs, rhs = self._params

                if op == '+':
                    lhs_fmt = rhs_fmt = '{}'

                elif op == '-':
                    lhs_fmt = '{}'

                    if isinstance(
                        rhs, CompositeParameter
                    ) and rhs._op_symbol not in {'*', '/', '^'}:
                        rhs_fmt = '({})'
                    else:
                        rhs_fmt = '{}'

                elif op == '*':
                    if isinstance(
                        lhs, CompositeParameter
                    ) and lhs._op_symbol not in {'*', '/', '^'}:
                        lhs_fmt = '({})'
                    else:
                        lhs_fmt = '{}'

                    if isinstance(
                        rhs, CompositeParameter
                    ) and rhs._op_symbol not in {'*', '/', '^'}:
                        rhs_fmt = '({})'
                    else:
                        rhs_fmt = '{}'

                elif op == '/':
                    if isinstance(
                        lhs, CompositeParameter
                    ) and lhs._op_symbol not in {'*', '/', '^'}:
                        lhs_fmt = '({})'
                    else:
                        lhs_fmt = '{}'

                    rhs_fmt = '({})'

                elif op == '^':
                    if isinstance(lhs, CompositeParameter):
                        lhs_fmt = '({})'
                    else:
                        lhs_fmt = '{}'

                    if isinstance(rhs, CompositeParameter):
                        rhs_fmt = '({})'
                    else:
                        rhs_fmt = '{}'

                else:
                    raise NotImplementedError(f'op_symbol: {op}')

                labels = (
                    lhs_fmt.format(lhs._id_to_label(mapping, label_type)),
                    rhs_fmt.format(rhs._id_to_label(mapping, label_type)),
                )

            else:
                labels = tuple(
                    (
                        '({})' if isinstance(p, CompositeParameter) else '{}'
                    ).format(p._id_to_label(mapping, label_type))
                    for p in self._params
                )

            temp = self._op_name if label_type == 'name' else self._op_latex
            return temp.format(*labels)

    @property
    def _id_to_value(self) -> Callable[[ParamIDValMapping], JAXFloat]:
        fns = [param._id_to_value for param in self._params]

        def id_to_value(mapping: ParamIDValMapping) -> JAXFloat:
            """Get the value of the composite parameter."""
            return self._op(*[fn(mapping) for fn in fns])

        return id_to_value

    @property
    def name(self) -> str:
        return self._name

    @property
    def latex(self) -> str:
        nodes_latex = [p.latex for p in self._nodes]
        latex = build_namespace(nodes_latex, True, True)['namespace']
        pid_to_latex = dict(zip(self._nodes_id, latex))
        return self._id_to_label(pid_to_latex, 'latex')

    @property
    def default(self) -> JAXFloat:
        if self._has_interval:
            raise RuntimeError(
                'cannot get default value of a composite interval'
            )

        return self._op(*[i.default for i in self._params])

    @property
    def log(self) -> bool:
        """If the sub-parameters are all logarithmically parameterized."""
        return all(i.log for i in self._params)

    @property
    def fixed(self) -> bool:
        """If the sub-parameters are all fixed."""
        return all(i.fixed for i in self._params)

    @property
    def _info(self) -> dict[ParamID, ParamInfo]:
        info = {
            self._id: ParamInfo(
                name=lambda mapping: self._id_to_label(mapping, 'name'),
                latex=lambda mapping: self._id_to_label(mapping, 'latex'),
                default=jnp.nan,  # this is not supposed to be used
                bound='',  # this is not supposed to be used
                prior='',  # this is not supposed to be used
                log=self.log,
                fixed=self.fixed,
                tracker=self._tracker,
                id_to_value=self._id_to_value,
                integrate=self._has_interval,
                composite=True,
            )
        }

        for p in self._params:
            info |= p._info

        return info


# class GPParameter(ParameterHelper):
#     """Parameter sampled from a Gaussian process."""
#
#     def __init__(
#         self,
#         name: str,
#         kernel: kernels.Kernel,
#         x: JAXFloat,
#         *,
#         diag: JAXFloat | None = None,
#         noise: noise.Noise | None = None,
#         mean: means.MeanBase | Callable | JAXFloat | None = None,
#         log: bool = False,
#         latex: str | None = None,
#     ):
#         self._log = bool(log)
#         super().__init__(name, latex, None)
#
#     def _fn(self, name: str | None, rng_key: PRNGKey | None) -> JAXFloat:
#         """Sample from the Gaussian process."""
#         raise NotImplementedError
#
#     @property
#     def default(self) -> None:
#         return None
#
#     @property
#     def log(self) -> bool:
#         return self._log
#
#     @property
#     def fixed(self) -> bool:
#         return False
#
#     @property
#     def _info(self) -> dict[ParamID, ParamInfo]:
#         return ...
