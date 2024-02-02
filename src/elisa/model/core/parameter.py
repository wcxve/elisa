"""The parameter."""
from __future__ import annotations

import threading
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Any, Callable, Literal, Optional

import jax.numpy as jnp
import numpyro
from numpyro.distributions import Distribution, LogUniform, Uniform
from tinygp import kernels, means, noise

from elisa.util.typing import JAXArray, JAXFloat, PRNGKey

SampleFn = Callable[[Optional[str], Optional[PRNGKey]], JAXFloat]


__all__ = [
    'Parameter',
    'UniformParameter',
    'ConstantValue',
    'ConstantInterval',
    'CompositeParameter',
    'GPParameter',
]


class ParameterContext:
    """Context to store the values of parameters.

    Parameters
    ----------
    mapping : dict
        A name mapping of parameters.

    """

    context = threading.local()
    params: dict[str, JAXFloat]

    def __init__(self, mapping: dict[str, str], stack_name: str | None):
        if not hasattr(type(self).context, 'stack'):
            type(self).context.stack = {}

        if stack_name is None:
            self.stack_name = 'default'
        else:
            if stack_name == 'default':
                raise ValueError('default is a preserved name')

            self.stack_name = str(stack_name)

        # FIXME: solve conflict when using the same stack name?
        if self.stack_name in type(self).context.stack:
            raise RuntimeError(f'stack {self.stack_name} already exists')

        self.params = {}
        self.mapping = mapping

    def __enter__(self) -> ParameterContext:
        type(self).context.stack[self.stack_name] = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        type(self).context.stack.pop(self.stack_name)

    @classmethod
    def get_context(
        cls, stack_name: str | None = None
    ) -> ParameterContext | None:
        """Return the context."""
        if stack_name is None:
            stack_name = 'current'
        else:
            stack_name = str(stack_name)

        if not hasattr(cls.context, 'stack'):
            cls.context.stack = {}

        return cls.context.stack.get(stack_name, None)


class ParameterBase(metaclass=ABCMeta):
    """Parameter base."""

    _component: int | None = None

    def __init__(self, fn: SampleFn):
        # fn is a wrapper of numpyro.sample or numpyro.deterministic
        self._fn = fn
        self._id = hex(id(self))[2:]

    def sample(self, rng_key: PRNGKey | None = None) -> JAXFloat:
        """Get a sample from the parameter's underlying distribution."""
        ctx = ParameterContext.get_context()
        if ctx is not None:
            if self._id not in ctx.params:
                if self._id in ctx.mapping:
                    name = ctx.mapping[self._id]
                else:
                    name = None
                ctx.params[self._id] = self._fn(name, rng_key)
            return ctx.params[self._id]
        else:
            return self._fn(None, rng_key)

    @property
    @abstractmethod
    def name(self) -> str:
        """Plain name of the parameter."""
        pass

    @property
    @abstractmethod
    def latex(self) -> str:
        """LaTeX presentation of the parameter."""
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

    def __repr__(self) -> str:
        return self.name

    def __add__(self, other: ParameterBase) -> CompositeParameter:
        return self._make_composite_parameter(self, other, '+')

    def __radd__(self, other: ParameterBase) -> CompositeParameter:
        return self._make_composite_parameter(other, self, '+')

    def __sub__(self, other: ParameterBase) -> CompositeParameter:
        return self._make_composite_parameter(self, other, '-')

    def __rsub__(self, other: ParameterBase) -> CompositeParameter:
        return self._make_composite_parameter(other, self, '-')

    def __mul__(self, other: ParameterBase) -> CompositeParameter:
        return self._make_composite_parameter(self, other, '*')

    def __rmul__(self, other: ParameterBase) -> CompositeParameter:
        return self._make_composite_parameter(other, self, '*')

    def __truediv__(self, other: ParameterBase) -> CompositeParameter:
        return self._make_composite_parameter(self, other, '/')

    def __rtruediv__(self, other: ParameterBase) -> CompositeParameter:
        return self._make_composite_parameter(other, self, '/')

    def __pow__(self, other: ParameterBase) -> CompositeParameter:
        return self._make_composite_parameter(self, other, '**')

    def __rpow__(self, other: ParameterBase) -> CompositeParameter:
        return self._make_composite_parameter(other, self, '**')

    @staticmethod
    def _make_composite_parameter(
        lhs: ParameterBase,
        rhs: ParameterBase,
        op: Literal['+', '-', '*', '/', '**'],
    ) -> CompositeParameter:
        # check if the type of lhs and rhs are both parameter
        if not (
            isinstance(lhs, ParameterBase) and isinstance(rhs, ParameterBase)
        ):
            raise TypeError(
                f'unsupported operand types for {op}: '
                f"'{type(lhs).__name__}' and '{type(rhs).__name__}'"
            )

        if op == '+':
            op = jnp.add
            op_name = '%s + %s'
            op_latex = '{%s} + {%s}'
        elif op == '-':
            op = jnp.subtract
            op_name = '%s - %s'
            op_latex = '{%s} - {%s}'
        elif op == '*':
            op = jnp.multiply
            op_name = '%s * %s'
            op_latex = r'{%s} \times {%s}'
        elif op == '/':
            op = jnp.divide
            op_name = '%s / %s'
            op_latex = r'{%s} / {%s}'
        elif op == '**':
            op = jnp.power
            op_name = '%s^%s'
            op_latex = r'{%s}^{%s}'
        else:
            raise NotImplementedError(f'op {op}')

        return CompositeParameter(
            params=[lhs, rhs], op=op, op_name=op_name, op_latex=op_latex
        )


class _Parameter(ParameterBase, metaclass=ABCMeta):
    """Handle name, latex, and default value of a parameter."""

    def __init__(
        self,
        name: str,
        latex: str,
        fn: SampleFn,
        default: Any | None = None,
    ):
        if latex is None:
            latex = name

        self._name = name
        self.latex = latex

        if default is not None:
            self.default = default

        super().__init__(fn)

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


class Parameter(_Parameter):
    """Parameter definition given a distribution.

    Parameters
    ----------
    name : str
        Parameter name.
    dist : Distribution
        Numpyro distribution to which the parameter is sampled.
    default : float
        Parameter default value.
    min : float, optional
        Parameter minimum value. The default is None.
    max : float, optional
        Parameter maximum value. The default is None.
    log : bool, optional
        Whether the parameter is parameterized in a logarithmic scale.
        The default is False.
    fixed : bool, optional
        Whether the parameter is fixed. The default is False.
    latex : str, optional
        LaTex presentation of the parameter. The default is as `name`.

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
            raise ValueError('dist must be a numpyro distribution')

        if jnp.shape(default) != ():
            raise ValueError('default must be a scalar')

        if not bool(dist._validate_sample(default)):
            raise ValueError('default should be within the dist support')

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

        def fn(name: str, rng_key: PRNGKey | None) -> JAXFloat:
            """Sample from the distribution."""
            if self.fixed:
                return self._default
            else:
                return numpyro.sample(name, self._dist, rng_key=rng_key)

        super().__init__(name, latex, fn, default)

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
    def min(self) -> JAXFloat | None:
        """Minimum value of the parameter."""
        return self._min

    @property
    def max(self) -> JAXFloat | None:
        """Maximum value of the parameter."""
        return self._max

    @property
    def log(self) -> bool:
        return self._log

    @property
    def fixed(self) -> bool:
        return self._fixed


class UniformParameter(Parameter):
    """Define the parameter by a uniform distribution.

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
    fixed : bool
        Whether the parameter is fixed. The default is False.
    latex : str
        LaTeX presentation of the parameter. The default is as `name`.

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
        return self._min

    @min.setter
    def min(self, min: float):
        self._check_and_set_values(min=min)

    @property
    def max(self) -> JAXFloat:
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
            raise ValueError(f'min ({_min}) must less than max ({_max})')

        if _default <= _min:
            raise ValueError(
                f'default ({_default}) must greater than min ({_min})'
            )

        if _default >= _max:
            raise ValueError(
                f'default ({_default}) must less than max ({_max})'
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


class ConstantParameter(_Parameter, metaclass=ABCMeta):
    """Constant parameter."""

    def __init__(
        self, name: str, value: Any, fn: Callable, latex: str | None = None
    ):
        super().__init__(name, latex, fn, value)

    @property
    def log(self) -> bool:
        return False

    @property
    def fixed(self) -> bool:
        return True


class ConstantValue(ConstantParameter):
    """Parameter with a fixed value.

    Parameters
    ----------
    name: str
        Parameter name.
    value: float
        Parameter value.
    latex : str, optional
        LaTeX presentation of the parameter. The default is as `name`.

    """

    def __init__(self, name: str, value: float, latex: str | None = None):
        super().__init__(name, value, lambda _, __: self.default, latex)

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


class ConstantInterval(ConstantParameter):
    """Constant parameter to be integrated over a given interval."""

    def __init__(
        self,
        name: str,
        interval: Sequence[float],
        latex: str | None = None,
        method: Literal['gk', 'cc', 'ts', 'romberg', 'rombergts'] = 'gk',
        **kwargs: dict,
    ):
        def fn(_, __) -> JAXFloat:
            """Get a value lying within the interval from ParameterContext."""
            ctx = ParameterContext.get_context(stack_name=self._id)
            if ctx is None:
                raise RuntimeError('cannot sample from an interval')
            return ctx.params[self._id]

        super().__init__(name, interval, fn, latex)

        method = str(method)
        supported = ['gk', 'cc', 'ts', 'romberg', 'rombergts']
        if method not in supported:
            raise ValueError(f'method must be one of {supported}')
        self._method = method
        self._kwargs = kwargs

    def __repr__(self) -> str:
        return f'{self.name} = [{self.default[0]:.4g}, {self.default[1]:.4g}]'

    @property
    def default(self) -> JAXArray:
        return self._default

    @default.setter
    def default(self, default):
        if jnp.shape(default) != (2,):
            print(jnp.shape(default))
            raise ValueError('interval must be a 2-element sequence')

        self._default = jnp.asarray(default, float)


# class DependentInterval(ParameterBase):
#     """Interval defined by functions of a parameter."""


class CompositeParameter(ParameterBase):
    """Compose parameters into a new parameter.

    Parameters
    ----------
    params : Parameter, or sequence of Parameter
        Parameters to be composed.
    op : callable
        Function to be applied to `params`.
    op_name : str
        Name of the composition operator `op`.
    op_latex : str, optional
        LaTeX presentation of the composition operator `op`. The default is as
        `op_name`.

    """

    _params: tuple[ParameterBase, ...]
    _intervals: tuple[ConstantInterval, ...]
    _name: str
    _latex: str

    def __init__(
        self,
        params: ParameterBase | Sequence[ParameterBase],
        op: Callable[..., JAXFloat],
        op_name: str,
        op_latex: str,
    ):
        # check if the type of params is parameter or sequence
        if not isinstance(params, (ParameterBase, Sequence)):
            raise TypeError(
                'parameters must be a Parameter or a sequence of Parameter'
            )

        # make params a list
        if isinstance(params, ParameterBase):
            params = [params]
        else:
            params = list(params)

        # check if params are all parameters
        if not all(isinstance(i, ParameterBase) for i in params):
            raise TypeError(
                'parameters must be a Parameter or a sequence of Parameter'
            )

        self._params = tuple(params)
        self._op = op

        intervals = []
        for p in self._params:
            if isinstance(p, ConstantInterval) and p not in intervals:
                intervals.append(p)
            if isinstance(p, CompositeParameter):
                for i in p._intervals:
                    if i not in intervals:
                        intervals.append(i)
        self._intervals = tuple(intervals)

        op_name = str(op_name)
        if op_latex is None:
            op_latex = op_name
        else:
            op_latex = str(op_latex)
        self._name = op_name % tuple(
            f'({p._name})' if isinstance(p, CompositeParameter) else p._id
            for p in self._params
        )
        self._latex = op_latex % tuple(
            f'({p._latex})' if isinstance(p, CompositeParameter) else p._id
            for p in self._params
        )

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
        self._name_mapping = self._get_mapping('name')

        def fn(name: str | None, rng_key: PRNGKey | None) -> JAXFloat:
            """Get the value of the composite parameter."""
            values = [i.sample(rng_key) for i in self._params]
            new_value = self._op(*values)
            if name is not None:
                numpyro.deterministic(name, new_value)
            return new_value

        super().__init__(fn)

    def _get_mapping(self, label_type: Literal['name', 'latex']):
        namespace = []
        labels = []
        nprime = []
        counter = {}
        for node in self._nodes:
            label = getattr(node, label_type)
            labels.append(label)
            if label not in namespace:
                counter[label] = 0
                namespace.append(label)
            else:
                counter[label] += 1
                namespace.append(f'{label}#{counter[label]}')

            nprime.append(counter[label])

        if label_type == 'name':
            primes = ['"' * (n // 2) + "'" * (n % 2) for n in nprime]
        else:
            primes = ["'" * n for n in nprime]

        mapping = {
            node._id: label + suffix
            for node, label, suffix in zip(self._nodes, labels, primes)
        }

        return mapping

    @property
    def name(self) -> str:
        name = self._name
        for k, v in self._name_mapping.items():
            name = name.replace(k, v)
        return name

    @property
    def latex(self) -> str:
        latex = self._latex
        for k, v in self._get_mapping('latex').items():
            latex = latex.replace(k, v)
        return latex

    @property
    def default(self) -> JAXFloat:
        if self._intervals:
            raise RuntimeError(
                'cannot get default value of a composite interval'
            )

        return self._op(*[i.default for i in self._params])

    @property
    def log(self) -> bool:
        return any(i.log for i in self._params)

    @property
    def fixed(self) -> bool:
        return all(i.fixed for i in self._params)


class GPParameter(_Parameter):
    """Parameter sampled from a Gaussian process."""

    def __init__(
        self,
        name: str,
        kernel: kernels.Kernel,
        x: JAXFloat,
        *,
        diag: JAXFloat | None = None,
        noise: noise.Noise | None = None,
        mean: means.MeanBase | Callable | JAXFloat | None = None,
        log: bool = False,
        latex: str | None = None,
    ):
        self._log = bool(log)
        raise NotImplementedError

    @property
    def default(self):
        raise RuntimeError('cannot get default value of a GPParameter')

    @property
    def log(self) -> bool:
        return self._log

    @property
    def fixed(self) -> bool:
        return False


# if __name__ == '__main__':
#     import jax
#     from numpyro.infer import MCMC, NUTS
#     numpyro.enable_x64(True)
#     numpyro.set_host_device_count(4)
#
#     a = UniformParameter('a', 2.0, 0.1, 5, log=True)
#     b = UniformParameter('b', 2.0, 0.1, 5, log=True)
#     c = a + b
#     def numpyro_model():
#         with ParameterContext({a._id: 'a', b._id: 'b', c._id: 'c'}):
#             c.sample()
#
#     sampler = MCMC(
#         NUTS(numpyro_model),
#         num_warmup=1000, num_samples=1000, num_chains=4)
#
#     sampler.run(jax.random.PRNGKey(0))
#     import arviz as az
#     idata = az.from_numpyro(sampler)
#     params = {}
#
#     def set(a):
#         params['a'] = a
#
#     from functools import partial
#
#     def powerlaw(e, alpha, K):
#         tmp = 1.0 - alpha
#         f = K / tmp * jnp.power(e, tmp)
#         return f[1:] - f[:-1]
#
#     def alpha():
#         return params['a']*params['t'] + params['b']
#
#     def K():
#         return params['K']
#
#     def set_t(t):
#         params['t'] = t
#
#     def eval(e):
#         return powerlaw(e, alpha(), K())
#
#     from quadax import quadgk
#
#     def interal(e, a, b, K):
#         params['a'] = a
#         params['b'] = b
#         params['K'] = K
#
#         def integrand(t):
#             params['t'] = t
#             return eval(e)
#
#         return quadgk(integrand, (0.1, 2.1))
#
#     f = jax.jit(interal)
#
