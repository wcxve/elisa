"""The spectral model."""
from __future__ import annotations

import inspect
from abc import ABCMeta, abstractmethod
from typing import Any, Literal

import jax.numpy as jnp

from elisa.model.core.parameter import ParameterBase
from elisa.util.typing import Array, JAXArray

ParamConfig = tuple[tuple[str, str, str, float, float, float, bool, bool], ...]
ExtraKw = tuple[tuple[str, Any], ...]


# __all__ = [
#     'ModelBase',
#     'Model',
#     'CompositeModel',
#     'Component',
# ]


class ModelBase(metaclass=ABCMeta):
    """Base model class."""

    def __init__(self):
        self._id = hex(id(self))[2:]

    @abstractmethod
    def eval(self, egrid: Array, *args, **kwargs) -> JAXArray:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name."""
        pass

    @property
    @abstractmethod
    def latex(self) -> str:
        """LaTeX representation of the model."""
        pass

    @property
    @abstractmethod
    def type(self) -> Literal['add', 'mul']:
        """Model type."""
        pass

    def __repr__(self) -> str:
        return self.name

    def __add__(self, other: ModelBase) -> CompositeModel:
        return CompositeModel(self, other, '+')

    def __radd__(self, other: ModelBase) -> CompositeModel:
        return CompositeModel(other, self, '+')

    def __mul__(self, other: ModelBase) -> CompositeModel:
        return CompositeModel(self, other, '*')

    def __rmul__(self, other: ModelBase) -> CompositeModel:
        return CompositeModel(other, self, '*')


class Model(ModelBase):
    def __init__(self, component: Component):
        self._component = component
        self._name = str(component.name)
        self._latex = str(component.latex)
        self._type = component.type
        self._fn = component.eval
        super().__init__()

    def eval(self, egrid: Array, *args, **kwargs) -> JAXArray:
        pass

    @property
    def name(self) -> str:
        """Model name."""
        return self._name

    @property
    def latex(self) -> str:
        """LaTeX representation of the model."""
        return self._latex

    @latex.setter
    def latex(self, latex: str):
        self._latex = str(latex)

    @property
    def type(self) -> Literal['add', 'mul']:
        """Model type."""
        return self._type


class CompositeModel(ModelBase):
    _name: str
    _latex: str
    _type: Literal['add', 'mul']

    def __init__(self, lhs: ModelBase, rhs: ModelBase, op: Literal['+', '*']):
        # check if the type of lhs and rhs are both model
        if not (isinstance(lhs, ModelBase) and isinstance(rhs, ModelBase)):
            raise TypeError(
                f'unsupported operand types for {op}: '
                f"'{type(lhs).__name__}' and '{type(rhs).__name__}'"
            )

        self._lhs = lhs
        self._rhs = rhs
        self._op_str = op

        # check if operand is legal for the operator
        type1 = lhs.type
        type2 = rhs.type

        if op == '+':
            if type1 != 'add':
                raise TypeError(f'{lhs} is not an additive model')
            if type2 != 'add':
                raise TypeError(f'{rhs} is not an additive model')

            self._type = 'add'
            self._op = jnp.add
            op_name = '%s + %s'
            op_latex = '{%s} + {%s}'

        elif op == '*':
            if 'add' == type1 == type2:
                raise TypeError(
                    f'unsupported model type for *: {lhs} (additive) '
                    f'and {rhs} (additive)'
                )

            self._type = 'add' if type1 == 'add' or type2 == 'add' else 'mul'
            self._op = jnp.multiply
            op_name = '%s * %s'
            op_latex = r'{%s} \times {%s}'

        else:
            raise NotImplementedError(f'op {op}')

        if isinstance(lhs, CompositeModel):
            lhs_name = lhs._name
            lhs_latex = lhs._latex
            if op == '*' and lhs._op_str == '+':
                lhs_name = f'({lhs_name})'
                lhs_latex = f'({lhs_latex})'
        else:
            lhs_name = lhs._id
            lhs_latex = lhs._id

        if isinstance(rhs, CompositeModel):
            rhs_name = rhs._name
            rhs_latex = rhs._latex
            if op == '*' and rhs._op_str == '+':
                rhs_name = f'({rhs_name})'
                rhs_latex = f'({rhs_latex})'
        else:
            rhs_name = rhs._id
            rhs_latex = rhs._id

        self._name = op_name % (lhs_name, rhs_name)
        self._latex = op_latex % (lhs_latex, rhs_latex)

        nodes = []
        for m in [lhs, rhs]:
            stack = [m]
            while stack:
                node = stack.pop(0)
                if isinstance(node, CompositeModel):
                    stack = [node._lhs, node._rhs] + stack
                elif node not in nodes:
                    nodes.append(node)
        self._nodes = tuple(nodes)
        self._name_mapping = self._get_mapping('name')

        super().__init__()

    def _get_mapping(self, label_type: Literal['name', 'latex']):
        namespace = []
        labels = []
        suffixes = []
        counter = {}
        for node in self._nodes:
            label = getattr(node, label_type)
            labels.append(label)
            if label not in namespace:
                counter[label] = 1
                namespace.append(label)
            else:
                counter[label] += 1
                namespace.append(f'{label}#{counter[label]}')

            suffixes.append(counter[label])

        template = '_%s' if label_type == 'name' else '_{%s}'
        suffixes = [template % n if n > 1 else '' for n in suffixes]

        mapping = {
            node._id: label + suffix
            for node, label, suffix in zip(self._nodes, labels, suffixes)
        }

        return mapping

    def eval(self, egrid: Array, *args, **kwargs) -> JAXArray:
        ...

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

    def type(self) -> Literal['add', 'mul']:
        return self._type


class ComponentMeta(ABCMeta):
    """Avoid cumbersome coding for subclass ``__init__``."""

    def __new__(cls, name, bases, dct, **kwargs) -> ComponentMeta:
        # check config and then define subclass __init__ method
        if isinstance(config := dct.get('_config', None), tuple):
            if any(len(cfg) != 8 for cfg in config):
                raise ValueError(
                    f'each parameter configuration of {name} should have '
                    '8 values: name (str), latex (str), unit (str), '
                    'default (float), min (float), max (float), '
                    'log (bool), fixed (bool)'
                )

            init_def = 'self, '
            init_body = ''
            for cfg in config:
                init_def += f'{cfg[0]}: ParameterBase | None = None, '
                init_body += f'{cfg[0]}={cfg[0]}, '

            init_def += 'latex: str | None = None'
            init_body += 'latex=latex'

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
            func_code += 'super(type(self), type(self))'
            func_code += f'.__init__(self, {init_body})\n'

            exec(func_code, tmp := {'ParameterBase': ParameterBase})
            __init__ = tmp['__init__']
            __init__.__qualname__ = f'{name}.__init__'
            dct['__init__'] = __init__

        return super().__new__(cls, name, bases, dct)

    def __init__(cls, name, bases, dct, **kwargs) -> None:
        super().__init__(name, bases, dct, **kwargs)
        sig = inspect.signature(cls.__init__)
        parameters = tuple(sig.parameters.values())
        cls.__signature__ = sig.replace(parameters=parameters[1:])

    def __call__(cls, *args, **kwargs) -> Model:
        # return Model object after Component initialization
        return Model(super().__call__(*args, **kwargs))


class Component(metaclass=ComponentMeta):
    def __init__(self, latex: str | None = None, **params) -> None:
        name = self.__class__.__name__

        if latex is None:
            latex = r'\mathrm{%s}' % name

        self.name = name.lower()
        self.latex = str(latex)
        self.params = params

    @property
    @abstractmethod
    def _config(self) -> ParamConfig:
        """Configuration of parameters."""
        pass

    @property
    def _extra_kw(self) -> ExtraKw:
        """Extra keywords passed to ``__init__`` method.

        Note that element of inner tuple should respect :func:`repr`.

        """
        return ()

    @staticmethod
    @abstractmethod
    def eval(*args, **kwargs) -> JAXArray:
        pass

    @property
    @abstractmethod
    def type(self) -> Literal['add', 'mul']:
        pass


class AdditiveComponent(Component, metaclass=ABCMeta):
    """Prototype class to define additive component."""

    def eval(self, egrid: Array, *args, **kwargs) -> JAXArray:
        return self.integrate(egrid, *args, **kwargs)

    @abstractmethod
    def integrate(self, *args, **kwargs) -> JAXArray:
        pass

    @property
    def type(self) -> Literal['add']:
        """Model type is additive."""
        return 'add'


class NumIntAdditive(AdditiveComponent, metaclass=ABCMeta):
    _extra_kw = (('method', 'trapz'),)

    def __init__(self, method='trapz', **kwargs):
        self.method = method
        super().__init__(**kwargs)

    @staticmethod
    @abstractmethod
    def continnum(egrid: Array, *args, **kwargs) -> JAXArray:
        pass

    def integrate(self, egrid: Array, *args, **kwargs) -> JAXArray:
        if self.method == 'trapz':
            de = egrid[1:] - egrid[:-1]
            f_grid = self.continnum(egrid, *args, **kwargs)
            return 0.5 * (f_grid[:-1] + f_grid[1:]) * de

        elif self.method == 'simpson':
            de = egrid[1:] - egrid[:-1]
            e_mid = (egrid[:-1] + egrid[1:]) / 2.0
            f_grid = self.continnum(egrid, *args, **kwargs)
            f_mid = self.continnum(e_mid, *args, **kwargs)
            return de / 6.0 * (f_grid[:-1] + 4.0 * f_mid + f_grid[1:])

        else:
            raise NotImplementedError(f'integration method "{self.method}"')

    @property
    def method(self) -> str:
        """Numerical integration method."""
        return self._method

    @method.setter
    def method(self, method: str):
        method = str(method)

        if method not in ('trapz', 'simpson'):
            raise ValueError(
                f'available integration methods are "trapz" and "simpson", '
                f'but got "{method}"'
            )

        self._method = method


class Powerlaw(AdditiveComponent):
    r"""The power law function, defined as

    .. math::
        N(E) = K \left(\frac{E}{E_\mathrm{pivot}}\right)^{-\alpha},

    where :math:`E_\mathrm{pivot}` is the pivot energy fixed at 1 keV.

    Parameters
    ----------
    alpha : ParameterBase
        The power law photon index, dimensionless.
    K : ParameterBase
        The normalization at 1 keV, in units of :math:`\mathrm{cm}^{-2} \,
        \mathrm{s}^{-1} \, \mathrm{keV}^{-1}`.
    latex : str, optional
        LaTeX representation of the model. The default is as its class name.

    """

    _config = (
        ('alpha', r'\alpha', '', -1.01, -10.0, 3.0, False, False),
        ('K', 'K', 'cm^-2 s^-1 keV^-1', 1.0, 1e-10, 1e10, False, False),
    )

    @staticmethod
    def integrate(egrid, alpha, K) -> JAXArray:
        # we ignore the case of alpha = 1.0
        tmp = 1.0 - alpha
        f = K / tmp * jnp.power(egrid, tmp)
        return f[1:] - f[:-1]


class Cutoffpl(NumIntAdditive):
    r"""The cut-off power law function, defined as

    .. math::
        N(E) = K \left(\frac{E}{E_\mathrm{pivot}}\right)^{-\alpha}
                \exp\left(-\frac{E}{E_\mathrm{c}}\right),

    where :math:`E_\mathrm{pivot}` is the pivot energy fixed at 1 keV.

    Parameters
    ----------
    alpha : ParameterBase, optional
        The power law photon index, dimensionless.
    Ec : ParameterBase, optional
        The folding energy of exponential rolloff, in units of keV.
    K : ParameterBase, optional
        The normalization at 1 keV, in units of :math:`\mathrm{cm}^{-2} \,
        \mathrm{s}^{-1} \, \mathrm{keV}^{-1}`.
    latex : str, optional
        LaTeX representation of the model. The default is as its class name.

    """

    _config = (
        ('alpha', r'\alpha', '', 1.0, -3.0, 10.0, False, False),
        ('Ec', r'E_\mathrm{c}', 'keV', 15.0, 0.01, 10000.0, False, False),
        ('K', 'K', 'cm^-2 s^-1 keV^-1', 1.0, 1e-10, 1e10, False, False),
    )

    @staticmethod
    def _continnum(egrid, alpha, Ec, K) -> JAXArray:
        e = egrid
        return K * jnp.power(e, alpha) * jnp.exp(-e / Ec)
