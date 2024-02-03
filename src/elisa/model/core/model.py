"""The spectral model bases."""
from __future__ import annotations

import inspect
from abc import ABCMeta, abstractmethod
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Union

import jax
import jax.numpy as jnp

from elisa.model.core.parameter import ParameterBase, UniformParameter
from elisa.util.typing import Array, Float, JAXArray, JAXFloat, NumPyArray

ComponentID = str
ComponentName = str
ParamName = str
ParamsMappingByName = dict[ComponentID, dict[ParamName, Union[Float, Array]]]
ParamsFloat = dict[ComponentID, dict[ParamName, JAXFloat]]
ParamsArray = dict[ComponentID, dict[ParamName, JAXArray]]
ParamsMappingByID = Union[ParamsFloat, ParamsArray]


class ModelBase(metaclass=ABCMeta):
    r"""Base model class.

    Actual model evaluation is performed under `_eval` method, which must be
    overriden by subclass.

    Support subscription to get a component by name, or a parameter by index.

    Informative repr shows the component name, :math:`LaTeX` format, type,
    and parameters.

    """

    _name_to_id: dict[ComponentName, ComponentID]
    _id_to_name: dict[ComponentID, ComponentName]
    _params: tuple[tuple[ComponentName, ParamName], ...]
    __initialized: bool = False

    def __init__(self, mapping: dict[ComponentName, ComponentBase]):
        self._name_to_id = {k: v._id for k, v in mapping.items()}
        self._id_to_name = {v: k for k, v in self._name_to_id.items()}
        self._params = tuple(
            (k, p) for k, v in mapping.items() for p in v.param_names
        )
        for k, v in mapping.items():
            setattr(self, k, v)
        self.__initialized = True

    @abstractmethod
    def _eval(self, egrid: JAXArray, params: ParamsFloat) -> JAXArray:
        """Evaluate the model.

        Note that `params` is a mapping from model id to parameter dict.
        This method should read its parameters from `params` according to
        component id, and return the model value depend on the parameters.

        """
        pass

    @property
    @abstractmethod
    def _name_by_id(self) -> str:
        """Model name by id."""
        pass

    @property
    @abstractmethod
    def _latex_by_id(self) -> str:
        r""":math:`\LaTeX` format by id."""
        pass

    def _validate_params(
        self, params: ParamsMappingByName
    ) -> tuple[ParamsMappingByID, tuple[int, ...]]:
        """Check if `params` is valid for the model."""
        params = {k: params[v] for k, v in self._id_to_name.items()}

        shapes = jax.tree_util.tree_flatten(
            tree=jax.tree_map(jnp.shape, params),
            is_leaf=lambda i: isinstance(i, tuple),
        )[0]

        if not shapes:
            raise ValueError('params are empty')

        shape = shapes[0]
        if any(s != shape for s in shapes[1:]):
            raise ValueError('all params must have the same shape')

        return params, shape

    def eval(self, egrid: Array, params: ParamsMappingByName) -> JAXArray:
        """Evaluate the model.

        Parameters
        ----------
        egrid : ndarray
            Photon energy grid in units of keV.
        params : dict
            Parameter dict for the model.

        Returns
        -------
        value : JAXArray
            The model value.

        """
        params, shape = self._validate_params(params)

        f = self._eval

        # iteratively vmap over params dimensions
        for _ in range(len(shape)):
            f = jax.vmap(f, in_axes=(None, 0))

        return f(jnp.asarray(egrid), params)

    def ne(
        self,
        egrid: Array,
        params: ParamsMappingByName,
        comps: bool = False,
    ) -> JAXArray | dict[str, JAXArray]:
        r"""Calculate :math:`N(E)`.

        Parameters
        ----------
        egrid : ndarray
            Photon energy grid in units of keV.
        params : dict
            Parameter dict for the model.
        comps : bool, optional
            Whether to return the result of each component. Defaults to False.

        Returns
        -------
        ne : JAXArray, or dict[str, JAXArray]
            The differential photon flux in units of :math:`\mathrm{cm}^{-2} \,
            \mathrm{s}^{-1} \, \mathrm{keV}^{-1}`.

        """
        if self.type != 'add':
            msg = f'ne is undefined for {self.type} type model "{self}"'
            raise TypeError(msg)

    def ene(  # noqa: B027
        self,
        egrid: Array,
        params: dict[str, dict[str, float | Array]],
        comps: bool = False,
    ) -> JAXArray | dict[str, JAXArray]:
        r"""Calculate :math:`E N(E)`, i.e. :math:`F(\nu)`.

        Parameters
        ----------
        egrid : ndarray
            Photon energy grid in units of keV.
        params : dict
            Parameter dict for the model.
        comps : bool, optional
            Whether to return the result of each component. Defaults to False.

        Returns
        -------
        ene : JAXArray, or dict[str, JAXArray]
            The differential energy flux in units of :math:`\mathrm {erg} \,
            \mathrm{cm}^{-2} \, \mathrm{s}^{-1} \, \mathrm{keV}^{-1}`.

        """
        pass

    def eene(  # noqa: B027
        self,
        egrid: Array,
        params: dict[str, dict[str, float | Array]],
        comps: bool = False,
    ) -> JAXArray | dict[str, JAXArray]:
        r"""Calculate :math:`E^2 N(E)`, i.e. :math:`\nu F(\nu)`.

        Parameters
        ----------
        egrid : ndarray
            Photon energy grid in units of keV.
        params : dict
            Parameter dict for the model.
        comps : bool, optional
            Whether to return the result of each component. Defaults to False.

        Returns
        -------
        eene : JAXArray or dict[str, JAXArray]
            The energy flux in units of :math:`\mathrm {erg} \,
            \mathrm{cm}^{-2} \, \mathrm{s}^{-1}`.

        """
        pass

    def ce(  # noqa: B027
        self,
        egrid: Array,
        params: dict[str, dict[str, float | Array]],
        resp_matrix: Array,
        ch_width: Array,
        comps: bool = False,
    ) -> JAXArray | dict[str, JAXArray]:
        r"""Calculate the folded model, i.e. :math:`C(E)`.

        Parameters
        ----------
        egrid : ndarray
            Photon energy grid of `resp_matrix`, in units of keV.
        params : dict
            Parameter dict for the model.
        resp_matrix : ndarray
            Instrumental response matrix used to fold the model.
        ch_width : ndarray
            Measured energy channel width of `resp_matrix`.
        comps : bool, optional
            Whether to return the result of each component. Defaults to False.

        Returns
        -------
        ce : JAXArray or dict[str, JAXArray]
            The folded model in units of :math:`\mathrm{s}^{-1} \,
            \mathrm{keV}^{-1}`.

        """
        pass

    def flux(  # noqa: B027
        self,
        params: dict[str, dict[str, float | Array]],
        emin: float | int | JAXFloat,
        emax: float | int | JAXFloat,
        energy: bool = True,
        comps: bool = False,
        ngrid: int = 1000,
        log: bool = True,
    ) -> jax.Array | dict[str, jax.Array]:
        r"""Calculate the flux of model between `emin` and `emax`.

        Parameters
        ----------
        params : dict
            Parameter dict for the model.
        emin : float or int
            Minimum value of energy range, in units of keV.
        emax : float or int
            Maximum value of energy range, in units of keV.
        energy : bool, optional
            Calculate energy flux in units of :math:`\mathrm {erg} \,
            \mathrm{cm}^{-2} \, \mathrm{s}^{-1}` when True. Calculate photon
            flux in units of :math:`\mathrm{cm}^{-2} \, \mathrm{s}^{-1}` when
            False. The default is True.
        comps : bool, optional
            Whether to return the result of each component. Defaults to False.
        ngrid : int, optional
            The energy grid number to use in integration. The default is 1000.
        log : bool, optional
            Whether to use logarithmically regular energy grid. The default is
            True.

        Returns
        -------
        flux : JAXArray or dict[str, JAXArray]
            The model flux.

        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name."""
        pass

    @property
    @abstractmethod
    def latex(self) -> str:
        r""":math:`\LaTeX` format of the model."""
        pass

    @property
    @abstractmethod
    def type(self) -> Literal['add', 'mul']:
        """Model type."""
        pass

    def __delattr__(self, key: str):
        if self.__initialized and hasattr(self, key):
            raise AttributeError("can't delete attribute")

        super().__delattr__(key)

    def __setattr__(self, key: str, value: Any):
        if self.__initialized and (
            not hasattr(self, key) or key in self._name_to_id
        ):
            raise AttributeError("can't set attribute")

        super().__setattr__(key, value)

    @singledispatchmethod
    def __getitem__(self, key):
        raise TypeError(f'unsupported key {key} ({type(key).__name__})')

    @__getitem__.register(str)
    def _(self, key: str) -> ComponentBase:
        if key not in self._name_to_id:
            raise KeyError(key)

        return getattr(self, key)

    @__getitem__.register(int)
    def _(self, key: int) -> ParameterBase:
        if key < 0 or key >= len(self._params):
            raise IndexError(key)

        cname, pname = self._params[key]
        return getattr(getattr(self, cname), pname)

    @singledispatchmethod
    def __setitem__(self, key, value):
        typ = type(key).__name__
        raise TypeError(f'unsupported key {key} ({typ}) for item assignment')

    @__setitem__.register(int)
    def _(self, key: int, value: Any):
        if key < 0 or key >= len(self._params):
            raise IndexError(key)

        cname, pname = self._params[key]
        setattr(getattr(self, cname), pname, value)

    def __repr__(self) -> str:
        return self.name

    def _repr_html_(self) -> str:
        return self.name

    def __add__(self, other: ModelBase) -> CompositeModel:
        return CompositeModel(self, other, '+')

    def __radd__(self, other: ModelBase) -> CompositeModel:
        return CompositeModel(other, self, '+')

    def __mul__(self, other: ModelBase) -> CompositeModel:
        return CompositeModel(self, other, '*')

    def __rmul__(self, other: ModelBase) -> CompositeModel:
        return CompositeModel(other, self, '*')

    del _


class Model(ModelBase):
    """Model defined by a single component."""

    @property
    def _name_by_id(self) -> str:
        return self._component_id

    @property
    def _latex_by_id(self) -> str:
        return self._component_id

    def __init__(self, component: ComponentBase):
        self._component = component
        self._component_id = component._id
        self._component_eval = component._eval
        super().__init__({component.name: component})

    def _eval(self, egrid: JAXArray, params: ParamsFloat) -> JAXArray:
        params = params[self._component_id]
        return self._component_eval(egrid, params)

    @property
    def name(self) -> str:
        return self._component.name

    @property
    def latex(self) -> str:
        return self._component.latex

    @latex.setter
    def latex(self, latex: str):
        self._component.latex = latex

    @property
    def type(self) -> Literal['add', 'mul']:
        return self._component.type


class CompositeModel(ModelBase):
    """Model defined by sum or multiplication of two components."""

    _type: Literal['add', 'mul']
    _operands: tuple[ModelBase, ModelBase]
    _op_str: Literal['+', '*']

    def __init__(self, lhs: ModelBase, rhs: ModelBase, op: Literal['+', '*']):
        # check if the type of lhs and rhs are both model
        if not (isinstance(lhs, ModelBase) and isinstance(rhs, ModelBase)):
            raise TypeError(
                f'unsupported operand types for {op}: '
                f"'{type(lhs).__name__}' and '{type(rhs).__name__}'"
            )

        self._operands = (lhs, rhs)
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

        lhs_name = lhs._name_by_id
        lhs_latex = lhs._latex_by_id
        rhs_name = rhs._name_by_id
        rhs_latex = rhs._latex_by_id

        if op == '*':
            if isinstance(lhs, CompositeModel) and lhs._op_str == '+':
                lhs_name = f'({lhs_name})'
                lhs_latex = f'({lhs_latex})'

            if isinstance(rhs, CompositeModel) and rhs._op_str == '+':
                rhs_name = f'({rhs_name})'
                rhs_latex = f'({rhs_latex})'

        self.__name_by_id = op_name % (lhs_name, rhs_name)
        self.__latex_by_id = op_latex % (lhs_latex, rhs_latex)

        nodes = []
        for m in [lhs, rhs]:
            stack = [m]
            while stack:
                node = stack.pop(0)
                if hasattr(node, '_operands'):
                    stack = list(node._operands) + stack
                elif node not in nodes:
                    nodes.append(node)
        self._nodes: tuple[Model, ...] = tuple(nodes)
        self._name_mapping = self._get_mapping('name')

        mapping = {
            self._name_mapping[node._component_id]: node._component
            for node in self._nodes
        }
        super().__init__(mapping)

    def _get_mapping(
        self, label_type: Literal['name', 'latex']
    ) -> dict[str, str]:
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
            node._component_id: label + suffix
            for node, label, suffix in zip(self._nodes, labels, suffixes)
        }

        return mapping

    def _eval(self, egrid: Array, params: ParamsFloat) -> JAXArray:
        lhs = self._operands[0]._eval(egrid, params)
        rhs = self._operands[1]._eval(egrid, params)
        return self._op(lhs, rhs)

    @property
    def _name_by_id(self) -> str:
        return self.__name_by_id

    @property
    def _latex_by_id(self) -> str:
        return self.__latex_by_id

    @property
    def name(self) -> str:
        name = self._name_by_id
        for k, v in self._name_mapping.items():
            name = name.replace(k, v)
        return name

    @property
    def latex(self) -> str:
        latex = self._latex_by_id
        for k, v in self._get_mapping('latex').items():
            latex = latex.replace(k, v)
        return latex

    @property
    def type(self) -> Literal['add', 'mul']:
        return self._type


class ComponentMeta(ABCMeta):
    """Avoid cumbersome coding for subclass ``__init__``."""

    def __new__(metacls, name, bases, dct, **kwargs) -> ComponentMeta:
        config = dct.get('_config', None)

        # check config
        if isinstance(config, tuple):
            if not config:
                raise ValueError(f'empty parameter configurations for {name}')

            if not all(isinstance(cfg, ParamConfig) for cfg in config):
                raise ValueError(
                    f'each parameter config of {name} must be a ParamConfig'
                )

            for i, cfg in enumerate(config):
                dct[cfg.name] = property(
                    fget=_param_getter(cfg.name),
                    fset=_param_setter(cfg.name, i),
                    doc=f'{name} parameter :math:`{cfg.latex}`.',
                )

        # create the class
        cls = super().__new__(metacls, name, bases, dct)

        # define __init__ method of the newly created class if necessary
        if config is not None and '__init__' not in dct:
            # signature of __init__ method
            sig1 = ['self']
            sig1 += [
                f'{i[0]}: ParameterBase | float | None = None' for i in config
            ]
            sig1 += ['latex: str | None = None']

            params = '{%s}' % ', '.join(f"'{i[0]}': {i[0]}" for i in config)

            # signature of super().__init__ method
            sig2 = [f'params={params}', 'latex=latex']

            if args := getattr(cls, '_args', None):
                sig1 = [f'{i}' for i in args] + sig1
                sig2 = [f'{i}={i}' for i in args] + sig2

            if kwargs := getattr(cls, '_kwargs', None):
                sig1 = sig1 + [f'{i}=None' for i in kwargs]
                sig2 = [f'{i}={i}' for i in kwargs] + sig2

            func_code = (
                f'def __init__({", ".join(sig1)}):\n'
                f'    super(type(self), self).__init__({", ".join(sig2)})\n'
                f'__init__.__qualname__ = "{name}.__init__"'
            )
            exec(func_code, tmp := {'ParameterBase': ParameterBase})
            cls.__init__ = tmp['__init__']

        return cls

    def __init__(cls, name, bases, dct, **kwargs) -> None:
        super().__init__(name, bases, dct, **kwargs)

        # restore the signature of __init__ method
        # see https://stackoverflow.com/a/65385559
        signature = inspect.signature(cls.__init__)
        cls.__signature__ = signature.replace(
            parameters=tuple(signature.parameters.values())[1:],
            return_annotation=Model,
        )

    def __call__(cls, *args, **kwargs) -> Model:
        # return Model object after Component initialization
        return Model(super().__call__(*args, **kwargs))


def _param_getter(name: str):
    def _(self: ComponentBase):
        return getattr(self, f'_{name}')

    return _


def _param_setter(name: str, idx: int):
    def _(self: ComponentBase, param: Any):
        cfg = self._config[idx]

        if param is None:
            # use default configuration
            setattr(
                self,
                f'_{name}',
                UniformParameter(
                    name=cfg.name,
                    default=cfg.default,
                    min=cfg.min,
                    max=cfg.max,
                    log=cfg.log,
                    fixed=cfg.fixed,
                    latex=cfg.latex,
                ),
            )

        elif isinstance(param, (float, int, jnp.number, JAXArray, NumPyArray)):
            # fixed to the given value
            if jnp.shape(param) != ():
                raise ValueError('scalar value is expected')

            setattr(
                self,
                f'_{name}',
                UniformParameter(
                    name=cfg.name,
                    default=param,
                    min=cfg.min,
                    max=cfg.max,
                    log=cfg.log,
                    fixed=True,
                    latex=cfg.latex,
                ),
            )

        elif isinstance(param, ParameterBase):
            # given parameter instance
            setattr(self, f'_{name}', param)

        else:
            # other input types are not supported yet
            raise TypeError(
                f'{type(self).__name__}.{cfg.name} got unsupported value '
                f'{param} ({type(param).__name__})'
            )

    return _


class ParamConfig(NamedTuple):
    """Configuration of a uniform parameter."""

    name: str
    latex: str
    unit: str
    default: float
    min: float
    max: float
    log: bool = False
    fixed: bool = False


class ComponentBase(metaclass=ComponentMeta):
    """Base class to define a spectral component."""

    _args: tuple[str, ...] = ()  # extra args passed to subclass __init__
    _kwargs: tuple[str, ...] = ()  # extra kwargs passed to subclass __init__
    _config: tuple[ParamConfig, ...]
    _id: ComponentID
    __initialized: bool = False

    def __init__(self, params: dict, latex: str | None):
        self._id = hex(id(self))[2:]

        if latex is None:
            latex = r'\mathrm{%s}' % self.__class__.__name__
        self.latex = latex

        self._name = self.__class__.__name__.lower()

        # parse parameters from params, which is a dict of parameters
        for cfg in self._config:
            setattr(self, cfg.name, params[cfg.name])

        self._param_names = tuple(cfg.name for cfg in self._config)

        self.__initialized = True

    if TYPE_CHECKING:

        @abstractmethod
        def _eval(self, *args, **kwargs) -> JAXArray:
            pass

    else:

        @abstractmethod
        def _eval(
            self, egrid: Array, params: dict[str, float | JAXFloat]
        ) -> JAXArray:
            pass

    @property
    def name(self) -> str:
        """Component name."""
        return self._name

    @property
    def latex(self) -> str:
        r""":math:`\LaTeX` format of the component."""
        return self._latex

    @latex.setter
    def latex(self, latex: str):
        self._latex = str(latex)

    @property
    @abstractmethod
    def type(self) -> Literal['add', 'mul']:
        """Component type."""
        pass

    @property
    def param_names(self) -> tuple[str, ...]:
        """Component's parameter names."""
        return self._param_names

    @property
    def nparam(self) -> int:
        """Number of parameters."""
        return len(self._param_names)

    @property
    @abstractmethod
    def _config(self) -> tuple[ParamConfig, ...]:
        """Default configuration of parameters."""
        pass

    def __repr__(self) -> str:
        return self.name

    def __delattr__(self, key: str):
        if self.__initialized and hasattr(self, key):
            raise AttributeError("can't delete attribute")

        super().__delattr__(key)

    def __setattr__(self, key: str, value: Any):
        if self.__initialized and not hasattr(self, key):
            raise AttributeError("can't set attribute")

        super().__setattr__(key, value)

    @singledispatchmethod
    def __getitem__(self, key):
        raise TypeError(f'unsupported key {key} ({type(key).__name__})')

    @__getitem__.register(str)
    def _(self, key: str) -> ParameterBase:
        if key not in self.param_names:
            raise KeyError(key)

        return getattr(self, key)

    @__getitem__.register(int)
    def _(self, key: int) -> ParameterBase:
        if key < 0 or key >= self.nparam:
            raise IndexError(key)

        return getattr(self, self.param_names[key])

    @singledispatchmethod
    def __setitem__(self, key, value):
        typ = type(key).__name__
        raise TypeError(f'unsupported key {key} ({typ}) for item assignment')

    @__setitem__.register(str)
    def _(self, key: str, value: Any):
        if key not in self.param_names:
            raise KeyError(key)

        setattr(self, key, value)

    @__setitem__.register(int)
    def _(self, key: int, value: Any):
        if key < 0 or key >= self.nparam:
            raise IndexError(key)

        setattr(self, self.param_names[key], value)

    del _
