"""The spectral model bases."""

from __future__ import annotations

import inspect
from abc import ABCMeta, abstractmethod
from collections.abc import Mapping, Sequence
from enum import Enum
from functools import reduce
from typing import Any, Callable, Literal, NamedTuple

import jax
import jax.numpy as jnp
from numpyro.distributions import Distribution

from elisa.model.parameter import ParameterBase, ParamInfo, UniformParameter
from elisa.util.integrate import IntegralFactory
from elisa.util.misc import build_namespace
from elisa.util.typing import (
    AdditiveFn,
    ArrayLike,
    CompEval,
    CompID,
    CompIDParamValMapping,
    CompIDStrMapping,
    CompName,
    CompParamName,
    ConvolveEval,
    JAXArray,
    JAXFloat,
    ModelCompiledFn,
    ModelEval,
    NameValMapping,
    NumPyArray,
    ParamID,
    ParamIDStrMapping,
    ParamIDValMapping,
    T,
)


class ModelBase(metaclass=ABCMeta):
    """Base model class."""

    _comps: tuple[ComponentBase, ...]
    _additive_comps: tuple[ModelBase, ...]
    __initialized: bool = False

    def __init__(
        self,
        name: str,
        latex: str,
        comps: list[ComponentBase],
    ):
        self._name = str(name)
        self._latex = str(latex)
        self._comps = tuple(comps)
        self._comps_id = tuple(comp._id for comp in comps)

        cid_to_cname = dict(
            zip(self._comps_id, build_namespace([c.name for c in comps]))
        )

        self._cid_to_cname = cid_to_cname

        self.__name = self._id_to_label(cid_to_cname, 'name')

        for name, comp in zip(cid_to_cname.values(), comps):
            setattr(self, name, comp)

        self.__initialized = True

    def compile(
        self,
        comps: Sequence[ComponentBase, ...] | None = None,
        cid_to_cname: CompIDStrMapping | None = None,
    ) -> CompiledModel:
        """Compile the model for fast evaluation.

        Parameters
        ----------
        comps : sequence of ComponentBase, optional
            Components to compile.
        cid_to_cname : dict, optional
            Mapping from component id to name.

        Returns
        -------
        model : CompiledModel
            The compiled model.

        """
        if self.type == 'conv':
            raise RuntimeError('cannot compile convolution model')

        if comps is None and cid_to_cname is None:
            comps = self._comps
            cid_to_cname = self._cid_to_cname
        else:
            if comps is None:
                comps = self._comps
            else:
                if set(self._comps) - set(comps) != set():
                    raise ValueError('comps must contain all components')

                comps = tuple(comps)

            cid = [c._id for c in comps]

            if cid_to_cname is None:
                names = build_namespace([c.name for c in comps])
                cid_to_cname = dict(zip(cid, names))
            else:
                if set(cid) - set(cid_to_cname) != set():
                    raise ValueError('cid_to_cname must contain all component')

                cid_to_cname = dict(cid_to_cname)

        model_info = get_model_info(comps, cid_to_cname)

        fn = self._compile_model_fn(model_info)
        additive_fn = self._compile_additive_fn(model_info)

        return CompiledModel(self.name, fn, additive_fn, model_info, self.type)

    @property
    @abstractmethod
    def eval(self) -> ModelEval:
        """Get side-effect free model evaluation function."""
        pass

    @property
    def name(self) -> str:
        """Model name."""
        return self.__name

    @property
    def latex(self) -> str:
        r""":math:`\LaTeX` format of the model."""
        cid_to_latex = dict(
            zip(
                self._comps_id,
                build_namespace([c.latex for c in self._comps], latex=True),
            )
        )

        return self._id_to_label(cid_to_latex, 'latex')

    @property
    @abstractmethod
    def type(self) -> Literal['add', 'mul']:
        """Model type."""
        pass

    @property
    def comp_names(self) -> tuple[CompName, ...]:
        """Component names."""
        return tuple(self._cid_to_cname.values())

    @property
    def _additive_comps(self) -> tuple[ModelBase, ...]:
        if self.type != 'add':
            return ()
        else:
            return (self,)

    def _id_to_label(
        self,
        mapping: CompIDStrMapping,
        label_type: Literal['name', 'latex'],
    ) -> str:
        """Get the label of the parameter."""
        if label_type not in {'name', 'latex'}:
            raise ValueError(f'unknown label type: {label_type}')

        label = self._name if label_type == 'name' else self._latex

        for k, v in mapping.items():
            label = label.replace(k, v)

        return label

    def _compile_model_fn(self, model_info: ModelInfo) -> ModelCompiledFn:
        """Get the model evaluation function."""
        pid_to_value = {
            comp._id: model_info.cid_pval[comp._id] for comp in self._comps
        }

        eval_fn = self.eval

        @jax.jit
        def fn(egrid: JAXArray, params: ParamIDValMapping) -> JAXArray:
            """The model evaluation function"""
            comps_params = jax.tree_map(lambda f: f(params), pid_to_value)
            return eval_fn(egrid, comps_params)

        for integrate in model_info.integrate.values():
            fn = jax.jit(integrate(fn))

        return fn

    def _compile_additive_fn(self, model_info: ModelInfo) -> AdditiveFn | None:
        """Get the additive model evaluation function."""
        additive_comps = self._additive_comps

        if len(additive_comps) <= 1:
            return None

        fns = [comp._compile_model_fn(model_info) for comp in additive_comps]
        comps_labels = [(comp._name, comp._latex) for comp in additive_comps]

        @jax.jit
        def _fn(egrid: JAXArray, params: ParamIDValMapping) -> JAXArray:
            return [f(egrid, params) for f in fns]

        @jax.jit
        def fn(
            egrid: JAXArray, params: ParamIDValMapping
        ) -> dict[tuple[str, str], JAXArray]:
            """The evaluation function of additive components."""
            return dict(zip(comps_labels, _fn(egrid, params)))

        return fn

    def _repr_html_(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        # TODO: show the component name, LaTeX, type and parameters
        return self.name

    def __delattr__(self, key: str):
        if self.__initialized and hasattr(self, key):
            raise AttributeError("can't delete attribute")

        super().__delattr__(key)

    def __setattr__(self, key: str, value: Any):
        if self.__initialized and (
            not hasattr(self, key) or key in self._cid_to_cname.values()
        ):
            raise AttributeError("can't set attribute")

        super().__setattr__(key, value)

    def __getitem__(self, key: str) -> ComponentBase:
        if key not in self._cid_to_cname.values():
            raise KeyError(key)

        return getattr(self, key)

    def __add__(self, other: ModelBase) -> CompositeModel:
        return CompositeModel(self, other, '+')

    def __radd__(self, other: ModelBase) -> CompositeModel:
        return CompositeModel(other, self, '+')

    def __mul__(self, other: ModelBase) -> CompositeModel:
        return CompositeModel(self, other, '*')

    def __rmul__(self, other: ModelBase) -> CompositeModel:
        return CompositeModel(other, self, '*')


class CompiledModel:
    """Model with fast evaluation and frozen configuration."""

    __initialized: bool = False

    def __init__(
        self,
        name: str,
        fn: ModelCompiledFn,
        additive_fn: AdditiveFn | None,
        model_info: ModelInfo,
        mtype: Literal['add', 'mul'],
    ):
        self.name = name
        self._fn = fn
        self._additive_fn = additive_fn
        self._model_info = model_info
        self._type = mtype

        self._pname_to_pid = {
            model_info.name[pid]: pid for pid in model_info.sample
        }
        self._nparam = len(model_info.sample)

        self.__initialized = True

    @property
    def type(self) -> Literal['add', 'mul']:
        """Model type."""
        return self._type

    def _prepare_eval(self, params: Sequence | Mapping | None):
        """Check if `params` is valid for the model."""
        if isinstance(params, Sequence):
            if len(params) != self._nparam:
                raise ValueError(
                    f'got {len(params)} params, expected {self._nparam}'
                )

            params = [jnp.asarray(p, float) for p in params]
            params = dict(zip(self._pname_to_pid.values(), params))

        elif isinstance(params, Mapping):
            if set(params) != set(self._model_info.sample):
                missing = set(self._model_info.sample) - set(params)
                raise ValueError(f'missing parameters: {", ".join(missing)}')

            params = jax.tree_map(jnp.asarray, params)
            params = {self._pname_to_pid[k]: v for k, v in params.items()}

        elif params is None:
            params = self._model_info.default

        else:
            raise TypeError('params must be a sequence or a mapping')

        shapes = jax.tree_util.tree_flatten(
            tree=jax.tree_map(jnp.shape, params),
            is_leaf=lambda i: isinstance(i, tuple),
        )[0]

        if not shapes:
            raise ValueError('params are empty')

        shape = shapes[0]
        if any(s != shape for s in shapes[1:]):
            raise ValueError('all params must have the same shape')

        # iteratively vmap and jit over params dimensions
        # nested-jit trick is used to reduce the compilation time

        fn = self._fn
        for _ in range(len(shape)):
            fn = jax.jit(jax.vmap(fn, in_axes=(None, 0)))

        additive_fn = self._additive_fn
        if additive_fn is not None:
            for _ in range(len(shape)):
                additive_fn = jax.jit(jax.vmap(additive_fn, in_axes=(None, 0)))

        return fn, additive_fn, params

    def eval(
        self, egrid: ArrayLike, params: Sequence | Mapping | None = None
    ) -> JAXArray:
        """Evaluate the model.

        Parameters
        ----------
        egrid : ndarray
            Photon energy grid in units of keV.
        params : sequence or mapping, optional
            Parameter sequence or mapping for the model.

        Returns
        -------
        value : jax.Array
            The model value.

        """
        f, _, params = self._prepare_eval(params)

        return f(jnp.asarray(egrid, float), params)

    def ne(
        self,
        egrid: ArrayLike,
        params: Sequence | Mapping | None = None,
        comps: bool = False,
    ) -> JAXArray | dict[str, JAXArray]:
        r"""Calculate :math:`N(E)`.

        Parameters
        ----------
        egrid : ndarray
            Photon energy grid in units of keV.
        params : sequence or mapping, optional
            Parameter sequence or mapping for the model.
        comps : bool, optional
            Whether to return the result of each component. Defaults to False.

        Returns
        -------
        ne : jax.Array, or dict[str, jax.Array]
            The differential photon flux in units of cm⁻² s⁻¹ keV⁻¹.

        """
        if self.type != 'add':
            msg = f'N(E) is undefined for {self.type} type model "{self}"'
            raise TypeError(msg)

        if self._additive_fn is None and comps:
            raise RuntimeError(f'{self} has no sub-models with additive type')

        egrid = jnp.asarray(egrid, float)
        de = egrid[1:] - egrid[:-1]

        if comps:
            _, additive_fn, params = self._prepare_eval(params)
            comps_value = additive_fn(egrid, params)
            ne = jax.tree_map(lambda v: v / de, comps_value)
        else:
            ne = self.eval(egrid, params) / de

        return ne

    def ene(
        self,
        egrid: ArrayLike,
        params: Sequence | Mapping | None = None,
        comps: bool = False,
    ) -> JAXArray | dict[str, JAXArray]:
        r"""Calculate :math:`E N(E)`, i.e. :math:`F(\nu)`.

        Parameters
        ----------
        egrid : ndarray
            Photon energy grid in units of keV.
        params : sequence or mapping, optional
            Parameter sequence or mapping for the model.
        comps : bool, optional
            Whether to return the result of each component. Defaults to False.

        Returns
        -------
        ene : jax.Array, or dict[str, jax.Array]
            The differential energy flux in units of erg cm⁻² s⁻¹ keV⁻¹.

        """
        if self.type != 'add':
            msg = f'EN(E) is undefined for {self.type} type model "{self}"'
            raise TypeError(msg)

        if self._additive_fn is None and comps:
            raise RuntimeError(f'{self} has no sub-models with additive type')

        keV_to_erg = 1.602176634e-9
        egrid = jnp.asarray(egrid, float)
        emid = jnp.sqrt(egrid[:-1] * egrid[1:])
        de = egrid[1:] - egrid[:-1]
        factor = keV_to_erg * emid / de

        ne = self.ne(egrid, params, comps)

        if comps:
            ene = jax.tree_map(lambda v: factor * v, ne)
        else:
            ene = factor * ne

        return ene

    def eene(
        self,
        egrid: ArrayLike,
        params: Sequence | Mapping,
        comps: bool = False,
    ) -> JAXArray | dict[str, JAXArray]:
        r"""Calculate :math:`E^2 N(E)`, i.e. :math:`\nu F(\nu)`.

        Parameters
        ----------
        egrid : ndarray
            Photon energy grid in units of keV.
        params : sequence or mapping, optional
            Parameter sequence or mapping for the model.
        comps : bool, optional
            Whether to return the result of each component. Defaults to False.

        Returns
        -------
        eene : jax.Array, or dict[str, jax.Array]
            The energy flux in units of erg cm⁻² s⁻¹.

        """
        if self.type != 'add':
            msg = f'EEN(E) is undefined for {self.type} type model "{self}"'
            raise TypeError(msg)

        if self._additive_fn is None and comps:
            raise RuntimeError(f'{self} has no sub-models with additive type')

        keV_to_erg = 1.602176634e-9
        egrid = jnp.asarray(egrid, float)
        e2 = egrid[:-1] * egrid[1:]
        de = egrid[1:] - egrid[:-1]
        factor = keV_to_erg * e2 / de

        ne = self.ne(egrid, params, comps)

        if comps:
            eene = jax.tree_map(lambda v: factor * v, ne)
        else:
            eene = factor * ne

        return eene

    def ce(
        self,
        egrid: ArrayLike,
        resp_matrix: ArrayLike,
        ch_width: ArrayLike,
        params: Sequence | Mapping | None = None,
        comps: bool = False,
    ) -> JAXArray | dict[str, JAXArray]:
        r"""Calculate the folded model, i.e. :math:`C(E)`.

        Parameters
        ----------
        egrid : ndarray
            Photon energy grid of `resp_matrix`, in units of keV.
        resp_matrix : ndarray
            Instrumental response matrix used to fold the model.
        ch_width : ndarray
            Measured energy channel width of `resp_matrix`.
        params : sequence or mapping, optional
            Parameter sequence or mapping for the model.
        comps : bool, optional
            Whether to return the result of each component. Defaults to False.

        Returns
        -------
        ce : jax.Array, or dict[str, jax.Array]
            The folded model in units of s⁻¹ keV⁻¹.

        """
        if self.type != 'add':
            msg = f'C(E) is undefined for {self.type} type model "{self}"'
            raise TypeError(msg)

        if self._additive_fn is None and comps:
            raise RuntimeError(f'{self} has no sub-models with additive type')

        egrid = jnp.asarray(egrid, float)
        resp_matrix = jnp.asarray(resp_matrix, float)
        ch_width = jnp.asarray(ch_width, float)

        ne = self.ne(egrid, params, comps)
        de = egrid[1:] - egrid[:-1]
        fn = jax.jit(lambda v: (v * de) @ resp_matrix / ch_width)

        if comps:
            return jax.tree_map(fn, ne)
        else:
            return fn(ne)

    def flux(
        self,
        emin: float | int | JAXFloat,
        emax: float | int | JAXFloat,
        params: Sequence | Mapping | None = None,
        energy: bool = True,
        comps: bool = False,
        ngrid: int = 1000,
        log: bool = True,
    ) -> jax.Array | dict[str, jax.Array]:
        r"""Calculate the flux of model between `emin` and `emax`.

        Warnings
        --------
        The flux is calculated by trapezoidal rule, which may not be accurate
        if not enough energy bins are used when the difference between
        `emin` and `emax` is large.

        Parameters
        ----------
        emin : float or int
            Minimum value of energy range, in units of keV.
        emax : float or int
            Maximum value of energy range, in units of keV.
        params : sequence or mapping, optional
            Parameter sequence or mapping for the model.
        energy : bool, optional
            When True, calculate energy flux in units of erg cm⁻² s⁻¹;
            otherwise calculate photon flux in units of cm⁻² s⁻¹.
            The default is True.
        comps : bool, optional
            Whether to return the result of each component. The default is
            False.
        ngrid : int, optional
            The energy grid number to use in integration. The default is 1000.
        log : bool, optional
            Whether to use logarithmically regular energy grid. The default is
            True.

        Returns
        -------
        flux : jax.Array, or dict[str, jax.Array]
            The model flux.

        """
        if self.type != 'add':
            msg = f'flux is undefined for {self.type} type model "{self}"'
            raise TypeError(msg)

        if log:
            egrid = jnp.geomspace(emin, emax, ngrid)
        else:
            egrid = jnp.linspace(emin, emax, ngrid)

        if energy:
            f = self.ene(egrid, params, comps)
        else:
            f = self.ne(egrid, params, comps)

        de = jnp.diff(egrid)
        fn = jax.jit(lambda v: jnp.sum(v * de, axis=-1))

        if comps:
            return jax.tree_map(fn, f)
        else:
            return fn(f)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        # TODO: show the component name, LaTeX, type and parameters
        return self.name

    def __delattr__(self, key: str):
        if self.__initialized and hasattr(self, key):
            raise AttributeError("can't delete attribute")

        super().__delattr__(key)

    def __setattr__(self, key: str, value: Any):
        if self.__initialized:
            raise AttributeError("can't set attribute")

        super().__setattr__(key, value)


class Model(ModelBase):
    """Model defined by a single additive or multiplicative component."""

    _component: ComponentBase

    def __init__(self, component: ComponentBase):
        self._component = component
        super().__init__(component._id, component._id, [component])

    @property
    def eval(self) -> ModelEval:
        comp_id = self._component._id
        _fn = self._component.eval

        def fn(egrid: JAXArray, params: CompIDParamValMapping) -> JAXArray:
            """The model evaluation function"""
            return _fn(egrid, params[comp_id])

        return fn

    @property
    def type(self) -> Literal['add', 'mul']:
        return self._component.type


class CompositeModel(ModelBase):
    """Model defined by sum or product of two models."""

    _operands: tuple[ModelBase, ModelBase]
    _op: Callable[[JAXArray, JAXArray], JAXArray]
    _op_symbol: Literal['+', '*']
    _type: Literal['add', 'mul']
    __additive_comps: tuple[ModelBase, ...] | None = None

    def __init__(self, lhs: ModelBase, rhs: ModelBase, op: Literal['+', '*']):
        # check if the type of lhs and rhs are both model
        if not (isinstance(lhs, ModelBase) and isinstance(rhs, ModelBase)):
            raise TypeError(
                f'unsupported operand types for {op}: '
                f"'{type(lhs).__name__}' and '{type(rhs).__name__}'"
            )

        self._operands = (lhs, rhs)
        self._op_symbol = op

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
            op_name = '{} + {}'
            op_latex = '{{{}}} + {{{}}}'

        elif op == '*':
            if 'add' == type1 == type2:
                raise TypeError(
                    f'unsupported model type for *: {lhs} (additive) '
                    f'and {rhs} (additive)'
                )

            if 'conv' == type1 or 'conv' == type2:
                raise TypeError(
                    f'unsupported model type for *: {lhs} ({type1}) '
                    f'and {rhs} ({type2})'
                )

            self._type = 'add' if type1 == 'add' or type2 == 'add' else 'mul'
            self._op = jnp.multiply
            op_name = '{} * {}'
            op_latex = r'{{{}}} \times {{{}}}'

        else:
            raise NotImplementedError(f'op {op}')

        lhs_name = lhs._name
        lhs_latex = lhs._latex
        rhs_name = rhs._name
        rhs_latex = rhs._latex

        if op == '*':
            if isinstance(lhs, CompositeModel) and lhs._op_symbol == '+':
                lhs_name = f'({lhs_name})'
                lhs_latex = rf'\left({lhs_latex}\right)'

            if isinstance(rhs, CompositeModel) and rhs._op_symbol == '+':
                rhs_name = f'({rhs_name})'
                rhs_latex = rf'\left({rhs_latex}\right)'

        comps = list(lhs._comps)
        for c in rhs._comps:
            if c not in comps:
                comps.append(c)

        super().__init__(
            op_name.format(lhs_name, rhs_name),
            op_latex.format(lhs_latex, rhs_latex),
            comps,
        )

    @property
    def eval(self) -> ModelEval:
        op = self._op
        lhs = self._operands[0].eval
        rhs = self._operands[1].eval

        def fn(egrid: JAXArray, params: CompIDParamValMapping) -> JAXArray:
            """The model evaluation function"""
            return op(lhs(egrid, params), rhs(egrid, params))

        return fn

    @property
    def _additive_comps(self) -> tuple[ModelBase, ...]:
        if self.__additive_comps is not None:
            return self.__additive_comps

        if self.type != 'add':
            comps = ()
        else:
            op = self._op_symbol
            lhs = self._operands[0]
            rhs = self._operands[1]

            if op == '+':  # add + add
                comps = lhs._additive_comps + rhs._additive_comps
            else:
                if lhs.type == 'add':  # add * mul
                    lhs_comps = lhs._additive_comps
                    if len(lhs_comps) >= 2:
                        comps = tuple(c * rhs for c in lhs_comps)
                    else:
                        comps = (self,)
                else:  # mul * add
                    rhs_comps = rhs._additive_comps
                    if len(rhs_comps) >= 2:
                        comps = tuple(lhs * c for c in rhs_comps)
                    else:
                        comps = (self,)

        self.__additive_comps = comps

        return self.__additive_comps

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

        # check if all static methods are defined correctly
        if config is not None:
            for mname in cls._staticmethod:
                method = inspect.getattr_static(cls, mname)
                if not isinstance(method, staticmethod):
                    raise TypeError(f'{name}.{mname} must be a staticmethod')

        # define __init__ method of the newly created class if necessary
        if config is not None and '__init__' not in dct:
            # signature of __init__ method
            sig1 = [
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
                f'def __init__(self, {", ".join(sig1)}):\n'
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

    def __call__(cls, *args, **kwargs) -> Model | ConvolutionModel:
        # return Model/ConvolutionModel after Component initialization
        component = super().__call__(*args, **kwargs)

        if component.type != 'conv':
            return Model(component)
        else:
            return ConvolutionModel(component)


def _param_getter(name: str) -> Callable[[ComponentBase], ParameterBase]:
    def getter(self: ComponentBase) -> ParameterBase:
        """Get parameter."""
        return getattr(self, f'__{name}')

    return getter


def _param_setter(name: str, idx: int) -> Callable[[ComponentBase, Any], None]:
    def setter(self: ComponentBase, param: Any) -> None:
        """Set parameter."""

        if param is not None and getattr(self, f'__{name}', None) is param:
            return

        cfg = self._config[idx]

        if param is None:
            # use default configuration
            param = UniformParameter(
                name=cfg.name,
                default=cfg.default,
                min=cfg.min,
                max=cfg.max,
                log=cfg.log,
                fixed=cfg.fixed,
                latex=cfg.latex,
            )

        elif isinstance(
            param, (float, int, jnp.number, jnp.ndarray, NumPyArray)
        ):
            # fixed to the given value
            if jnp.shape(param) != ():
                raise ValueError('scalar value is expected')

            param = UniformParameter(
                name=cfg.name,
                default=param,
                min=cfg.min,
                max=cfg.max,
                log=cfg.log,
                fixed=True,
                latex=cfg.latex,
            )

        elif isinstance(param, ParameterBase):
            # given parameter instance
            param = param

        else:
            # other input types are not supported yet
            raise TypeError(
                f'{type(self).__name__}.{cfg.name} got unsupported value '
                f'{param} ({type(param).__name__})'
            )

        prev_param: ParameterBase | None = getattr(self, f'__{name}', None)
        if prev_param is not None:
            prev_param._tracker.remove(self._id, name)

        setattr(self, f'__{name}', param)

        param._tracker.append(self._id, name)

    return setter


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

    _config: tuple[ParamConfig, ...]
    _id: CompID
    _args: tuple[str, ...] = ()  # extra args passed to subclass __init__
    _kwargs: tuple[str, ...] = ()  # extra kwargs passed to subclass __init__
    _staticmethod: tuple[str, ...] = ()  # method needs to be static
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

    @property
    @abstractmethod
    def eval(self) -> CompEval:
        """Get side-effect free component evaluation function."""
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
    @abstractmethod
    def _config(self) -> tuple[ParamConfig, ...]:
        """Default configuration of parameters."""
        pass

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        # TODO: show the component name, LaTeX, type and parameters
        return self.name

    def __getitem__(self, key: str) -> ParameterBase:
        if key not in self.param_names:
            raise KeyError(key)

        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        if key not in self.param_names:
            raise KeyError(key)

        setattr(self, key, value)

    def __delattr__(self, key: str):
        if self.__initialized and hasattr(self, key):
            raise AttributeError("can't delete attribute")

        super().__delattr__(key)

    def __setattr__(self, key: str, value: Any):
        if self.__initialized and not hasattr(self, key):
            raise AttributeError("can't set attribute")

        super().__setattr__(key, value)


class AdditiveComponent(ComponentBase):
    """Prototype class to define an additive component."""

    @property
    def type(self) -> Literal['add']:
        """Component type is additive."""
        return 'add'


class MultiplicativeComponent(ComponentBase):
    """Prototype class to define a multiplicative component."""

    @property
    def type(self) -> Literal['mul']:
        """Component type is multiplicative."""
        return 'mul'


class AnalyticalIntegral(ComponentBase):
    """Prototype component to calculate model integral analytically."""

    _integral_jit: CompEval | None = None
    _staticmethod = ('integral',)

    def __init__(self, params: dict, latex: str | None):
        super().__init__(params, latex)

    @property
    def eval(self) -> CompEval:
        if self._integral_jit is None:
            self._integral_jit = jax.jit(self.integral)

        return self._integral_jit

    @staticmethod
    @abstractmethod
    def integral(*args, **kwargs) -> JAXArray:
        """Calculate the model value over grid."""
        pass


class NumericalIntegral(ComponentBase):
    """Prototype component to calculate model integral numerically."""

    _kwargs = ('method',)
    _continnum_jit: CompEval | None = None
    _staticmethod = ('continnum',)

    def __init__(
        self,
        params: dict,
        latex: str | None,
        method: Literal['trapz', 'simpson'] | None = None,
    ):
        self.method = 'trapz' if method is None else method
        super().__init__(params, latex)

    @property
    def eval(self) -> CompEval:
        if self._continnum_jit is None:
            # _continnum is assumed to be a pure function, independent of self
            f_jit = jax.jit(self.continnum)
            f_vmap = jax.vmap(f_jit, in_axes=(0, None))
            self._continnum_jit = jax.jit(f_vmap)

        continnum = self._continnum_jit
        mtype = self.type

        if self.method == 'trapz':

            def fn(egrid: JAXArray, params: NameValMapping) -> JAXArray:
                """Numerical integration using trapezoidal rule."""
                if mtype == 'add':
                    factor = 0.5 * (egrid[1:] - egrid[:-1])
                else:
                    factor = 0.5
                f_grid = continnum(egrid, params)
                return factor * (f_grid[:-1] + f_grid[1:])

        elif self.method == 'simpson':

            def fn(egrid: JAXArray, params: NameValMapping) -> JAXArray:
                """Numerical integration using Simpson's 1/3 rule."""
                if mtype == 'add':
                    factor = (egrid[1:] - egrid[:-1]) / 6.0
                else:
                    factor = 1.0 / 6.0
                e_mid = 0.5 * (egrid[:-1] + egrid[1:])
                f_grid = self._continnum_jit(egrid, params)
                f_mid = self._continnum_jit(e_mid, params)
                return factor * (f_grid[:-1] + 4.0 * f_mid + f_grid[1:])

        else:
            raise NotImplementedError(f"integration method '{self.method}'")

        return jax.jit(fn)

    @staticmethod
    @abstractmethod
    def continnum(*args, **kwargs) -> JAXArray:
        """Calculate the model value at the energy grid."""
        pass

    @property
    def method(self) -> Literal['trapz', 'simpson']:
        """Numerical integration method."""
        return self._method

    @method.setter
    def method(self, method: Literal['trapz', 'simpson']):
        if method not in ('trapz', 'simpson'):
            raise ValueError(
                f"available integration methods are 'trapz' and 'simpson', "
                f"but got '{method}'"
            )

        self._method = method


class AnaIntAdditive(AnalyticalIntegral, AdditiveComponent):
    """Prototype additive component with integral expression defined."""

    @staticmethod
    @abstractmethod
    def integral(*args, **kwargs) -> JAXArray:
        """Calculate the photon flux integrated over the energy grid.

        Parameters
        ----------
        egrid : ndarray
            Photon energy grid in units of keV.
        params : dict
            Parameter dict for the model.

        Returns
        -------
        flux : ndarray
            The photon flux integrated over `egrid`, in units of cm⁻² s⁻¹.

        """
        pass


class NumIntAdditive(NumericalIntegral, AdditiveComponent):
    """Prototype additive component with continnum expression defined."""

    @staticmethod
    @abstractmethod
    def continnum(*args, **kwargs) -> JAXArray:
        """Calculate the photon flux at the energy grid.

        Parameters
        ----------
        egrid : ndarray
            Photon energy grid in units of keV.
        params : dict
            Parameter dict for the model.

        Returns
        -------
        flux : ndarray
            The photon flux at `egrid`, in units of cm⁻² s⁻¹ keV⁻¹.

        """
        pass


class AnaIntMultiplicative(AnalyticalIntegral, MultiplicativeComponent):
    """Prototype multiplicative component with integral expression defined."""

    @staticmethod
    @abstractmethod
    def integral(*args, **kwargs) -> JAXArray:
        """Calculate the average value of the model between the grid.

        Parameters
        ----------
        egrid : ndarray
            Photon energy grid in units of keV.
        params : dict
            Parameter dict for the model.

        Returns
        -------
        value : ndarray
            The model value, dimensionless.

        """
        pass


class NumIntMultiplicative(NumericalIntegral, MultiplicativeComponent):
    """Prototype multiplicative component with continnum expression defined."""

    @staticmethod
    @abstractmethod
    def continnum(*args, **kwargs) -> JAXArray:
        """Calculate the model value at the energy grid.

        Parameters
        ----------
        egrid : ndarray
            Photon energy grid in units of keV.
        params : dict
            Parameter dict for the model.

        Returns
        -------
        value : ndarray
            The model value at `egrid`, dimensionless.

        """
        pass


class ConvolutionModel(ModelBase):
    """Model defined by a convolution component."""

    def __init__(self, component: ConvolutionComponent):
        self._component = component
        super().__init__(component._id, component._id, [component])

    def __call__(self, model: ModelBase) -> ConvolvedModel:
        if model.type not in self._component._supported:
            accepted = [f"'{i}'" for i in self._component._supported]
            raise TypeError(
                f'{self.name} convolution model supports models with type: '
                f"{', '.join(accepted)}; got '{model.type}' type model {model}"
            )

        return ConvolvedModel(self._component, model)

    @property
    def eval(self):
        # this is not supposed to be called
        raise RuntimeError('convolution model has no eval defined')

    @property
    def type(self) -> Literal['conv']:
        """Model type is convolution."""
        return 'conv'


class ConvolvedModel(ModelBase):
    """Model created by convolution."""

    _op: ConvolutionComponent
    _model: ModelBase

    def __init__(self, op: ConvolutionComponent, model: ModelBase):
        self._op = op
        self._model = model
        name = f'{op._id}({model._name})'
        latex = rf'{{{op._id}}}\left({model._latex}\right)'

        comps = list(model._comps)
        if op not in comps:
            comps = comps + [op]

        self.__additive_comps = None

        super().__init__(name, latex, comps)

    @property
    def _additive_comps(self) -> tuple[ModelBase, ...]:
        if self.__additive_comps is not None:
            return self.__additive_comps

        if self.type != 'add':
            comps = ()
        else:
            op = self._op
            model_comps = self._model._additive_comps

            if len(model_comps) >= 2:
                comps = tuple(ConvolvedModel(op, m) for m in model_comps)
            else:
                comps = (self,)

        self.__additive_comps = comps

        return self.__additive_comps

    @property
    def eval(self) -> ModelEval:
        comp_id = self._op._id
        _fn = self._op.eval
        _model_fn = self._model.eval

        def fn(egrid: JAXArray, params: CompIDParamValMapping) -> JAXArray:
            """The convolved model evaluation function."""
            return _fn(egrid, params[comp_id], lambda e: _model_fn(e, params))

        return fn

    @property
    def type(self) -> Literal['add', 'mul']:
        return self._model.type


class ConvolutionComponent(ComponentBase):
    """Prototype class to define a convolution component."""

    _supported: frozenset = frozenset({'add', 'mul'})
    _convolve_jit: ConvolveEval | None = None
    _staticmethod = ('convolve',)

    @property
    def type(self) -> Literal['conv']:
        return 'conv'

    @property
    def eval(self) -> ConvolveEval:
        if self._convolve_jit is None:
            self._convolve_jit = jax.jit(self.convolve, static_argnums=2)

        return self._convolve_jit

    @staticmethod
    @abstractmethod
    def convolve(*args, **kwargs) -> JAXArray:
        """Convolve a model function.

        Parameters
        ----------
        egrid : ndarray
            Photon energy grid in units of keV.
        params : dict
            Parameter dict for the convolution model.
        model_fn : callable
            The model function to be convolved, which takes the energy grid as
            input and returns the model value over the grid.

        Returns
        -------
        value : ndarray
            The convolved model value over `egrid`.

        """
        pass


class ModelInfo(NamedTuple):
    """Model information.

    Parameters
    ----------
    info : tuple
        The model parameter information.
    name : dict
        The mapping of parameter id to parameter name.
    latex : dict
        The mapping of component parameters names to LaTeX.
    sample : dict
        The mapping of free parameter id to numpyro Distribution.
    default : dict
        The mapping of free parameter id to default value.
    deterministic : tuple
        The id of deterministic parameters.
    fixed : dict
        The mapping of fixed parameter id to fixed parameter value.
    log: dict
        The mapping of parameter id to whether the parameter is logarithmic.
    cid_pval : dict
        The mapping of component id to parameter value getter function.
    integrate : dict
        The mapping of interval parameter id to integral operator.
    setup : dict
        The mapping of component parameter setup.

    """

    info: tuple[tuple[str, str, str, str, str, str], ...]
    name: ParamIDStrMapping
    latex: dict[CompParamName, str]
    sample: dict[ParamID, Distribution]
    default: ParamIDValMapping
    deterministic: tuple[ParamID, ...]
    fixed: ParamIDValMapping
    log: dict[ParamID, bool]
    cid_pval: dict[CompID, Callable[[ParamIDValMapping], NameValMapping]]
    integrate: dict[ParamID, IntegralFactory]
    setup: dict[CompParamName, tuple[ParamID, ParamSetup]]


class ParamSetup(Enum):
    """Model parameter setup."""

    Free = 0
    Composite = 1
    Forwarded = 2
    Fixed = 3
    Integrated = 4


def get_model_info(
    comps: Sequence[ComponentBase],
    cid_to_cname: CompIDStrMapping,
) -> ModelInfo:
    """Get the model information.

    Parameters
    ----------
    comps : sequence of ComponentBase
        The sequence of components.
    cid_to_cname : mapping
        The mapping of component id to component name.

    Returns
    -------
    info : ModelInfo
        The model parameter information dict.

    """
    # get all parameter information
    params_info: dict[ParamID, ParamInfo] = reduce(
        lambda x, y: x | y,
        [comp[name]._info for comp in comps for name in comp.param_names],
    )

    # mapping from pid to assigned component id and parameter name
    comp_param: dict[ParamID, tuple[CompID, CompParamName] | None] = {
        pid: tmp
        for pid, info in params_info.items()
        if (tmp := info.tracker.comp_param(cid_to_cname.keys()))
    }

    # names of non-composite params with no component assigned (aux params)
    aux_params = {
        pid: info
        for pid, info in params_info.items()
        if (pid not in comp_param) and (not info.composite)
    }

    aux_names = [i.name for i in aux_params.values()]
    aux_names = build_namespace(aux_names, prime=True)

    # name mapping of aux params
    name_mapping: ParamIDStrMapping = dict(zip(aux_params.keys(), aux_names))

    # record name mapping of params directly assigned to a component
    name_mapping |= {
        pid: f'{cid_to_cname[cid]}.{pname}'
        for pid, (cid, pname) in comp_param.items()
    }

    # record the LaTeX format of aux parameters
    aux_latex = [params_info[pid].latex for pid in aux_params]
    aux_latex = build_namespace(aux_latex, latex=True, prime=True)
    latex_mapping = dict(zip(aux_params.keys(), aux_latex))

    # record the LaTeX format of component parameters from _config
    latex_mapping |= {
        comp[name]._id: comp._config[i].latex
        for comp in comps
        for (i, name) in enumerate(comp.param_names)
    }

    # record whether the parameter is logarithmic
    log = {pid: params_info[pid].log for pid in name_mapping}

    # record the value of fixed parameters
    fixed = {
        pid: info.default
        for pid, info in params_info.items()
        if (pid in name_mapping) and info.fixed and (not info.integrate)
    }

    # integral operator
    integrate = {
        pid: info.integrate
        for pid, info in params_info.items()
        if info.integrate and not info.composite
    }

    # record the sample distribution of free parameters
    sample: dict[ParamID, Distribution] = {
        pid: info.dist for pid, info in params_info.items() if info.dist
    }

    # record the default value of free parameters
    default: dict[ParamID, JAXFloat] = {
        pid: params_info[pid].default for pid in sample
    }

    # ========== generate component parameter value getter function ===========
    def factory1(id_to_value: T) -> T:
        """Fill in the default parameter values for fn."""

        def fn(value_mapping: ParamIDValMapping) -> JAXFloat:
            """id_to_value with default values."""
            return id_to_value(fixed_values | value_mapping)

        return fn

    def factory2(
        pname_to_pid: dict[CompParamName, ParamID],
    ) -> Callable[[ParamIDValMapping], NameValMapping]:
        """Factory for component parameter value getter."""

        def fn(value_mapping: ParamIDValMapping) -> NameValMapping:
            """Component parameter value getter."""
            return {
                pname: pid_to_value[pid](value_mapping)
                for pname, pid in pname_to_pid.items()
            }

        return fn

    fixed_values = fixed
    pid_to_value = {
        pid: factory1(params_info[pid].id_to_value) for pid in comp_param
    }
    cid_to_pval = {
        comp._id: factory2({name: comp[name]._id for name in comp.param_names})
        for comp in comps
    }
    # ========== generate component parameter value getter function ===========

    # record non-interval params which has a component assigned
    deterministic = []

    # record the setup of component parameters
    setup: dict[CompParamName, (ParamID, ParamSetup)] = {}

    # model_info is list of (No., Component, Parameter, Value, Bound, Prior)
    model_info = []
    sample_order = []
    n = 0

    for comp in comps:  # iterate through components
        cid = comp._id
        cname = cid_to_cname[cid]
        for pname in comp.param_names:  # iterate through parameters of comp
            model_pname = f'{cname}.{pname}'
            pid = comp[pname]._id
            info = params_info[pid]

            idx = ''
            bound = info.bound
            prior = info.prior

            if (cid, pname) != comp_param[pid]:  # param is linked to another
                value = name_mapping[pid]
                setup[model_pname] = (pid, ParamSetup.Forwarded)

            elif info.composite:  # param is composite
                mapping = dict(name_mapping)
                del mapping[pid]
                value = info.name(mapping)

                if info.integrate:  # param is composite interval
                    setup[model_pname] = (pid, ParamSetup.Integrated)
                elif info.fixed:  # param is composite, but fixed
                    setup[model_pname] = (pid, ParamSetup.Fixed)
                else:  # param is composite but free to vary, add it to determ
                    setup[model_pname] = (pid, ParamSetup.Composite)
                    deterministic.append(pid)

            else:  # a single param
                value = f'{info.default:.4g}'

                if info.fixed:
                    setup[model_pname] = (pid, ParamSetup.Fixed)
                else:
                    setup[model_pname] = (pid, ParamSetup.Free)
                    n += 1
                    idx = str(n)
                    sample_order.append(pid)

            model_info.append((idx, cname, pname, value, bound, prior))

    for pid, info in aux_params.items():
        idx = ''
        bound = info.bound
        prior = info.prior

        if info.integrate:  # param is integral-type
            value = f'[{info.default[0]:.4g}, {info.default[1]:.4g}]'
        else:  # param is not integral-type, record its default value
            value = f'{info.default:.4g}'

        if not info.fixed:  # param is free to vary
            n += 1
            idx = str(n)
            sample_order.append(pid)

        model_info.append((idx, '', name_mapping[pid], value, bound, prior))

    if set(sample_order) != set(sample):
        raise RuntimeError('sample_order and sample are not consistent')

    sample = {pid: sample[pid] for pid in sample_order}

    return ModelInfo(
        info=tuple(model_info),
        name=name_mapping,
        latex=latex_mapping,
        sample=sample,
        default=default,
        deterministic=tuple(deterministic),
        fixed=fixed,
        log=log,
        cid_pval=cid_to_pval,
        integrate=integrate,
        setup=setup,
    )
