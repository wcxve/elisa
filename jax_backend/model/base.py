"""Module to handle spectral model construction.
TODO: I think refactoring is needed.
      The model construction and parameter binding can be split.
"""

from __future__ import annotations

from abc import ABCMeta, ABC, abstractmethod
from typing import Any, Callable, Optional, Union
from uuid import uuid4

import jax.numpy as jnp
from numpyro.distributions import Uniform, LogUniform

_UUID = []  # stores uuid that has been used by Node


class NodeABC(ABC):
    """The abstract node class."""

    type: str
    attrs: dict[str, Any]
    predecessor: list[NodeABC]
    name: str
    fmt: str

    def __init__(
        self,
        name: str,
        fmt: str,
        is_operation: bool = False,
        predecessor: Optional[list[NodeABC]] = None,
        **kwargs: Any
    ) -> None:

        if predecessor is None:
            self.predecessor = []
        else:
            self.predecessor = predecessor

        node_id = str(uuid4().hex)

        # check if uuid4 collides, which could result from problems with code
        if node_id in _UUID:
            raise RuntimeError('UUID4 collision found!')

        self.attrs = dict(
            name=name,
            fmt=fmt,
            type=self.type,
            is_operation=is_operation,
            id=node_id
        )
        self.attrs.update(kwargs)  # FIXME: should not have kws appeared above

    def __repr__(self) -> str:
        return self.name

    @abstractmethod
    def __add__(self, other: NodeABC):
        pass

    @abstractmethod
    def __mul__(self, other: NodeABC):
        pass

    @property
    def name(self) -> str:
        """Name of the node."""

        return self.label_with_id('name')

    @property
    def fmt(self):
        """Plot format of the node."""

        return self.label_with_id('fmt')

    def label_with_id(self, label):
        """Add node id suffix for name or fmt. The id can be replaced by
        number later.
        """

        if label != 'name' and label != 'fmt':
            raise ValueError('``label`` should be "name" or "fmt"')

        return f'{self.attrs[label]}_{self.attrs["id"]}'


class OperationABC(NodeABC, ABC):
    """The abstract operation class."""

    def __init__(
        self,
        lh: NodeABC,
        rh: NodeABC,
        operator: str
    ) -> None:
        self.type = self.get_type(lh, rh, operator)

        # NOTE:
        #   - fmt for a convolution model is '*',
        #   - for Component, subtype is given when initialized
        #   - for ComponentOperation, subtype is automatically determined
        if not isinstance(lh, ComponentOperation) and \
                lh.attrs.get('subtype') == 'con':
            fmt = '*'
        else:
            fmt = r'$\times$' if operator == '*' else operator

        super().__init__(
            name=operator,
            fmt=fmt,
            is_operation=True,
            predecessor=[lh, rh]
        )

    def label_with_id(self, label) -> str:
        """Pretty format by composing operands for operation."""

        lh, rh = self.predecessor
        op_name = self.attrs['name']
        op = self.attrs[label]
        lh_label = getattr(lh, label)
        rh_label = getattr(rh, label)

        if op_name == '*':
            if lh.attrs['name'] == '+':
                lh_label = f'({lh_label})'

            if rh.attrs['name'] == '+':
                rh_label = f'({rh_label})'

        return f'{lh_label} {op} {rh_label}'

    @staticmethod
    def get_type(
        lh: NodeABC,
        rh: NodeABC,
        operator: str
    ) -> str:
        """Check if lh and rh are correct and return the type."""

        if operator not in {'+', '*'}:
            raise ValueError(f'operator "{operator}" is not supported')

        if not isinstance(lh, NodeABC):
            raise TypeError(f'got wrong input {lh}')

        if not isinstance(rh, NodeABC):
            raise TypeError(f'got wrong input {rh}')

        type1 = lh.type
        type2 = rh.type
        if type1 != type2:
            raise TypeError(
                f'unsupported types for {operator}: "{type1}" and "{type2}"'
            )

        return type1


class Parameter(NodeABC):
    """Handle parameter definition of a component."""

    type: str = 'parameter'
    default: float

    def __init__(
        self,
        name: str,
        fmt: str,
        default: float,
        distribution: Any,  # FIXME: type hint
        deterministic: Any  # FIXME: type hint
    ) -> None:

        super().__init__(
            name=name,
            fmt=fmt,
            default=default,
            distribution=distribution,
            deterministic=deterministic
        )

    def __add__(
        self,
        other: Union[Parameter, ParameterOperation]
    ) -> ParameterOperation:
        return ParameterOperation(self, other, '+')

    def __mul__(
        self,
        other: Union[Parameter, ParameterOperation]
    ) -> ParameterOperation:
        return ParameterOperation(self, other, '*')

    @property
    def default(self):
        """Default value of the parameter."""
        return self.attrs['default']


class ParameterOperation(OperationABC):
    """Handle parameter operation by user."""

    # type is assigned in Operation
    default: float

    def __init__(
        self,
        lh: Union[Parameter, ParameterOperation],
        rh: Union[Parameter, ParameterOperation],
        operator: str
    ) -> None:
        # TODO: implementation detail
        super().__init__(lh, rh, operator)

    def __repr__(self):
        lh = repr(self.predecessor[0])
        rh = repr(self.predecessor[1])
        op = self.attrs['name']

        return f'({lh}) {op} ({rh})'

    def __add__(
        self,
        other: Union[Parameter, ParameterOperation]
    ) -> ParameterOperation:
        return ParameterOperation(self, other, '+')

    def __mul__(
        self,
        other: Union[Parameter, ParameterOperation]
    ) -> ParameterOperation:
        return ParameterOperation(self, other, '*')

    @property
    def default(self):
        """Default value of the parameter operation."""

        lh = self.predecessor[0].attrs['default']
        rh = self.predecessor[1].attrs['default']

        if self.attrs['name'] == '+':
            return lh + rh
        else:
            return lh * rh


class UniformParameter(Parameter):
    """The default class to handle parameter definition."""

    def __init__(
        self,
        name: str,
        fmt: str,
        default: float,
        low: float,
        high: float,
        frozen: bool = False,
        log: bool = False
    ):
        self._config = {
            'name': name,
            'fmt': fmt,
            'default': default,
            'low': low,
            'high': high,
            'frozen': frozen,
            'log': log
        }

        if frozen:
            distribution = default  # TODO: use a Delta distribution?
        else:
            if log:
                distribution = LogUniform(low, high)
            else:
                distribution = Uniform(low, high)

        if not frozen and log:
            deterministic = (rf'\ln({fmt})', lambda x: jnp.log(x))
        else:
            deterministic = None

        super().__init__(name, fmt, default, distribution, deterministic)

    def __repr__(self):
        name = self._config["name"]
        init = self._config["default"]

        if self._config['frozen']:
            s = f'{name} = {init}'
        else:
            low = self._config["low"]
            high = self._config["high"]
            if self._config['log']:
                s = f'{name} ~ LogUniform(min={low}, max={high}, init={init})'
            else:
                s = f'{name} ~ Uniform(min={low}, max={high}, init={init})'

        return s


class Component(NodeABC):
    """Handle component definition."""

    type: str = 'component'
    predecessor: list[Union[Parameter, ParameterOperation]]
    params: dict[str, Union[Parameter, ParameterOperation]]
    comps: dict[str, Component]

    def __init__(
        self,
        name: str,
        fmt: str,
        subtype: str,
        params: dict[str, Union[Parameter, ParameterOperation]],
        func: Callable,
        is_ncon: bool
    ) -> None:
        if params is None:
            self._params = ()
            predecessor = None
        else:
            for v in params.values():
                if v.type != 'parameter':
                    raise ValueError('Parameter type is required')

            self._params = tuple(params.keys())
            predecessor = list(params.values())

        super().__init__(
            name=name,
            fmt=fmt,
            predecessor=predecessor,
            subtype=subtype,
            func=func,
            is_ncon=is_ncon
        )

    def __getitem__(self, key: str) -> Union[Parameter, ParameterOperation]:
        """Get parameter node."""

        if key not in self._params:
            raise ValueError(
                f'{self.attrs["name"]} has parameter(s) {self._params}, but no'
                f'"{key}"'
            )

        idx = self._params.index(key)

        return self.predecessor[idx]

    def __setitem__(
        self, key: str,
        value: Union[Parameter, ParameterOperation]
    ):
        """Set parameter given by user."""

        if key not in self._params:
            raise ValueError(
                f'{self.attrs["name"]} has parameter(s) {self._params}, but no'
                f'"{key}"'
            )

        idx = self._params.index(key)
        self.predecessor[idx] = value

    def __add__(
        self,
        other: Union[Component, ComponentOperation]
    ) -> ComponentOperation:
        return ComponentOperation(self, other, '+')

    def __mul__(
        self,
        other: Union[Component, ComponentOperation]
    ) -> ComponentOperation:
        return ComponentOperation(self, other, '*')

    @property
    def params(self) -> dict:
        """Get parameter configure."""

        return {
            self.name: {
                name: p for name, p in zip(self._params, self.predecessor)
            }
        }

    @property
    def comps(self) -> dict[str, Component]:
        """Get component dict."""

        return {self.name: self}

    @property
    def func(self):
        """Get component evaluation function."""

        comp_name = str(self.name)
        func = self.attrs['func']
        mtype = self.attrs['subtype']

        # notation: p=params, e=egrid, f=flux, ff=flux_func
        # params structure should be {component_id: {par1: ..., param2: ...}}
        if mtype == 'add':
            def add_wrapper(p, e, *_):
                """evaluate add component"""
                return func(e, **p[comp_name])

            return add_wrapper

        elif mtype == 'mul':
            def mul_wrapper(p, e, *_):
                """evaluate mul component"""
                return func(e, **p[comp_name])

            return mul_wrapper

        elif mtype == 'con':
            if self.attrs['is_ncon']:
                def ncon_wrapper(p, _=None, f=None, ff=None):
                    """evaluate ncon component, f and ff must be given"""
                    return func(ff, f, **p[comp_name])

                return ncon_wrapper

            else:
                def con_wrapper(p, e, f, *_):
                    """evaluate con component"""
                    return func(e, f, **p[comp_name])

                return con_wrapper

        else:
            raise TypeError(f'unrecognized model type "{mtype}"')

    # def set_params(
    #     self,
    #     params: dict[str, Union[Parameter, OperationABC]]
    # ) -> None:
    #     """Set parameters."""
    #
    #     params_in = set(params.keys())
    #     params_all = set(self.params.keys())
    #
    #     if not params_in <= params_all:
    #         diff = params_all - params_in
    #         raise ValueError(f'input params {diff} not included in {self}')
    #
    #     for v in params.values():
    #         if v.type != 'parameter':
    #             raise ValueError('Parameter type is required')
    #
    #     new_params = self.params | params
    #     self.predecessor = list(new_params.values())


class ComponentOperation(OperationABC):
    """Handle component operation."""

    # type is assigned in Operation
    params: dict

    def __init__(
        self,
        lh: Union[Component, ComponentOperation],
        rh: Union[Component, ComponentOperation],
        operator: str
    ) -> None:
        # check input and get the result type
        if self.get_type(lh, rh, operator) != 'component':
            raise TypeError('lh and rh must be component type')

        subtype1 = lh.attrs['subtype']
        subtype2 = rh.attrs['subtype']

        # check if operand is legal for the operator
        if operator == '+':
            if subtype1 != 'add':
                raise TypeError(f'{lh} is not additive')

            if subtype2 != 'add':
                raise TypeError(f'{rh} is not additive')

        else:  # operator == '*'
            if subtype1 == 'add':
                if subtype2 == 'add':
                    raise TypeError(
                        f'unsupported types for *: {lh} (add) and {rh} (add)'
                    )
                elif subtype2 == 'con':
                    raise TypeError(
                        f'unsupported order for *: {lh} (add) and {rh} (con)'
                    )

            if lh.attrs['is_ncon'] and rh.attrs['is_ncon']:
                raise TypeError(
                    f'unsupported types for *: {lh} (ncon) and {rh} (ncon), '
                    f'norm convolution can only be used once for one component'
                )

        is_ncon = False

        # determine the subtype
        if subtype1 == 'add' or subtype2 == 'add':
            subtype = 'add'
        else:
            if subtype1 == 'con' or subtype2 == 'con':
                subtype = 'con'
                if lh.attrs['is_ncon'] or rh.attrs['is_ncon']:
                    is_ncon = True
            else:
                subtype = 'mul'

        super().__init__(lh, rh, operator)
        self.attrs['subtype'] = subtype
        self.attrs['is_ncon'] = is_ncon

    def __add__(
        self,
        other: Union[Component, ComponentOperation]
    ) -> ComponentOperation:
        return ComponentOperation(self, other, '+')

    def __mul__(
        self,
        other: Union[Component, ComponentOperation]
    ) -> ComponentOperation:
        return ComponentOperation(self, other, '*')

    @property
    def params(self) -> dict:
        """Get parameter configure."""

        lh, rh = self.predecessor

        return lh.params | rh.params

    @property
    def comps(self) -> dict[str, Component]:
        """Get component dict."""

        lh, rh = self.predecessor

        return lh.comps | rh.comps

    @property
    def func(self):
        """Get evaluation function after the operation."""

        op = self.attrs['name']
        lh, rh = self.predecessor
        m1 = lh.func
        m2 = rh.func
        type1 = lh.attrs['subtype']
        type2 = rh.attrs['subtype']

        # notation: p=params, e=egrid, f=flux, ff=flux_func
        if op == '+':
            def add_add_wrapper(p, e, *_):
                """add + add"""
                return m1(p, e) + m2(p, e)

            return add_add_wrapper

        if type1 != 'con':  # type1 is add or mul
            if type2 != 'con':  # type2 is add or mul
                def op_wrapper(p, e, *_):  # add * add not allowed
                    """add * mul, mul * add, mul * mul"""
                    return m1(p, e) * m2(p, e)

                return op_wrapper

            else:  # type2 is con
                if rh.attrs['is_ncon']:  # type2 is ncon
                    def mul_ncon_wrapper(p, e, f, ff):
                        """mul * ncon"""
                        return m1(p, e) * m2(p, e, f, ff)

                    return mul_ncon_wrapper

                else:  # type2 is con
                    def mul_con_wrapper(p, e, f, *_):
                        """mul * con"""
                        return m1(p, e) * m2(p, e, f)

                    return mul_con_wrapper

        else:  # type1 is con
            if lh.attrs['is_ncon']:  # type1 is ncon
                if type2 == 'add':
                    def ncon_add_wrapper(p, e, *_):
                        """ncon * add"""
                        return m1(p, e, m2(p, e), m2)

                    return ncon_add_wrapper

                elif type2 == 'mul':
                    def ncon_mul_wrapper(e, p, f, ff):
                        """ncon * mul"""
                        def m2_ff(e_, p_, *_):
                            """mul * add, this will be * by ncon"""
                            return m2(e_, p_) * ff(e_, p_)

                        return m1(p, e, m2(e, p) * f, m2_ff)

                    return ncon_mul_wrapper

                else:  # type2 == 'con'
                    def ncon_con_wrapper(p, e, f, ff):
                        """ncon * con"""
                        def m2_ff(p_, e_, *_):
                            """con * add, this will be * by ncon"""
                            return m2(p_, e_, ff(p_, e_))

                        return m1(p, e, m2(p, e, f), m2_ff)

                    return ncon_con_wrapper

            else:  # type1 is con
                if type2 == 'add':
                    def con_add_wrapper(p, e, *_):
                        """con * add"""
                        return m1(p, e, m2(p, e))

                    return con_add_wrapper

                elif type2 == 'mul':
                    def con_mul_wrapper(p, e, f, *_):
                        """con * mul"""
                        return m1(p, e, m2(p, e) * f)

                    return con_mul_wrapper

                else:
                    if rh.attrs['is_ncon']:
                        def con_ncon_wrapper(p, e, f, ff):
                            """con * ncon"""
                            return m1(p, e, m2(p, e, f, ff))

                        return con_ncon_wrapper

                    else:
                        def con_con_wrapper(p, e, f, *_):
                            """con * con"""
                            return m1(p, e, m2(p, e, f))

                        return con_con_wrapper


class LabelSpace:
    """Helper class to handle the name/fmt of composition of components or
    parameters.
    TODO: parameter name/fmt
    """

    name: str
    fmt: str

    def __init__(self, node: NodeABC):
        self.node = node

        self.label_space = {
            'name': self.get_sub_nodes_label('name'),
            'fmt': self.get_sub_nodes_label('fmt')
        }

        self.label_map = {
            'name': self.get_label_map('name'),
            'fmt': self.get_label_map('fmt')
        }

    @staticmethod
    def check_label_type(label_type) -> None:
        """Check if label_type is name/fmt."""

        if label_type != 'name' and label_type != 'fmt':
            raise ValueError('``label_type`` should be "name" or "fmt"')

    def get_sub_nodes_label(self, label_type) -> list[tuple]:
        """Get the address and the name/fmt of sub-nodes."""

        self.check_label_type(label_type)

        labels = []
        node_stack = [self.node]

        while node_stack:
            i = node_stack.pop(0)

            if not i.attrs['is_operation']:
                # record address, name and fmt of the sub-node
                labels.append((i, i.attrs[label_type]))

            else:  # push predecessors of operation node to the node stack
                node_stack = i.predecessor + node_stack

        return labels

    def get_label_map(self, label_type: str) -> dict[str, str]:
        """Solve name/fmt collision of sub-nodes and return suffix mapping."""

        self.check_label_type(label_type)

        label_space = {}
        id_to_str = {}
        node_stack = [self.node]

        while node_stack:
            i = node_stack.pop(0)

            if not i.attrs['is_operation']:
                label = i.attrs[label_type]
                id_ = i.attrs['id']

                # check label collision
                if label not in label_space:  # no label collision found
                    label_space[label] = [id_]  # record label and node id
                    id_to_str[f'{label}_{id_}'] = label

                else:  # there is a label collision
                    same_label_nodes = label_space[label]

                    if id_ not in same_label_nodes:  # not cause by node itself
                        same_label_nodes.append(id_)  # record node id
                        num = len(same_label_nodes)

                        if label_type == 'name':
                            str_ = f'{num}'
                        else:
                            str_ = f'$_{num}$'
                        id_to_str[f'{label}_{id_}'] = f'{label}{str_}'

            else:  # push predecessors of operation to the node stack
                node_stack = i.predecessor + node_stack

        return id_to_str

    def label(self, label_type: str) -> str:
        """Return name/fmt with node id replaced by number"""

        # check if the name/fmt of sub-nodes changed.
        flag = False
        for node, label in self.label_space[label_type]:
            if node.attrs[label_type] != label:
                flag = True
                break

        # if changed, reset label space and id to suffix mapping
        if flag:
            self.label_space[label_type] = self.get_sub_nodes_label(label_type)
            self.label_map[label_type] = self.get_label_map(label_type)

        label = getattr(self.node, label_type)

        for k, v in self.label_map[label_type].items():
            label = label.replace(k, v)

        return label

    @property
    def name(self) -> str:
        """Return name with node id replaced by number."""
        return self.label('name')

    @property
    def fmt(self) -> str:
        """Return fmt with node id replaced by number."""
        return self.label('fmt')


class ComponentMeta(ABCMeta):
    """This metaclass avoids cumbersome code for __init__ of subclass."""

    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # if the component subclass has ``default`` defined correctly,
        # then override the __init__ of subclass
        if hasattr(cls, 'default') and isinstance(cls.default, tuple):
            name = cls.__name__.lower()

            if not hasattr(cls, 'fmt'):  # set fmt to name if not specified
                setattr(cls, 'fmt', name)

            # >>> construct __init__ function >>>
            default = cls.default
            init_def = 'self, *, '
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

            for kw in getattr(cls, 'extra_kwargs', tuple()):
                init_def += f', {kw[0]}={repr(kw[1])}'  # FIXME: repr may fail
                init_body += f', {kw[0]}={kw[0]}'

            func_code = f'def __init__({init_def}):\n    '
            func_code += f'super(type(self), type(self))'
            func_code += f'.__init__(self, {init_body})\n'
            func_code += f'def {name}({par_def}):\n    '
            func_code += f'return {par_body}'
            # <<< construct __init__ function <<<

            # print(func_code)

            # now create and set __init__ function
            exec(func_code, tmp := {})
            init_func = tmp['__init__']
            init_func.__qualname__ = f'{name}.__init__'
            cls.__init__ = init_func
            cls._get_args = tmp[name]


class SpectralModel(metaclass=ComponentMeta):
    """TODO: docstring"""

    type: str
    _node: Union[Component, ComponentOperation]

    def __init__(self, node: Union[Component, ComponentOperation]):
        if not isinstance(node, Component) and \
                not isinstance(node, ComponentOperation):
            raise TypeError('input node must be "component" type')

        self._node = node
        self._type = node.attrs['subtype']
        self._label = LabelSpace(node)
        self._comps = []

    def __add__(self, other: SpectralModel):
        return SpectralModel(self._node + other._node)

    def __mul__(self, other: SpectralModel):
        return SpectralModel(self._node * other._node)

    def __getitem__(self, key: str):
        """Get a parameter of component."""

        if '.' not in key:
            raise ValueError('key format should be "component.parameter"')

        c, p = key.split('.')

        name_map = {v: k for k, v in self._label.label_map['name'].items()}

        if c not in name_map:
            raise ValueError(f'No component "{c}" in model "{self}"')

        return self._node.comps[name_map[c]][p]

    def __setitem__(
        self,
        key: str,
        value: Union[Parameter, ParameterOperation]
    ):
        """Set a parameter of component."""

        if '.' not in key:
            raise ValueError('key format should be "component.parameter"')

        c, p = key.split('.')

        name_map = {v: k for k, v in self._label.label_map['name'].items()}

        if c not in name_map:
            raise ValueError(f'No component "{c}" in model "{self}"')

        self._node.comps[name_map[c]][p] = value

    def __repr__(self):
        return self.expression

    @property
    def type(self):
        """Model type: add, mul, or con"""
        return self._type

    @property
    def params(self):
        """Get parameter configure."""

        name_map = self._label.label_map['name']

        return {
            name_map[k]: v for k, v in self._node.params.items()
        }

    @property
    def comps(self):
        """Additive type of compositions of subcomponents."""

        if self.type != 'add':
            raise TypeError(
                f'"{self.type}" model has no additive sub-components'
            )
        else:
            if not self._comps:
                raise NotImplementedError

            return self._comps

    @property
    def expression(self):
        """Get model expression in plain format."""
        return self._label.name

    @property
    def expression_tex(self):
        """Get model expression in LaTex format."""
        return self._label.fmt

    # def flux(self, params, e_range, energy=True, ngrid=1000, log=True):
    #     """Evaluate model by
    #         * analytic expression
    #         * trapezoid or Simpson's 1/3 rule given :math:`N_E`
    #         * Xspec model library
    #
    #     TODO: docstring
    #
    #     """
    #
    #     if self.type != 'add':
    #         raise TypeError(
    #             f'flux is undefined for "{self.type}" type model "{self}"'
    #         )
    #
    #     if log:  # evenly spaced grid in log space
    #         egrid = np.geomspace(*e_range, ngrid)
    #     else:  # evenly spaced grid in linear space
    #         egrid = np.linspace(*e_range, ngrid)
    #
    #     if energy:  # energy flux
    #         flux = np.sum(self.ENE(pars, ebins) * np.diff(ebins), axis=-1)
    #     else:  # photon flux
    #         flux = np.sum(self.NE(pars, ebins) * np.diff(ebins), axis=-1)
    #
    #     return flux

    def _build_prior(self) -> Callable:
        """Get the prior function which will be used in numpyro."""

        def prior():
            """This should be called inside numpyro."""
            self
            ...

        return prior

    def __call__(self, egrid, flux=None):
        # TODO:
        return self.eval(egrid, flux)

    def eval(self, *args, **kwargs):
        """TODO"""
        if self.type != 'con':
            ...
        else:
            ...

    def _build_func(self) -> Callable:
        """Get the model evaluation function."""

        # mapping: component_num -> component_id
        id_map = {v: k for k, v in self._label.label_map['name'].items()}

        # default parameter configure
        default = {
            c: {
                name: param.default for name, param in params.items()
            } for c, params in self.params.items()
        }

        def get_params(params):
            """Get parameters with id."""
            if params is None:
                # set to default if None
                params = default

            else:
                # get params
                params = {
                    k: v | params[k] if k in params else v
                    for k, v in default.items()
                }

            # map to component_id
            return {id_map[k]: v for k, v in params.items()}

        # wrap the function from Component or ComponentOperation
        f = self._node.func

        def func(egrid, params, *args, **kwargs):
            """Model evaluation wrapper."""
            return f(get_params(params), egrid, *args, **kwargs)

        return func


class SpectralComponentABC(SpectralModel):
    """The abstract spectral component class."""

    fmt: str
    type: str

    # the format is ((name, fmt, default, low, high, frozen, log),)
    default: tuple[tuple]

    # element of inner tuple should respect repr, or meta.__init__ may fail
    extra_kwargs: tuple[tuple]

    def __init__(self, fmt=None, **params):
        if fmt is None:
            fmt = self.fmt

        # parse parameters
        params_dict = {}
        for cfg in self.default:
            param_name = cfg[0]
            param = params[param_name]

            if param is None:
                # use default configure
                params_dict[param_name] = UniformParameter(*cfg)

            elif isinstance(param, Parameter) \
                    or isinstance(param, ParameterOperation):
                # specified by user
                params_dict[param_name] = param

            elif isinstance(param, float) or isinstance(param, int):
                # frozen to the value given by user
                cfg = list(cfg)
                cfg[2] = float(param)
                cfg[5] = True
                params_dict[param_name] = UniformParameter(*cfg)

            else:
                # other input types are not supported yet
                cls_name = type(self).__name__
                type_name = type(param).__name__
                raise TypeError(
                    f'got {type_name} type {repr(param)} for '
                    f'{cls_name}.{param_name}, which is not supported'
                )

        if self.type == 'ncon':  # normalization convolution type
            subtype = 'con'
            is_ncon = True
        else:
            subtype = self.type
            is_ncon = False

        component = Component(
            name=type(self).__name__.lower(),
            fmt=fmt,
            subtype=subtype,
            params=params_dict,
            func=self._eval,
            is_ncon=is_ncon
        )

        super().__init__(component)

    @property
    @abstractmethod
    def default(self):
        """Default configuration of parameters, overriden by subclass."""
        pass

    @staticmethod
    @abstractmethod
    def _eval(*args):
        """Actual evaluation is defined here, overriden by subclass."""
        pass
