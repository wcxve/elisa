import warnings

import numpy as np
import pytensor.tensor as pt
from pytensor import function
from pytensor.gradient import grad_not_implemented


class SpectralParameter:
    r"""
    name : name of parameter
    component : SpectralComponent to which the parameter belong
    root : root TensorVariable, must be overwritten
    root_name : name of root TensorVariable, must be overwritten
    rv : TensorVariable, must be overwritten
    rv_name : name of TensorVariable
    """
    # TODO: transformation from root to rv
    def __init__(self, name, component=None):
        self.component = component
        self._name = None
        self.name = name

    @property
    def name(self):
        if self._name is None:
            return ''
        else:
            return self._name

    @name.setter
    def name(self, value):
        if value is None:
            self._name = None
        else:
            name = str(value)

            # avoid duplicate parameter name in SpectralComponent
            if self._component is not None:
                if name != self.name:
                    alias = self._component.pars_alias
                    if name in alias:
                        raise ValueError(
                            f'name "{name}" already exists in '
                            f'"{self._component}" {tuple(alias)}, where '
                            f'"{self.name}" is current name, please set '
                            'another name'
                        )

            self._name = name

    @property
    def component(self):
        if self._component is None:
            return ''
        else:
            return self._component.name

    @component.setter
    def component(self, value):
        if value is None:
            self._component = None
        elif not isinstance(value, SpectralComponent):
            raise TypeError(
                '"SpectralComponent" type is required for `component`'
            )
        else:
            self._component = value

    @property
    def root(self):
        raise NotImplementedError

    @property
    def root_name(self):
        raise NotImplementedError

    @property
    def root_default(self):
        raise NotImplementedError

    @property
    def rv(self):
        raise NotImplementedError

    @property
    def rv_name(self):
        if self._component is None:
            return self.name
        else:
            return f'{self.component}.{self.name}'

    @property
    def rv_default(self):
        raise NotImplementedError

    @property
    def frozen(self):
        raise NotImplementedError

    def __add__(self, other):
        if not isinstance(other, SpectralParameter):
            type1 = type(self).__name__
            type2 = type(other).__name__
            raise TypeError(
                f"unsupported operand type(s) for +: '{type1}' and '{type2}'"
            )
        return SuperParameter(self, other, '+')

    def __mul__(self, other):
        if not isinstance(other, SpectralParameter):
            type1 = type(self).__name__
            type2 = type(other).__name__
            raise TypeError(
                f"unsupported operand type(s) for *: '{type1}' and '{type2}'"
            )
        return SuperParameter(self, other, '*')

    def __repr__(self):
        return self.rv_name


class SuperParameter(SpectralParameter):
    def __init__(self, p1, p2, operator):
        self._p1 = p1
        self._p2 = p2
        self._operator = operator

        super().__init__(name=None)
        self._set_root_and_rv()

    @SpectralParameter.name.getter
    def name(self):
        if self._name is None:
            return self._get_expr()
        else:
            return self._name

    @SpectralParameter.component.setter
    def component(self, value):
        if value is None:
            self._component = None
        elif not isinstance(value, SpectralComponent):
            raise TypeError(
                '"SpectralComponent" type is required for `component`'
            )
        else:
            self._component = value
            if self._p1.component == '':
                self._p1.component = value
            if self._p2.component == '':
                self._p2.component = value

    @SpectralParameter.root.getter
    def root(self):
        if self._root != self._get_root():
            self._set_root_and_rv()
        return self._root

    @SpectralParameter.root_name.getter
    def root_name(self):
        return self._p1.root_name + self._p2.root_name

    @SpectralParameter.root_default.getter
    def root_default(self):
        return self._p1.root_default + self._p2.root_default

    @SpectralParameter.rv.getter
    def rv(self):
        if self._root != self._get_root():
            self._set_root_and_rv()
        return self._rv

    @SpectralParameter.rv_name.getter
    def rv_name(self):
        if self._component is None:
            return self.name
        else:
            if '+' in self.name or '*' in self.name:
                return f'{self.component}.({self.name})'
            else:
                return f'{self.component}.{self.name}'

    @SpectralParameter.rv_default.getter
    def rv_default(self):
        if self._operator == '+':
            return self._p1.rv_default + self._p2.rv_default
        elif self._operator == '*':
            return self._p1.rv_default * self._p2.rv_default
        else:
            raise ValueError(f'{self._operator} is not supported')

    @SpectralParameter.frozen.getter
    def frozen(self):
        return self._p1.frozen and self._p2.frozen

    def _set_root_and_rv(self):
        self._root = self._get_root()

        if self._operator == '+':
            self._rv = self._p1.rv + self._p2.rv

        elif self._operator == '*':
            self._rv = self._p1.rv * self._p2.rv

        else:
            raise ValueError(f'{self._operator} is not supported')

    def _get_root(self):
        root_list = self._p1.root + self._p2.root

        root = []

        for i in range(len(root_list)):
            r = root_list[i]
            if r not in root:
                root.append(r)

        return root

    def _get_expr(self):
        if self._operator == '+':
            return f'{self._p1.rv_name} + {self._p2.rv_name}'

        elif self._operator == '*':
            name1 = self._p1.rv_name
            name2 = self._p2.rv_name

            if isinstance(self._p1, SuperParameter) \
                and self._p1._operator == '+' \
                and '+' in self._p1.name:
                name1 = f'({name1})'

            if isinstance(self._p2, SuperParameter) \
                and self._p2._operator == '+' \
                and '+' in self._p2.name:
                name2 = f'({name2})'

            return f'{name1} * {name2}'

        else:
            raise ValueError(f'{self._operator} is not supported')


class UniformParameter(SpectralParameter):
    def __init__(
        self, name, default, min, max,
        frozen=False, log=False, component=None
    ):
        super().__init__(name, component)
        self._frozen = None
        self._log = None

        self.values = default, min, max
        self.frozen = frozen
        self.log = log

        self._set_root_and_rv()

    @SpectralParameter.root.getter
    def root(self):
        if self._root is None:
            self._set_root_and_rv()
        return self._root

    @SpectralParameter.root_name.getter
    def root_name(self):
        if self.frozen:
            return []
        else:
            if self.log:
                return [self.rv_name.replace(self.name, f'__ln({self.name})')]
            else:
                return [self.rv_name.replace(self.name, f'__{self.name}')]

    @SpectralParameter.root_default.getter
    def root_default(self):
        if self.log:
            return [np.log(self.default)]
        else:
            return [self.default]

    @SpectralParameter.rv.getter
    def rv(self):
        if self._rv is None:
            self._set_root_and_rv()
        return self._rv

    @SpectralParameter.rv_default.getter
    def rv_default(self):
        return self.default

    @property
    def default(self):
        return self._default

    @default.setter
    def default(self, value):
        self._check_and_set_values(default=value)

    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, value):
        self._check_and_set_values(min=value)

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, value):
        self._check_and_set_values(max=value)

    @property
    def values(self):
        return self.default, self.min, self.max

    @values.setter
    def values(self, value):
        values = np.atleast_1d(value)
        if len(values) == 1:
            self._check_and_set_values(default=values[0])
        elif len(values) == 2:
            self._check_and_set_values(min=values[0], max=values[1])
        elif len(values) == 3:
            self._check_and_set_values(*values)
        else:
            raise ValueError(f'wrong values ({value})')

    @property
    def frozen(self):
        return self._frozen

    @frozen.setter
    def frozen(self, value):
        flag = bool(value)
        if self._frozen != flag:
            self._frozen = flag
            self._reset_root_and_rv()

    @property
    def log(self):
        return self._log

    @log.setter
    def log(self, value):
        log_flag = bool(value)

        if self._log != log_flag:
            if log_flag and self.min <= 0.0:
                raise ValueError(
                    f'"{self.name}" can not be log-parameterized for its '
                    f'minimum being non-positive ({self.min})'
                )

            self._log = log_flag
            self._reset_root_and_rv()

    def _check_and_set_values(self, default=None, min=None, max=None):
        if default is None:
            _default = self.default
        else:
            _default = default

        if min is None:
            _min = self.min
        else:
            _min = min

        if max is None:
            _max = self.max
        else:
            _max = max

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
            self._default = default

        if min is not None:
            self._min = min

        if max is not None:
            self._max = max

        if (min is not None) or (max is not None):
            self._reset_root_and_rv()

    def _set_root_and_rv(self):
        if self.frozen:
            self._root = []
            self._rv = pt.constant(self.default)
        else:
            if self.log:
                log_rv = pt.random.uniform(np.log(self.min), np.log(self.max))
                rv = pt.exp(log_rv)
                self._root = [log_rv]
                self._rv = rv
            else:
                rv = pt.random.uniform(self.min, self.max)
                self._root = [rv]
                self._rv = rv

    def _reset_root_and_rv(self):
        self._root = None
        self._rv = None


class NumericGradOp(pt.Op):
    optype = None
    def __init__(self, pars, grad_method='c', eps=1e-7):
        if self.optype is None:
            raise TypeError(
                '`optype` must be specified, supported are "add", "mul", and '
                '"con"'
            )

        if self.optype not in ['add', 'mul', 'con']:
            raise ValueError(
                f'wrong value ({self.optype}) for `optype`, supported are '
                '"add", "mul", and "con"'
            )

        self._pars = [
            p if type(p) == pt.TensorVariable
            else pt.constant(p, dtype='floatX')
            for p in pars
        ]
        self._npars = len(pars)
        self.grad_method = grad_method
        self.eps = eps
        self.otypes = [pt.TensorType('floatX', shape=(None,))]

    def __call__(self, ebins, flux=None):
        if type(ebins) not in [pt.TensorVariable,
                               pt.sharedvar.TensorSharedVariable]:
            ebins = pt.constant(ebins, dtype='floatX')

        if self.optype != 'con':
            return super().__call__(*self._pars, ebins)
        else:
            if flux is None:
                raise ValueError('`flux` is required for convolution model')
            if type(flux) != pt.TensorVariable:
                flux = pt.constant(flux, dtype='floatX')

            return super().__call__(*self._pars, ebins, flux)

    @property
    def eps(self):
        return self._eps.value

    @eps.setter
    def eps(self, value):
        self._eps = pt.constant(value, dtype='floatX')

    @property
    def grad_method(self):
        return self._grad_method

    @grad_method.setter
    def grad_method(self, value):
        if value not in ['b', 'c', 'f', 'n']:
            raise ValueError(
                f'wrong value ({value}) for `grad_method`, supported '
                'difference approximation types are "c" for central, '
                '"f" for forward, "b" for backward, and "n" for no gradient'
            )
        else:
            self._grad_method = value

    @property
    def npars(self):
        return self._npars

    def perform(self, node, inputs, output_storage, params=None):
        # the last element of inputs is ebins
        # returns model value
        output_storage[0][0] = self._perform(*inputs)

    def grad(self, inputs, output_grads):
        # the last element of inputs is ebins
        # returns grad Op in backward mode
        if self.grad_method not in ['c', 'n'] \
                or (self.grad_method != 'n' and self.optype == 'con'):
            self._tensor_output = self._create_tensor(*inputs)

        return [
            self._grad_for_inputs(inputs, index, output_grads[0])
            for index in range(len(inputs))
        ]

    def _perform(self, *args):
        raise NotImplementedError

    def _create_tensor(self, *args):
        # in this case, pars is included in args
        return super().__call__(*args)

    def _grad_for_inputs(self, inputs, index, output_grad):
        # https://en.wikipedia.org/wiki/Finite_difference
        # https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf
        if index == self.npars:  # case for input is ebins
            return grad_not_implemented(self, index, inputs[index])
        elif index == self.npars + 1:  # case for input is flux
            if self.grad_method != 'n':
                # TODO: numeric gradient for convolution is the hardest part
                warnings.warn(
                    'gradient for convolution component is not implemented',
                    GradientWarning
                )
                return grad_not_implemented(self, index, inputs[index])
            else:
                return grad_not_implemented(self, index, inputs[index])

        pars = inputs[:self.npars]
        others = inputs[self.npars:]  # ebins, and possibly flux if "con" model

        if self.grad_method == 'f':
            # forward difference approximation
            pars[index] = pars[index] + self._eps
            flux_eps = self._create_tensor(*pars, *others)
            g = (flux_eps - self._tensor_output) / self._eps
            return pt.dot(output_grad, g)
        elif self.grad_method == 'c':
            # central difference approximation, accurate when compute hessian
            par_i = pars[index]
            pars[index] = par_i + self._eps
            flux_peps = self._create_tensor(*pars, *others)
            pars[index] = par_i - self._eps
            flux_meps = self._create_tensor(*pars, *others)
            g = (flux_peps - flux_meps) / (2.0 * self._eps)
            return pt.dot(output_grad, g)
        elif self.grad_method == 'b':
            # backward difference approximation
            pars[index] = pars[index] - self._eps
            flux_eps = self._create_tensor(*pars, *others)
            g = (self._tensor_output - flux_eps) / self._eps
            return pt.dot(output_grad, g)
        else:
            return grad_not_implemented(self, index, inputs[index])


class AutoGradOp:
    optype = None
    def __init__(self, pars, integral_method='trapz'):
        if self.optype is None:
            raise TypeError(
                '`optype` must be specified, supported are "add", "mul", and '
                '"con"'
            )

        if self.optype not in ['add', 'mul', 'con']:
            raise ValueError(
                f'wrong value ({self.optype}) for `optype`, supported are '
                '"add", "mul", and "con"'
            )

        self._pars = [
            p if type(p) == pt.TensorVariable
            else pt.constant(p, dtype='floatX')
            for p in pars
        ]
        self.integral_method = integral_method

    def __call__(self, ebins, flux=None):
        if type(ebins) not in [pt.TensorVariable,
                               pt.sharedvar.TensorSharedVariable]:
            ebins = pt.constant(ebins, dtype='floatX')

        if self.optype == 'add':
            return self._eval_flux(ebins)
        elif self.optype == 'mul':
            return self._eval(ebins)
        else:
            if flux is None:
                raise ValueError('`flux` is required for convolution model')
            if type(flux) != pt.TensorVariable:
                flux = pt.constant(flux, dtype='floatX')

            return self._eval(ebins, flux)

    @property
    def integral_method(self):
        return self._method

    @integral_method.setter
    def integral_method(self, value):
        if value not in ['trapz', 'simpson']:
            raise ValueError(
                f'wrong value ({value}) for `integral_method`, supported '
                'are "trapz" and "simpson"'
            )
        else:
            self._method = value

    def _eval_flux(self, ebins):
        if self.optype == 'add':
            if self.integral_method == 'trapz':
                dE = ebins[1:] - ebins[:-1]
                NE = self._NE(ebins)
                flux = (NE[:-1] + NE[1:]) / 2.0 * dE
            else:  # simpson's 1/3 rule
                dE = ebins[1:] - ebins[:-1]
                E_mid = (ebins[:-1] + ebins[1:]) / 2.0
                NE = self._NE(ebins)
                NE_mid = self._NE(E_mid)
                flux = dE / 6.0 * (NE[:-1] + 4.0 * NE_mid + NE[1:])
        else:
            raise TypeError(f'flux is undefined for "{self.optype}" model')

        return flux

    def _NE(self, ebins):
        raise NotImplementedError('NE is not defined')

    def _eval(self, ebins, flux=None):
        raise NotImplementedError('eval is not defined')


class GradientWarning(Warning):
    """
    issued by no implementation of gradient
    """
    pass


class SpectralComponent:
    r"""
    provide model tensor

    name :
    parameters :
    parameters_name : [name of params]
    op_class : __call__ will return an op based on the parameters,
                and the __call__ on op will return model tensor
    _root : root and corresponding name
    """
    _comp_name = None   # overwritten by subclass
    _config = None      # overwritten by subclass
    _op_class = None  # overwritten by subclass
    def __init__(self, **kwargs):
        if self._comp_name is None \
                or self._config is None \
                or self._op_class is None:
            raise NotImplementedError(
                'SpectralComponent can only be initialized by subclass'
            )

        if self._op_class.optype not in ['add', 'mul', 'con']:
            raise TypeError(
                f'"{self._op_class.optype}" type op is not supported, '
                'supported are "add", "mul", and "con"'
            )

        self.name = kwargs.pop('name')

        if self._op_class.optype == 'add' \
            and ('norm' not in self._config.keys()
                 or 'norm' not in kwargs.keys()):
            raise ValueError(
                '"norm" is required for "add" type SpectralComponent '
                f'{self.name}'
            )

        self._pars_dict = {}
        for p in self._config.keys():
            self._set_par(p, kwargs[p])

        self._op_kwargs = {}
        for k, v in kwargs.items():
            if k not in self._config:
                self._op_kwargs[k] = v

        self._pars_tensor = {
            name: pt.scalar(par.name)
            for name, par in self._pars_dict.items()
        }

    def __call__(self, ebins, flux=None, fit_call=True):
        if fit_call:
            rv = {name: par.rv for (name, par) in self._pars_dict.items()}
        else:
            rv = self.pars_tensor

        if self.mtype == 'add':
            norm = rv.pop('norm')
            return norm * self._op_class(**rv, **self._op_kwargs)(ebins, flux)
        else:
            return self._op_class(**rv, **self._op_kwargs)(ebins, flux)

    @property
    def name(self):
        if self._name is None:
            return self._comp_name
        else:
            return self._name

    @name.setter
    def name(self, value):
        if value is None:
            self._name = None
        else:
            self._name = str(value)

    @property
    def mtype(self):
        return self._op_class.optype

    @property
    def pars_name(self):
        return list(self._config.keys())

    @property
    def pars_alias(self):
        return [
            p.name for p in self._pars_dict.values() if p._component is self
        ]

    @property
    def pars_tensor(self):
        for name, par in self._pars_dict.items():
            self._pars_tensor[name].name = par.rv_name

        return dict(self._pars_tensor)

    def _set_par(self, p_name, p_input):
        if p_input is None:
            par = UniformParameter(p_name, *self._config[p_name], self)
        elif type(p_input) in [float, int]:
            value = float(p_input)
            config = (value, value, value, True, False)
            par = UniformParameter(p_name, *config, self)
        else:
            if not isinstance(p_input, SpectralParameter):
                raise TypeError(
                    '"SpectralParameter" type is required for '
                    f'`{self.name}.{p_name}`'
                )
            par = p_input

        if par is self._pars_dict.get(p_name):
            return

        if par.component == '':
            par.component = self

        # avoid duplicate parameter name in SpectralComponent
        alias = self.pars_alias
        if (par._component is self) and (par.name in self.pars_alias):
            raise ValueError(
                f'name "{par.name}" already exists in "{self}" {tuple(alias)},'
                f' please set another name for parameter "{p_input}"'
            )

        if par._name is None:
            par._name = p_name

        self._pars_dict[p_name] = par

        super().__setattr__(p_name, par)

    def __setattr__(self, key, value):
        if key in self._config.keys():
            self._set_par(key, value)
        else:
            super().__setattr__(key, value)

    def __repr__(self):
        return self.name


_EBINS = pt.vector('ebins')
_EMID2 = _EBINS[:-1] * _EBINS[1:]
_CH_EMIN = pt.vector('ch_emin')
_CH_EMAX = pt.vector('ch_emax')
_RESP_MATRIX = pt.matrix('resp_matrix')
_FLUX = pt.vector('flux')

class SpectralModel:
    # 初始化时，初始化内部的tensorOp，该op输入参数rv，输出能谱rv
    # 该op另外有可调参数，如数值微分方式、微分步长、数值积分方式
    # 记录名字、成分
    # Model之间运算时，形成新的Model，新的Model的rv和root由旧的叠加，记录成分
    # 三个内部property: op、rv、root
    # 外部property: expression, comps_name, component
    # DONE: components具有重复名称时，后缀加_2, _3, ...
    # DONE: comps_name
    # DONE: 动态expression
    # DONE: root, root_name
    # DONE: rv, rv_name
    # DONE: CE NE ENE EENE
    # DONE: 叠加成分的CE NE ENE EENE
    # DONE: flux计算
    def __init__(self, components):
        self._components = components

        self._ebins = _EBINS
        self._emid2 = _EMID2
        self._ch_emin = _CH_EMIN
        self._ch_emax = _CH_EMAX
        self._resp_matrix = _RESP_MATRIX
        self._flux = _FLUX

        self.__eval_tensor = None
        self.__CE_tensor = None
        self.__NE_tensor = None
        self.__ENE_tensor = None
        self.__EENE_tensor = None

        self._eval_func = None
        self._NE_func = None
        self._CE_func = None
        self._ENE_func = None
        self._EENE_func = None

        self._CE_comps_func = None
        self._NE_comps_func = None
        self._ENE_comps_func = None
        self._EENE_comps_func = None

        for c, name in zip(components, self.comps_name):
            super().__setattr__(name, c)

    def __call__(self, ebins, flux=None, fit_call=True):
        return self._components[0](ebins, flux, fit_call)

    @property
    def expression(self):
        return self._components[0].name

    @property
    def mtype(self):
        return self._components[0].mtype

    @property
    def comps_name(self):
        return [c._comp_name for c in self._components]

    @property
    def root(self):
        root = []
        name = []
        default = []

        for c in self._components:
            for par in c._pars_dict.values():
                for i, j, k in zip(par.root, par.root_name, par.root_default):
                    if i not in root:
                        root.append(i)
                        name.append(j)
                        default.append(k)

        root_dict = {
            'root': tuple(root),
            'name': tuple(name),
            'default': tuple(default)
        }

        return root_dict

    @property
    def params(self):
        name = []
        default = []
        frozen = []
        rv = []
        sup = []

        for c in self._components:
            for par in c._pars_dict.values():
                name.append(par.rv_name)
                default.append(par.rv_default)
                frozen.append(par.frozen)
                rv.append(par.rv)
                sup.append(isinstance(par, SuperParameter))

        rv_dict = {
            'name': tuple(name),
            'default': tuple(default),
            'frozen': tuple(frozen),
            'rv': tuple(rv),
            'super': tuple(sup)
        }

        return rv_dict

    @property
    def _pars_tensor(self):
        return [i for c in self._components for i in c.pars_tensor.values()]

    @property
    def _eval_tensor(self):
        if self.__eval_tensor is None:
            if self.mtype != 'con':
                self.__eval_tensor = self(self._ebins, fit_call=False)
            else:
                self.__eval_tensor = self(self._ebins, self._flux, False)

        return self.__eval_tensor

    @property
    def _NEdE_tensor(self):
        return self._eval_tensor

    @property
    def _CE_tensor(self):
        if self.__CE_tensor is None:
            ch_emin = self._ch_emin
            ch_emax = self._ch_emax
            resp = self._resp_matrix
            dch = ch_emax - ch_emin
            NEdE = self._NEdE_tensor
            self.__CE_tensor = pt.dot(NEdE, resp) / dch

        return self.__CE_tensor

    @property
    def _NE_tensor(self):
        if self.__NE_tensor is None:
            ebins = self._ebins
            dE = ebins[1:] - ebins[:-1]
            NEdE = self._NEdE_tensor
            self.__NE_tensor = NEdE / dE

        return self.__NE_tensor

    @property
    def _ENE_tensor(self):
        if self.__ENE_tensor is None:
            E = pt.sqrt(self._emid2)
            NE = self._NE_tensor
            self.__ENE_tensor = E * NE * 1.6022e-9  # keV to erg

        return self.__ENE_tensor

    @property
    def _EENE_tensor(self):
        if self.__EENE_tensor is None:
            EE = self._emid2
            NE = self._NE_tensor
            self.__EENE_tensor = EE * NE * 1.6022e-9  # keV to erg

        return self.__EENE_tensor

    @property
    def _model_comps(self):
        return [self]

    @property
    def _CE_comps(self):
        return {m.expression: m._CE_tensor for m in self._model_comps}

    @property
    def _NE_comps(self):
        return {m.expression: m._NE_tensor for m in self._model_comps}

    @property
    def _ENE_comps(self):
        return {m.expression: m._ENE_tensor for m in self._model_comps}

    @property
    def _EENE_comps(self):
        return {m.expression: m._EENE_tensor for m in self._model_comps}

    def _call_func(self, pars, func, **kwargs):
        pars = np.asarray(pars)
        if len(pars.shape) == 1:
            return func(*pars, **kwargs)
        elif len(pars.shape) == 2:
            return np.array([func(*p, **kwargs) for p in pars])
        else:
            raise ValueError('pars should be 1 or 2 dimensional')

    def _call_func_with_comps(self, pars, func, **kwargs):
        pars = np.asarray(pars)
        if len(pars.shape) == 1:
            return func(*pars, **kwargs)
        elif len(pars.shape) == 2:
            res = [func(*p, **kwargs) for p in pars]
            return {
                name: np.array([r[name] for r in res])
                for name in [m.expression for m in self._model_comps]
            }
        else:
            raise ValueError('pars should be 1 or 2 dimensional')

    def CE(self, pars, ebins, ch_emin, ch_emax, resp_matrix, comps=False):
        if self.mtype != 'add':
            raise TypeError(
                f'CE is undefined for "{self.mtype}" type model "{self}"'
            )

        kwargs = dict(
            ebins=ebins,
            ch_emin=ch_emin,
            ch_emax=ch_emax,
            resp_matrix=resp_matrix
        )

        if comps:
            if self._CE_comps_func is None:
                CE_inputs = [
                    self._ebins, self._ch_emin, self._ch_emax,
                    self._resp_matrix
                ]
                inputs = self._pars_tensor + CE_inputs
                self._CE_comps_func = function(inputs, self._CE_comps)
            return self._call_func_with_comps(pars, self._CE_comps_func,
                                              **kwargs)
        else:
            if self._CE_func is None:
                CE_inputs = [
                    self._ebins, self._ch_emin, self._ch_emax, self._resp_matrix
                ]
                inputs = self._pars_tensor + CE_inputs
                self._CE_func = function(inputs, self._CE_tensor)
            return self._call_func(pars, self._CE_func, **kwargs)

    def counts(self, pars, ebins, ch_emin, ch_emax, resp_matrix, exposure, comps=False):
        if self.mtype != 'add':
            raise TypeError(
                f'counts is undefined for "{self.mtype}" type model "{self}"'
            )

        CE = self.CE(pars, ebins, ch_emin, ch_emax, resp_matrix, comps)
        if comps:
            return {
                k: v * (ch_emax - ch_emin) * exposure
                for k, v in CE.items()
            }
        else:
            return CE * (ch_emax - ch_emin) * exposure

    def NE(self, pars, ebins, comps=False):
        if self.mtype != 'add':
            raise TypeError(
                f'NE is undefined for "{self.mtype}" type model "{self}"'
            )

        if comps:
            if self._NE_comps_func is None:
                inputs = self._pars_tensor + [self._ebins]
                self._NE_comps_func = function(inputs, self._NE_comps)
            return self._call_func_with_comps(pars, self._NE_comps_func,
                                              ebins=ebins)
        else:
            if self._NE_func is None:
                inputs = self._pars_tensor + [self._ebins]
                self._NE_func = function(inputs, self._NE_tensor)
            return self._call_func(pars, self._NE_func, ebins=ebins)

    def ENE(self, pars, ebins, comps=False):
        if self.mtype != 'add':
            raise TypeError(
                f'NE is undefined for "{self.mtype}" type model "{self}"'
            )

        if comps:
            if self._ENE_comps_func is None:
                inputs = self._pars_tensor + [self._ebins]
                self._ENE_comps_func = function(inputs, self._ENE_comps)
            return self._call_func_with_comps(pars, self._ENE_comps_func,
                                              ebins=ebins)
        else:
            if self._ENE_func is None:
                inputs = self._pars_tensor + [self._ebins]
                self._ENE_func = function(inputs, self._ENE_tensor)
            return self._call_func(pars, self._ENE_func, ebins=ebins)

    def EENE(self, pars, ebins, comps=False):
        if self.mtype != 'add':
            raise TypeError(
                f'EENE is undefined for "{self.mtype}" type model "{self}"'
            )

        if comps:
            if self._EENE_comps_func is None:
                inputs = self._pars_tensor + [self._ebins]
                self._EENE_comps_func = function(inputs, self._EENE_comps)
            return self._call_func_with_comps(pars, self._EENE_comps_func,
                                              ebins=ebins)
        else:
            if self._EENE_func is None:
                inputs = self._pars_tensor + [self._ebins]
                self._EENE_func = function(inputs, self._EENE_tensor)
            return self._call_func(pars, self._EENE_func, ebins=ebins)

    def eval(self, pars, ebins, flux=None):
        if self._eval_func is None:
            if self.mtype != 'con':
                inputs = self._pars_tensor + [self._ebins]
            else:
                inputs = self._pars_tensor + [self._ebins, self._flux]

            self._eval_func = function(inputs, self._eval_tensor)

        if self.mtype != 'con':
            return self._call_func(pars, self._eval_func, ebins=ebins)
        else:
            if flux is None:
                raise ValueError(
                    f'flux input is required for "con" type model "{self}"'
                )
            return self._call_func(pars, self._eval_func, ebins=ebins, flux=flux)

    def flux(self, pars, erange, ngrid=1000, log=True, energy=True):
        if self.mtype != 'add':
            raise TypeError(
                f'flux is undefined for "{self.mtype}" type model "{self}"'
            )

        if log:  # evenly-spaced energies in log space
            ebins = np.geomspace(*erange, ngrid)
        else:  # evenly-spaced energies in linear space
            ebins = np.linspace(*erange, ngrid)

        if energy:  # photon flux
            flux = np.sum(self.ENE(pars, ebins) * np.diff(ebins), axis=-1)
        else:  # energy flux
            flux = np.sum(self.NE(pars, ebins) * np.diff(ebins), axis=-1)

        return flux

    def __add__(self, other):
        if isinstance(other, SpectralModel):
            if self.mtype != 'add':
                raise TypeError(f'model ({self}) is not additive')

            if other.mtype != 'add':
                raise TypeError(f'model ({other}) is not additive')
        else:
            raise TypeError(
                f'"SpectralModel" is required for "+", got "{other}"'
            )

        return SuperModel(self, other, '+')

    def __mul__(self, other):
        if isinstance(other, SpectralModel):
            if self.mtype == 'add':
                if other.mtype == 'add':
                    raise TypeError(
                        'unsupported operand types for *: "additive" and '
                        '"additive"'
                    )
                elif other.mtype == 'con':
                    raise TypeError(
                        'unsupported operand order for *: "additive" and '
                        '"convolution"'
                    )
        else:
            raise TypeError(
                f'"SpectralModel" is required for "*", got "{other}"'
            )

        return SuperModel(self, other, '*')

    def __radd__(self, other):
        raise TypeError(
            f'"SpectralModel" is required for "+", got "{other}"'
        )

    def __rmul__(self, other):
        raise TypeError(
            f'"SpectralModel" is required for "*", got "{other}"'
        )

    def __setattr__(self, key, value):
        if key in ['_components', '_comps_name']:
            super().__setattr__(key, value)
        elif key in self.comps_name:
            raise AttributeError(
                f"can't set read-only attribute '{key}'"
            )
        else:
            super().__setattr__(key, value)

    def __repr__(self):
        return self.expression


from .flux_model import FluxModel


class SuperModel(SpectralModel):
    def __init__(self, m1, m2, operator, rename=True):
        components = []
        comps_name = []

        for c in m1._components + m2._components:
            if c in components:
                continue

            components.append(c)

            if rename:  # avoid duplicate name or alias
                name = c._comp_name
                flag = True
                n = 1
                while flag:
                    flag = name in comps_name
                    if not flag:
                        break
                    else:
                        n += 1
                    name = f'{c._comp_name}_{n}'
                c.name = name
            else:
                name = c.name

            comps_name.append(name)

        self._comps_name = comps_name
        self._m1 = m1
        self._m2 = m2
        self._operator = operator
        super().__init__(components)

    def __call__(self, ebins, flux=None, model=None, fit_call=True):
        m1 = lambda *args: self._m1(*args, fit_call=fit_call)
        m2 = lambda *args: self._m2(*args, fit_call=fit_call)

        if self._operator == '+':
            return m1(ebins) + m2(ebins)  # add+add

        if self._m1.mtype != 'con':
            if self._m2.mtype != 'con':
                return m1(ebins) * m2(ebins)  # add*mul, mul*add, mul*mul
            else:  # mul*con
                if not isinstance(self._m2, FluxModel):
                    return m1(ebins) * m2(ebins, flux)
                else:
                    return m1(ebins) * m2(flux, model)
        else:
            if not isinstance(self._m1, FluxModel):
                if self._m2.mtype == 'add':  # con*add
                    return m1(ebins, m2(ebins))
                elif self._m2.mtype == 'mul':  # con*mul
                    return m1(ebins, m2(ebins) * flux)
                else:  # con*con
                    if not isinstance(self._m2, FluxModel):
                        return m1(ebins, m2(ebins, flux))
                    else:
                        return m1(ebins, m2(flux, model))
            else:
                if self._m2.mtype == 'add':  # con*add
                    return m1(m2(ebins), self._m2)
                elif self._m2.mtype == 'mul':  # con*mul
                    return m1(m2(ebins) * flux, self._m2 * model)
                else:  # con*con
                    if not isinstance(self._m2, FluxModel):
                        return m1(m2(ebins, flux), self._m2 * model)
                    else:
                        return m1(m2(flux, model), self._m2 * model)

    @SpectralModel.expression.getter
    def expression(self):
        if self._operator == '+':
            return f'{self._m1.expression} + {self._m2.expression}'

        elif self._operator == '*':
            expr1 = self._m1.expression
            expr2 = self._m2.expression

            if isinstance(self._m1, SuperModel) and self._m1._operator == '+' \
                and '+' in expr1:
                expr1 = f'({expr1})'

            if isinstance(self._m2, SuperModel) and self._m2._operator == '+' \
                and '+' in expr2:
                expr2 = f'({expr2})'

            return f'{expr1} * {expr2}'

    @SpectralModel.mtype.getter
    def mtype(self):
        if self._m1.mtype == 'add' or self._m2.mtype == 'add':
            return 'add'
        else:
            if self._m1.mtype == 'con' or self._m2.mtype == 'con':
                return 'con'
            else:
                return 'mul'

    @SpectralModel.comps_name.getter
    def comps_name(self):
        return self._comps_name

    @SpectralModel._model_comps.getter
    def _model_comps(self):
        if self._operator == '+':  # add+add
            return self._m1._model_comps + self._m2._model_comps

        if self._m1.mtype == 'add':  # add*mul
            m2_comp = self._m2._model_comps[0]
            return [
                SuperModel(i, m2_comp, '*', False)
                for i in self._m1._model_comps
            ]
        elif self._m2.mtype == 'add':  # mul*add, con*add
            m1_comp = self._m1._model_comps[0]
            return [
                SuperModel(m1_comp, i, '*', False)
                for i in self._m2._model_comps
            ]
        else:  # mul*mul, con*mul, mul*con, con*con
            return [
                SuperModel(
                    self._m1._model_comps[0],
                    self._m2._model_comps[0],
                    '*',
                    False
                )
            ]
