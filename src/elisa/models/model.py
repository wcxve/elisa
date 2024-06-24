"""The spectral model bases."""

from __future__ import annotations

import inspect
import warnings
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Mapping, Sequence
from enum import Enum
from functools import reduce
from typing import TYPE_CHECKING, NamedTuple

import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
from astropy.cosmology import Planck18
from scipy.sparse import sparray

from elisa.data.base import ObservationData, ResponseData, SpectrumData
from elisa.models.parameter import Parameter, UniformParameter
from elisa.util.misc import build_namespace, define_fdjvp, make_pretty_table

if TYPE_CHECKING:
    from typing import Any, Callable, Literal

    from astropy.cosmology.flrw.lambdacdm import LambdaCDM
    from numpyro.distributions import Distribution

    from elisa.models.parameter import ParamInfo
    from elisa.util.integrate import IntegralFactory
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
        ParamID,
        ParamIDStrMapping,
        ParamIDValMapping,
        ParamNameValMapping,
    )

    NDArray = np.ndarray


class Model(ABC):
    """Base model class."""

    _comps: tuple[Component, ...]
    _additive_comps: tuple[Model, ...]
    __initialized: bool = False

    def __init__(
        self,
        name: str,
        latex: str,
        comps: list[Component],
    ):
        self._name = str(name)
        self._latex = str(latex)
        self._comps = tuple(comps)
        self._comps_id = tuple(comp._id for comp in comps)

        cname = build_namespace([c.name for c in comps])['namespace']
        cid_to_cname = dict(zip(self._comps_id, cname))
        self._cid_to_cname = cid_to_cname

        self.__name = self._id_to_label(cid_to_cname, 'name')

        for name, comp in zip(cid_to_cname.values(), comps):
            setattr(self, name, comp)

        self.__initialized = True

    @property
    def _cid_to_clatex(self) -> CompIDStrMapping:
        clatex = [c.latex for c in self._comps]
        clatex = build_namespace(clatex, latex=True)['namespace']
        return dict(zip(self._comps_id, clatex))

    def compile(self, *, model_info: ModelInfo | None = None) -> CompiledModel:
        """Compile the model for fast evaluation.

        Parameters
        ----------
        model_info : ModelInfo, optional
            Optional model information used to compile the model.

        Returns
        -------
        CompiledModel
            The compiled model.
        """
        if self.type == 'conv':
            raise RuntimeError('cannot compile convolution model')

        if model_info is None:
            model_info = get_model_info(
                self._comps, self._cid_to_cname, self._cid_to_clatex
            )
        else:
            if not isinstance(model_info, ModelInfo):
                raise TypeError('`model_info` must be a ModelInfo instance')

            if not set(self._comps_id).issubset(set(model_info.cid_to_params)):
                raise ValueError('inconsistent model information')

        # model name
        name = self._id_to_label(model_info.cid_to_name, 'name')

        # model parameter id
        params_id = []
        fixed_id = []
        for c in self._comps:
            for p in c.param_names:
                for pid in c[p]._nodes_id:
                    if pid in model_info.sample:
                        params_id.append(pid)
                    elif pid in model_info.fixed:
                        fixed_id.append(pid)

        # compiled model evaluation function
        fn = self._compile_model_fn(model_info)
        additive_fn = self._compile_additive_fn(model_info)

        # model type
        mtype = self.type

        return CompiledModel(
            name, params_id, fixed_id, fn, additive_fn, mtype, model_info
        )

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
        latex = [c.latex for c in self._comps]
        cid_to_latex = dict(
            zip(
                self._comps_id,
                build_namespace(latex, latex=True)['namespace'],
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
    def _additive_comps(self) -> tuple[Model, ...]:
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
            c._id: model_info.cid_to_params[c._id] for c in self._comps
        }

        eval_fn = jax.jit(self.eval)

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
        comps_labels = [
            comp._id_to_label(model_info.cid_to_latex, 'latex')
            for comp in additive_comps
        ]

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

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        fields = ['No.', 'Component', 'Parameter', 'Value', 'Bound', 'Prior']
        info = get_model_info(
            self._comps, self._cid_to_cname, self._cid_to_clatex
        ).info
        tab_params = make_pretty_table(fields, info)
        return (
            f'Model: {self.name} [{self.type}]\n' f'{tab_params.get_string()}'
        )

    def _repr_html_(self) -> str:
        fields = ['No.', 'Component', 'Parameter', 'Value', 'Bound', 'Prior']
        info = get_model_info(
            self._comps, self._cid_to_cname, self._cid_to_clatex
        ).info
        tab_params = make_pretty_table(fields, info)
        return (
            '<details open>'
            f'<summary><b>Model: {self.name} [{self.type}]</b></summary>'
            f'{tab_params.get_html_string(format=True)}'
            '</details>'
        )

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

    def __getitem__(self, key: str) -> Component:
        if key not in self._cid_to_cname.values():
            raise KeyError(key)

        return getattr(self, key)

    def __add__(self, other: Model) -> CompositeModel:
        return CompositeModel(self, other, '+')

    def __radd__(self, other: Model) -> CompositeModel:
        return CompositeModel(other, self, '+')

    def __mul__(self, other: Model) -> CompositeModel:
        return CompositeModel(self, other, '*')

    def __rmul__(self, other: Model) -> CompositeModel:
        return CompositeModel(other, self, '*')


class CompiledModel:
    """Model with fast evaluation and fixed configuration."""

    __initialized: bool = False

    def __init__(
        self,
        name: str,
        params_id: Sequence[ParamID],
        fixed_id: Sequence[ParamID],
        fn: ModelCompiledFn,
        additive_fn: AdditiveFn | None,
        mtype: Literal['add', 'mul'],
        model_info: ModelInfo,
    ):
        pname_to_pid = {model_info.name[pid]: pid for pid in params_id}
        self.name = name
        self._pname_to_pid = pname_to_pid
        self._params_name = tuple(pname_to_pid)
        self._params_id = params_id
        self._params_default = dict(model_info.default)
        self._value_sequence_to_params: Callable[
            [Sequence[JAXFloat]], ParamIDValMapping
        ] = jax.jit(lambda sequence: dict(zip(params_id, sequence)))
        fixed_name_to_pid = {model_info.name[pid]: pid for pid in fixed_id}
        pname_pid_all = pname_to_pid | fixed_name_to_pid
        self._value_mapping_to_params: Callable[
            [ParamNameValMapping], ParamIDValMapping
        ] = jax.jit(
            lambda mapping: {
                v: mapping[k] for k, v in pname_pid_all.items() if k in mapping
            }
        )
        self._fn = fn
        self._additive_fn = additive_fn
        self._type = mtype
        self._model_info = model_info
        self._nparam = len(pname_to_pid)
        self.__initialized = True

    @property
    def params_name(self) -> tuple[str, ...]:
        """Parameter names."""
        return self._params_name

    @property
    def type(self) -> Literal['add', 'mul']:
        """Model type."""
        return self._type

    @property
    def has_comps(self) -> bool:
        """Whether the model has additive subcomponents."""
        return self._additive_fn is not None

    def _prepare_eval(self, params: ArrayLike | Sequence | Mapping | None):
        """Check if `params` is valid for the model."""
        if isinstance(params, (np.ndarray, jax.Array, Sequence)):
            params = jnp.atleast_1d(jnp.asarray(params, float))
            if params.shape[-1] != self._nparam:
                raise ValueError(
                    f'expected params shape (..., {self._nparam}), got '
                    f'{jnp.shape(params)}'
                )
            params = jnp.moveaxis(params, -1, 0)
            params = self._value_sequence_to_params(params)

        elif isinstance(params, Mapping):
            # if not set(self.params_name).issubset(params):
            #     missing = set(self.params_name) - set(params)
            #     raise ValueError(f'missing parameters: {", ".join(missing)}')

            params = jax.tree_map(jnp.asarray, params)
            params = self._value_mapping_to_params(params)

        elif params is None:
            params = self._params_default

        else:
            raise TypeError('params must be a array, sequence or mapping')

        fn = self._fn
        add_fn = self._additive_fn
        shapes = jax.tree_util.tree_flatten(
            tree=jax.tree_map(jnp.shape, params),
            is_leaf=lambda i: isinstance(i, tuple),
        )[0]

        if self.params_name:
            if not shapes:
                raise ValueError('params are empty')

            shape = shapes[0]
            if any(s != shape for s in shapes[1:]):
                raise ValueError('all params must have the same shape')

            # iteratively vmap and jit over params dimensions
            # use the nested-jit trick to reduce the compilation time
            for _ in range(len(shape)):
                fn = jax.jit(jax.vmap(fn, in_axes=(None, 0)))

            if add_fn is not None:
                for _ in range(len(shape)):
                    add_fn = jax.jit(jax.vmap(add_fn, in_axes=(None, 0)))

        return fn, add_fn, params

    def eval(
        self,
        egrid: ArrayLike,
        params: Sequence | Mapping | None = None,
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
        jax.Array
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
        jax.Array, or dict[str, jax.Array]
            The differential photon flux in units of ph cm⁻² s⁻¹ keV⁻¹.
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
        jax.Array, or dict[str, jax.Array]
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
        factor = keV_to_erg * emid

        ne = self.ne(egrid, params, comps)

        if comps:
            ene = jax.tree_map(lambda v: factor * v, ne)
        else:
            ene = factor * ne

        return ene

    def eene(
        self,
        egrid: ArrayLike,
        params: Sequence | Mapping | None = None,
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
        jax.Array, or dict[str, jax.Array]
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
        factor = keV_to_erg * e2

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
        channel_width: ArrayLike,
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
        channel_width : ndarray
            Measured energy channel width of `resp_matrix`.
        params : sequence or mapping, optional
            Parameter sequence or mapping for the model.
        comps : bool, optional
            Whether to return the result of each component. Defaults to False.

        Returns
        -------
        jax.Array, or dict[str, jax.Array]
            The folded model in units of count s⁻¹ keV⁻¹.
        """
        if self.type != 'add':
            msg = f'C(E) is undefined for {self.type} type model "{self}"'
            raise TypeError(msg)

        if self._additive_fn is None and comps:
            raise RuntimeError(f'{self} has no sub-models with additive type')

        egrid = jnp.asarray(egrid, float)
        resp_matrix = jnp.asarray(resp_matrix, float)
        channel_width = jnp.asarray(channel_width, float)

        ne = self.ne(egrid, params, comps)
        de = egrid[1:] - egrid[:-1]
        fn = jax.jit(lambda v: (v * de) @ resp_matrix / channel_width)

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

        .. warning::
            The flux is calculated by trapezoidal rule, which may not be
            accurate if not enough energy bins are used when the difference
            between `emin` and `emax` is large.

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
            otherwise calculate photon flux in units of ph cm⁻² s⁻¹.
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
        jax.Array, or dict[str, jax.Array]
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

    def lumin(
        self,
        emin_rest: float | int,
        emax_rest: float | int,
        z: float | int,
        params: Sequence | Mapping | None = None,
        comps: bool = False,
        ngrid: int = 1000,
        log: bool = True,
        cosmo: LambdaCDM = Planck18,
    ) -> JAXArray | dict[str, JAXArray]:
        """Calculate the luminosity of model.

        .. warning::
            The luminosity is calculated by trapezoidal rule, and is accurate
            only if enough numbers of energy grids are used.

        Parameters
        ----------
        emin_rest : float or int
            Minimum value of rest-frame energy range, in units of keV.
        emax_rest : float or int
            Maximum value of rest-frame energy range, in units of keV.
        z : float or int
            Redshift of the source.
        params : sequence or mapping, optional
            Parameter sequence or mapping for the model.
        comps : bool, optional
            Whether to return the result of each component. The default is
            False.
        ngrid : int, optional
            The energy grid number to use in integration. The default is 1000.
        log : bool, optional
            Whether to use logarithmically regular energy grid. The default is
            True.
        cosmo : LambdaCDM, optional
            Cosmology model used to calculate luminosity. The default is
            Planck18.

        Returns
        -------
        jax.Array, or dict[str, jax.Array]
            The model luminosity.
        """
        if self.type != 'add':
            msg = f'lumin is undefined for {self.type} type model "{self}"'
            raise TypeError(msg)

        z = float(z)
        flux = self.flux(
            emin=float(emin_rest) / (1.0 + z),
            emax=float(emax_rest) / (1.0 + z),
            params=params,
            energy=True,
            comps=bool(comps),
            ngrid=int(ngrid),
            log=bool(log),
        )
        flux_unit = u.Unit('erg cm^-2 s^-1')

        factor = 4.0 * np.pi * cosmo.luminosity_distance(z) ** 2
        to_lumin = lambda x: (x * flux_unit * factor).to('erg s^-1')

        if comps:
            return jax.tree_map(to_lumin, flux)
        else:
            return to_lumin(flux)

    def eiso(
        self,
        emin_rest: float | int,
        emax_rest: float | int,
        z: float | int,
        duration: float | int,
        params: Sequence | Mapping | None = None,
        comps: bool = False,
        ngrid: int = 1000,
        log: bool = True,
        cosmo: LambdaCDM = Planck18,
    ) -> JAXArray | dict[str, JAXArray]:
        r"""Calculate the isotropic emission energy of model.

        .. warning::
            The :math:`E_\mathrm{iso}` is calculated by trapezoidal rule,
            and is accurate only if enough numbers of energy grids are used.

        Parameters
        ----------
        emin_rest : float or int
            Minimum value of rest-frame energy range, in units of keV.
        emax_rest : float or int
            Maximum value of rest-frame energy range, in units of keV.
        z : float or int
            Redshift of the source.
        duration : float or int
            Observed duration of the source, in units of second.
        comps : bool, optional
            Whether to return the result of each component. The default is
            False.
        ngrid : int, optional
            The energy grid number to use in integration. The default is
            1000.
        log : bool, optional
            Whether to use logarithmically regular energy grid. The default
            is True.
        params : dict, optional
            Parameters dict to overwrite the fitted parameters.
        cosmo : LambdaCDM, optional
            Cosmology model used to calculate luminosity. The default is
            Planck18.

        Returns
        -------
        jax.Array, or dict[str, jax.Array]
            The isotropic emission energy of the model.
        """
        if self.type != 'add':
            msg = f'eiso is undefined for {self.type} type model "{self}"'
            raise TypeError(msg)

        lumin = self.lumin(
            emin_rest, emax_rest, z, params, comps, ngrid, log, cosmo
        )

        # This includes correction for time dilation.
        factor = duration / (1 + z) * u.s
        to_eiso = lambda x: (x * factor).to('erg')

        if comps:
            return jax.tree_map(to_eiso, lumin)
        else:
            return to_eiso(lumin)

    def simulate_from_data(
        self,
        data: ObservationData,
        spec_exposure: float | None = None,
        back_exposure: float | None = None,
        params: Sequence | Mapping | None = None,
        seed: int = 42,
        **kwargs: dict,
    ) -> ObservationData:
        """Simulate observation based on the configuration of existing data.

        Parameters
        ----------
        data : ObservationData
            Observation data to read observation configuration.
        spec_exposure : float, optional
            Exposure time of the source. Defaults to the exposure time of
            `spec_data`.
        back_exposure : float, optional
            Exposure time of the background. Defaults to the exposure time of
            `back_data`.
        params : sequence or mapping, optional
            Parameter sequence or mapping for the model.
        seed : int, optional
            Random seed for the simulation. The default is 42.
        **kwargs : dict, optional
            Additional keyword arguments passed to :class:`ObservationData`.

        Returns
        -------
        ObservationData
            Simulated observation data.
        """
        if self.type != 'add':
            msg = f'cannot simulate data from {self.type} type model "{self}"'
            raise TypeError(msg)

        if not isinstance(data, ObservationData):
            raise TypeError('data must be an ObservationData instance')

        if spec_exposure is None:
            spec_exposure = data.spec_exposure
        else:
            spec_exposure = float(spec_exposure)

        if data.has_back:
            back_counts = data.back_data.counts
            back_errors = data.back_data.errors
            back_poisson = data.back_poisson
            back_area_scale = data.back_data.area_scale
            back_back_scale = data.back_data.back_scale
            if back_exposure is None:
                back_exposure = data.back_exposure
            else:
                back_exposure = float(back_exposure)
        else:
            back_counts = back_errors = back_exposure = back_poisson = None
            back_area_scale = back_back_scale = 1.0

        spec_data = data.spec_data
        resp_data = data.resp_data

        simulation = self.simulate(
            photon_egrid=resp_data.photon_egrid,
            channel_emin=resp_data.channel_emin,
            channel_emax=resp_data.channel_emax,
            response_matrix=resp_data.sparse_matrix,
            spec_exposure=spec_exposure,
            spec_poisson=spec_data.poisson,
            spec_errors=spec_data.errors,
            back_counts=back_counts,
            back_errors=back_errors,
            back_exposure=back_exposure,
            back_poisson=back_poisson,
            spec_area_scale=spec_data.area_scale,
            spec_back_scale=spec_data.back_scale,
            back_area_scale=back_area_scale,
            back_back_scale=back_back_scale,
            quality=np.where(data.good_quality, 0, 1),
            grouping=data.grouping,
            channel=resp_data.channel,
            channel_type=resp_data.channel_type,
            response_sparse=resp_data.sparse,
            params=params,
            name=data.name,
            seed=seed,
            **kwargs,
        )
        simulation.set_erange(data.erange)

        return simulation

    def simulate(
        self,
        photon_egrid: NDArray,
        channel_emin: NDArray,
        channel_emax: NDArray,
        response_matrix: NDArray | sparray,
        spec_exposure: float,
        spec_poisson: bool = True,
        spec_errors: NDArray | None = None,
        back_counts: NDArray | None = None,
        back_errors: NDArray | None = None,
        back_exposure: float | None = None,
        back_poisson: bool | None = None,
        spec_area_scale: float | NDArray = 1.0,
        spec_back_scale: float | NDArray = 1.0,
        back_area_scale: float | NDArray = 1.0,
        back_back_scale: float | NDArray = 1.0,
        quality: NDArray | None = None,
        grouping: NDArray | None = None,
        channel: NDArray | None = None,
        channel_type: str = 'Ch',
        response_sparse: bool = False,
        params: Sequence | Mapping | None = None,
        name: str = 'simulation',
        seed: int = 42,
        **kwargs: dict,
    ) -> ObservationData:
        """Simulate observation data.

        Parameters
        ----------
        photon_egrid : ndarray
            Photon energy grid in units of keV.
        channel_emin : ndarray
            Lower energy bounds of the detector channels.
        channel_emax : ndarray
            Upper energy bounds of the detector channels.
        response_matrix : ndarray
            Response matrix of the detector.
        spec_exposure : float
            Exposure time of the source.
        spec_poisson : bool, optional
            Whether the source spectrum is Poisson distributed. If false,
            `spec_errors` must be provided. The default is True.
        spec_errors : ndarray, optional
            Errors of the source spectrum. Must be provided if `spec_poisson`
            is False.
        back_counts : ndarray, optional
            Background counts in each channel.
        back_errors : ndarray, optional
            Errors of the background counts. Must be provided if `back_poisson`
            is False.
        back_exposure : float, optional
            Exposure time of the background.
        back_poisson : bool, optional
            Whether the background spectrum is Poisson distributed. If false,
            `back_errors` must be provided.
        spec_area_scale : float or ndarray, optional
            Area scale factor of the source. The default is 1.0.
        spec_back_scale : float or ndarray, optional
            Background scale factor of the source. The default is 1.0.
        back_area_scale : float or ndarray, optional
            Area scale factor of the background. The default is 1.0.
        back_back_scale : float or ndarray, optional
            Background scale factor of the background. The default is 1.0.
        quality : ndarray, optional
            Quality flags of the data.
        grouping : ndarray, optional
            Grouping flags of the data.
        channel : ndarray, optional
            Channel numbers.
        channel_type : str, optional
            Channel type. The default is 'Ch'.
        response_sparse : bool, optional
            Whether the response matrix is sparse. The default is False.
        params : sequence or mapping, optional
            Parameter sequence or mapping for the model.
        name : str, optional
            Name of the simulation data. The default is 'simulation'.
        seed : int, optional
            Random seed for simulation. The default is 42.
        **kwargs : dict, optional
            Additional keyword arguments passed to :class:`ObservationData`.

        Returns
        -------
        ObservationData
            Simulated observation data.
        """
        if self.type != 'add':
            msg = f'cannot simulate data from {self.type} type model "{self}"'
            raise TypeError(msg)

        if channel is None:
            channel = np.arange(len(channel_emin)).astype(str)

        rng = np.random.default_rng(int(seed))

        resp_data = ResponseData(
            photon_egrid=photon_egrid,
            channel_emin=channel_emin,
            channel_emax=channel_emax,
            response_matrix=response_matrix,
            channel=channel,
            channel_type=channel_type,
            sparse=response_sparse,
        )

        if not spec_poisson:
            if spec_errors is None:
                raise ValueError(
                    'spec_errors must be provided if spec_poisson is False'
                )
            else:
                spec_errors = np.asarray(spec_errors, float)
                if spec_errors.size != resp_data.channel_number:
                    raise ValueError(
                        f'spec_errors size ({np.size(spec_errors)}) must be '
                        f'channel number ({resp_data.channel_number})'
                    )

        if not (
            back_counts is None
            and back_exposure is None
            and back_poisson is None
            and back_errors is None
        ):
            if back_counts is None:
                raise ValueError('back_counts must be also provided')
            if back_exposure is None:
                raise ValueError('back_exposure must be also provided')
            if back_poisson is None:
                raise ValueError('back_poisson must be also provided')

            back_counts = np.asarray(back_counts, np.float64)
            if back_counts.size != resp_data.channel_number:
                raise ValueError(
                    f'back_counts size ({np.size(back_counts)}) must be '
                    f'channel number ({resp_data.channel_number})'
                )

            if spec_poisson or back_poisson:
                if np.any(back_counts < 0):
                    warnings.warn(
                        'negative background counts is set to 0 for '
                        'Poisson counts simulation',
                        Warning,
                        stacklevel=2,
                    )
                    back_counts = np.clip(back_counts, 0, None)

            if not back_poisson:
                if back_errors is None:
                    raise ValueError(
                        'back_errors must be provided if back_poisson is False'
                    )
                else:
                    back_errors = np.asarray(back_errors, float)
                    if back_errors.size != resp_data.channel_number:
                        raise ValueError(
                            f'back_errors size ({np.size(back_errors)}) must '
                            f'be channel number ({resp_data.channel_number})'
                        )
            has_back = True
        else:
            has_back = False

        if has_back:
            if back_poisson:
                back_sim = np.array(rng.poisson(back_counts), np.float64)
                back_errors = np.sqrt(back_sim)
            else:
                # TODO: should fractional systematic errors be added?
                back_sim = rng.normal(back_counts, back_errors)

            back_data = SpectrumData(
                counts=back_sim,
                errors=back_errors,
                poisson=back_poisson,
                exposure=back_exposure,
                quality=quality,
                grouping=grouping,
                area_scale=back_area_scale,
                back_scale=back_back_scale,
            )
        else:
            back_data = None

        matrix = resp_data.sparse_matrix.T
        folded_rate = matrix @ self.eval(resp_data.photon_egrid, params)
        spec_counts = folded_rate * spec_exposure * spec_area_scale

        if has_back:
            back_ratio = (
                spec_exposure * spec_area_scale * spec_back_scale
            ) / (back_exposure * back_area_scale * back_back_scale)
            spec_counts += back_ratio * back_counts

        if spec_poisson:
            spec_sim = np.array(rng.poisson(spec_counts), np.float64)
            spec_errors = np.sqrt(spec_sim)
        else:
            # TODO: should fractional systematic errors be added?
            spec_sim = rng.normal(spec_counts, spec_errors)

        spec_data = SpectrumData(
            counts=spec_sim,
            errors=spec_errors,
            poisson=spec_poisson,
            exposure=spec_exposure,
            quality=quality,
            grouping=grouping,
            area_scale=spec_area_scale,
            back_scale=spec_back_scale,
        )

        return ObservationData(
            name=name,
            erange=[resp_data.channel_emin[0], resp_data.channel_emax[-1]],
            spec_data=spec_data,
            resp_data=resp_data,
            back_data=back_data,
            **kwargs,
        )

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        fields = ['No.', 'Component', 'Parameter', 'Value', 'Bound', 'Prior']
        info = self._model_info.info
        tab_params = make_pretty_table(fields, info)
        return (
            f'Model: {self.name} [{self.type}]\n' f'{tab_params.get_string()}'
        )

    def _repr_html_(self) -> str:
        fields = ['No.', 'Component', 'Parameter', 'Value', 'Bound', 'Prior']
        info = self._model_info.info
        tab_params = make_pretty_table(fields, info)
        return (
            '<details open>'
            f'<summary><b>Model: {self.name} [{self.type}]</b></summary>'
            f'{tab_params.get_html_string(format=True)}'
            '</details>'
        )

    def __delattr__(self, key: str):
        if self.__initialized and hasattr(self, key):
            raise AttributeError("can't delete attribute")

        super().__delattr__(key)

    def __setattr__(self, key: str, value: Any):
        if self.__initialized:
            raise AttributeError("can't set attribute")

        super().__setattr__(key, value)


class UniComponentModel(Model):
    """Model defined by a single additive or multiplicative component."""

    _component: Component

    def __init__(self, component: Component):
        self._component = component
        super().__init__(component._id, component._id, [component])

    @property
    def eval(self) -> ModelEval:
        comp_id = self._component._id
        _fn = self._component.eval

        def fn(egrid: JAXArray, params: CompIDParamValMapping) -> JAXArray:
            """The model evaluation function"""
            return _fn(egrid, params[comp_id])

        return jax.jit(fn)

    @property
    def type(self) -> Literal['add', 'mul']:
        return self._component.type


class CompositeModel(Model):
    """Model defined by sum or product of two models."""

    _operands: tuple[Model, Model]
    _op: Callable[[JAXArray, JAXArray], JAXArray]
    _op_symbol: Literal['+', '*']
    _type: Literal['add', 'mul']
    __additive_comps: tuple[Model, ...] | None = None

    def __init__(self, lhs: Model, rhs: Model, op: Literal['+', '*']):
        # check if the type of lhs and rhs are both model
        if not (isinstance(lhs, Model) and isinstance(rhs, Model)):
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

        return jax.jit(fn)

    @property
    def _additive_comps(self) -> tuple[Model, ...]:
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
                f"{i[0]}: 'Parameter' | float | None = None" for i in config
            ]
            sig1 += ['latex: str | None = None']

            params = '{' + ', '.join(f"'{i[0]}': {i[0]}" for i in config) + '}'

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
            exec(func_code, tmp := {})
            cls.__init__ = tmp['__init__']

        return cls

    def __init__(cls, name, bases, dct, **kwargs) -> None:
        super().__init__(name, bases, dct, **kwargs)

        # restore the signature of __init__ method
        # see https://stackoverflow.com/a/65385559
        signature = inspect.signature(cls.__init__)
        cls.__signature__ = signature.replace(
            parameters=tuple(signature.parameters.values())[1:],
            return_annotation=UniComponentModel,
        )

    def __call__(cls, *args, **kwargs) -> UniComponentModel | ConvolutionModel:
        # return Model/ConvolutionModel after Component initialization
        component = super().__call__(*args, **kwargs)

        if component.type != 'conv':
            return UniComponentModel(component)
        else:
            return ConvolutionModel(component)


def _param_getter(name: str) -> Callable[[Component], Parameter]:
    def getter(self: Component) -> Parameter:
        """Get parameter."""
        return getattr(self, f'__{name}')

    return getter


def _param_setter(name: str, idx: int) -> Callable[[Component, Any], None]:
    def setter(self: Component, param: Any) -> None:
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
            param, (float, int, jnp.number, jnp.ndarray, np.ndarray)
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

        elif isinstance(param, Parameter):
            # given parameter instance
            param = param

        elif isinstance(param, Mapping):
            # given mapping

            if not {'default', 'min', 'max'}.issubset(param.keys()):
                raise ValueError(
                    f'{type(self).__name__}.{cfg.name} expected dict with keys'
                    f' "default", "min", "max", and optional "log", but got '
                    f'{param}'
                )

            param = UniformParameter(
                name=cfg.name,
                default=param['default'],
                min=param['min'],
                max=param['max'],
                log=param.get('log', False),
                fixed=param.get('fixed', False),
                latex=param.get('latex', cfg.latex),
            )

        elif isinstance(param, Sequence):
            # given sequence
            if len(param) not in {1, 3, 4}:
                raise ValueError(
                    f'{type(self).__name__}.{cfg.name} expected sequence of '
                    'length 1, 3, or 4: '
                    '[default (float)], '
                    '[default (float), min (float), max (float)], or '
                    '[default (float), min (float), max (float), log (bool)], '
                    f'but got {param}'
                )

            if len(param) == 1:
                (default,) = param
                min_ = cfg.min
                max_ = cfg.max
                log = cfg.log
            elif len(param) == 3:
                default, min_, max_ = param
                log = cfg.log
            else:
                default, min_, max_, log = param

            param = UniformParameter(
                name=cfg.name,
                default=default,
                min=min_,
                max=max_,
                log=log,
                fixed=cfg.fixed,
                latex=cfg.latex,
            )

        else:
            # other input types are not supported yet
            raise TypeError(
                f'{type(self).__name__}.{cfg.name} got unsupported value '
                f'{param} ({type(param).__name__})'
            )

        prev_param: Parameter | None = getattr(self, f'__{name}', None)
        if prev_param is not None:
            prev_param._tracker.remove(self._id, name)

        setattr(self, f'__{name}', param)

        param._tracker.append(self._id, name)

    return setter


class ParamConfig(NamedTuple):
    """Configuration of a uniform parameter for spectral component."""

    name: str
    """Plain name of the parameter."""

    latex: str
    r""":math:`\LaTeX` of the parameter."""

    unit: str
    """Physical unit."""

    default: float
    """Default value of the parameter."""

    min: float
    """Minimum value of the parameter."""

    max: float
    """Maximum value of the parameter."""

    log: bool = False
    """Whether the parameter is parameterized in a logarithmic scale."""

    fixed: bool = False
    """Whether the parameter is fixed."""


class Component(ABC, metaclass=ComponentMeta):
    """Base class to define a spectral component."""

    _config: tuple[ParamConfig, ...]
    _id: CompID
    _args: tuple[str, ...] = ()  # extra args passed to subclass __init__
    _kwargs: tuple[str, ...] = ()  # extra kwargs passed to subclass __init__
    _staticmethod: tuple[str, ...] = ()  # methods need to be static
    __initialized: bool = False

    def __init__(self, params: dict, latex: str | None):
        self._id = hex(id(self))[2:]

        if latex is None:
            latex = rf'\mathrm{{{self.__class__.__name__}}}'
        self.latex = latex

        self._name = self.__class__.__name__

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

    def __getitem__(self, key: str) -> Parameter:
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


class AdditiveComponent(Component):
    """Prototype class to define an additive component."""

    @property
    def type(self) -> Literal['add']:
        """Component type is additive."""
        return 'add'


class MultiplicativeComponent(Component):
    """Prototype class to define a multiplicative component."""

    @property
    def type(self) -> Literal['mul']:
        """Component type is multiplicative."""
        return 'mul'


class AnalyticalIntegral(Component):
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


class NumericalIntegral(Component):
    """Prototype component to calculate model integral numerically."""

    _kwargs = ('method',)
    _continuum_jit: CompEval | None = None
    _staticmethod = ('continuum',)

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
        if self._continuum_jit is None:
            # _continuum is assumed to be a pure function, independent of self
            self._continuum_jit = jax.jit(self.continuum)

        return self._make_integral(self._continuum_jit)

    def _make_integral(self, continuum: CompEval):
        mtype = self.type

        if self.method == 'trapz':

            def fn(egrid: JAXArray, params: NameValMapping) -> JAXArray:
                """Numerical integration using trapezoidal rule."""
                if mtype == 'add':
                    factor = 0.5 * (egrid[1:] - egrid[:-1])
                else:
                    factor = 0.5
                f_grid = continuum(egrid, params)
                return factor * (f_grid[:-1] + f_grid[1:])

        elif self.method == 'simpson':

            def fn(egrid: JAXArray, params: NameValMapping) -> JAXArray:
                """Numerical integration using Simpson's 1/3 rule."""
                if mtype == 'add':
                    factor = (egrid[1:] - egrid[:-1]) / 6.0
                else:
                    factor = 1.0 / 6.0
                e_mid = 0.5 * (egrid[:-1] + egrid[1:])
                f_grid = continuum(egrid, params)
                f_mid = continuum(e_mid, params)
                return factor * (f_grid[:-1] + 4.0 * f_mid + f_grid[1:])

        else:
            raise NotImplementedError(f"integration method '{self.method}'")

        return jax.jit(fn)

    @staticmethod
    @abstractmethod
    def continuum(*args, **kwargs) -> JAXArray:
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
        jax.Array
            The photon flux integrated over `egrid`, in units of ph cm⁻² s⁻¹.
        """
        pass


class NumIntAdditive(NumericalIntegral, AdditiveComponent):
    """Prototype additive component with continuum expression defined."""

    @staticmethod
    @abstractmethod
    def continuum(*args, **kwargs) -> JAXArray:
        """Calculate the photon flux at the energy grid.

        Parameters
        ----------
        egrid : ndarray
            Photon energy grid in units of keV.
        params : dict
            Parameter dict for the model.

        Returns
        -------
        jax.Array
            The photon flux at `egrid`, in units of ph cm⁻² s⁻¹ keV⁻¹.
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
        jax.Array
            The model value, dimensionless.
        """
        pass


class NumIntMultiplicative(NumericalIntegral, MultiplicativeComponent):
    """Prototype multiplicative component with continuum expression defined."""

    @staticmethod
    @abstractmethod
    def continuum(*args, **kwargs) -> JAXArray:
        """Calculate the model value at the energy grid.

        Parameters
        ----------
        egrid : ndarray
            Photon energy grid in units of keV.
        params : dict
            Parameter dict for the model.

        Returns
        -------
        jax.Array
            The model value at `egrid`, dimensionless.
        """
        pass


class ConvolutionModel(Model):
    """Model defined by a convolution component."""

    def __init__(self, component: ConvolutionComponent):
        self._component = component
        super().__init__(component._id, component._id, [component])

    def __call__(self, model: Model) -> ConvolvedModel:
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


class ConvolvedModel(Model):
    """Model created by convolution."""

    _op: ConvolutionComponent
    _model: Model

    def __init__(self, op: ConvolutionComponent, model: Model):
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
    def _additive_comps(self) -> tuple[Model, ...]:
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

        return jax.jit(fn)

    @property
    def type(self) -> Literal['add', 'mul']:
        return self._model.type


class ConvolutionComponent(Component):
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
        jax.Array
            The convolved model value over `egrid`.
        """
        pass


class PyComponent(Component):
    """Prototype component with pure Python expression defined."""

    _kwargs: tuple[str, ...] = ('grad_method',)

    def __init__(
        self,
        params: dict,
        latex: str | None,
        grad_method: Literal['central', 'forward'] | None,
    ):
        self.grad_method = grad_method

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


class PyAnaInt(PyComponent, AnalyticalIntegral):
    """Prototype component with python integral expression defined."""

    @property
    def eval(self) -> CompEval:
        if self._integral_jit is None:
            integral_fn = self.integral

            def eval_integral(egrid, params):
                egrid = np.asarray(egrid)
                params = {k: np.asarray(v) for k, v in params.items()}
                return integral_fn(egrid, params)

            def integral(egrid: JAXArray, params: NameValMapping) -> JAXArray:
                shape_dtype = jax.ShapeDtypeStruct(
                    (egrid.size - 1,), egrid.dtype
                )
                return jax.pure_callback(
                    eval_integral, shape_dtype, egrid, params
                )

            self._integral_jit = jax.jit(
                define_fdjvp(jax.jit(integral), self.grad_method)
            )

        return self._integral_jit


class PyNumInt(PyComponent, NumericalIntegral):
    """Prototype component with python continuum expression defined."""

    _kwargs = ('method', 'grad_method')

    def __init__(
        self,
        params: dict,
        latex: str | None,
        method: Literal['trapz', 'simpson'] | None,
        grad_method: Literal['central', 'forward'] | None,
    ):
        super().__init__(params, latex, grad_method)
        self.method = 'trapz' if method is None else method

    @property
    def eval(self) -> CompEval:
        if self._continuum_jit is None:
            # continuum is assumed to be a pure function, independent of self
            continuum_fn = self.continuum

            def eval_continuum(egrid, params):
                egrid = np.asarray(egrid)
                params = {k: np.asarray(v) for k, v in params.items()}
                return continuum_fn(egrid, params)

            def continuum(egrid: JAXArray, params: NameValMapping) -> JAXArray:
                shape_dtype = jax.ShapeDtypeStruct(egrid.shape, egrid.dtype)
                return jax.pure_callback(
                    eval_continuum, shape_dtype, egrid, params
                )

            self._continuum_jit = jax.jit(
                define_fdjvp(jax.jit(continuum), self.grad_method)
            )

        return self._make_integral(self._continuum_jit)


def get_model_info(
    comps: Sequence[Component],
    cid_to_name: CompIDStrMapping,
    cid_to_latex: CompIDStrMapping,
) -> ModelInfo:
    """Get the model information.

    Parameters
    ----------
    comps : sequence of Component
        The sequence of components.
    cid_to_name : mapping
        The mapping of component id to component name.
    cid_to_latex : mapping
        The mapping of component id to component LaTeX format.

    Returns
    -------
    ModelInfo
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
        if (tmp := info.tracker.get_comp_param(cid_to_name.keys()))
    }

    # names of non-composite params with no component assigned (aux params)
    aux_params = {
        pid: info
        for pid, info in params_info.items()
        if (pid not in comp_param) and (not info.composite)
    }

    aux_names = [i.name for i in aux_params.values()]
    aux_names = build_namespace(aux_names, prime=True)['namespace']

    # name mapping from aux params id to name
    name_mapping: ParamIDStrMapping = dict(zip(aux_params.keys(), aux_names))

    # record name mapping of params directly assigned to a component
    name_mapping |= {
        pid: f'{cid_to_name[cid]}.{pname}'
        for pid, (cid, pname) in comp_param.items()
    }

    # record the component LaTeX format of parameters
    comp_latex_mapping = {
        pid: cid_to_latex[cid] for pid, (cid, _) in comp_param.items()
    }
    comp_latex_mapping |= {pid: '' for pid in aux_params}

    # record the LaTeX format of aux parameters
    aux_latex = [params_info[pid].latex for pid in aux_params]
    aux_latex = build_namespace(aux_latex, latex=True, prime=True)['namespace']
    latex_mapping = dict(zip(aux_params.keys(), aux_latex))

    # record the LaTeX format of component parameters from _config
    latex_mapping |= {
        comp[name]._id: comp._config[i].latex
        for comp in comps
        for (i, name) in enumerate(comp.param_names)
    }

    # TODO: record aux params unit & unit consistency check
    # record the unit of component parameters from _config
    unit_mapping = {
        comp[name]._id: comp._config[i].unit
        for comp in comps
        for (i, name) in enumerate(comp.param_names)
    }
    unit_mapping |= {pid: '' for pid, info in aux_params.items()}

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
    def factory1(
        id_to_value: Callable[[ParamIDValMapping], JAXFloat],
    ) -> Callable[[ParamIDValMapping], JAXFloat]:
        """Fill in the default parameter values for fn."""

        def fn(value_mapping: ParamIDValMapping) -> JAXFloat:
            """id_to_value with default values."""
            return id_to_value(fixed_values | value_mapping)

        return fn

    def factory2(
        pname_to_pid: dict[CompParamName, ParamID],
    ) -> Callable[[ParamIDValMapping], NameValMapping]:
        """Generate component parameter value getter."""

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
    cid_to_params = {
        comp._id: factory2({name: comp[name]._id for name in comp.param_names})
        for comp in comps
    }
    # ========== generate component parameter value getter function ===========

    # record non-interval params which has a component assigned
    deterministic = {}

    # record the setup of component parameters
    setup: dict[CompParamName, (ParamID, ParamSetup)] = {}

    # model_info is list of (No., Component, Parameter, Value, Bound, Prior)
    model_info = []
    sample_order = []
    n = 0

    for comp in comps:  # iterate through components
        cid = comp._id
        cname = cid_to_name[cid]
        for pname in comp.param_names:  # iterate through parameters of comp
            model_pname = f'{cname}.{pname}'
            pid = comp[pname]._id
            info = params_info[pid]

            idx = ''
            bound = info.bound
            prior = info.prior

            if (cid, pname) != comp_param[pid]:  # param is linked to another
                idx = ''
                value = name_mapping[pid]
                setup[model_pname] = (value, ParamSetup.Forwarded)

            elif info.composite:  # param is composite
                mapping = dict(name_mapping)
                del mapping[pid]
                value = info.name(mapping)

                if info.integrate:  # param is composite interval
                    setup[model_pname] = (model_pname, ParamSetup.Integrated)
                elif info.fixed:  # param is composite, but fixed
                    setup[model_pname] = (model_pname, ParamSetup.Fixed)
                else:  # param is composite but free to vary, add it to determ
                    idx = '*'
                    setup[model_pname] = (model_pname, ParamSetup.Composite)
                    deterministic[pid] = pid_to_value[pid]

            elif info.integrate:  # param is interval
                value = f'[{info.default[0]:.4g}, {info.default[1]:.4g}]'
                setup[model_pname] = (model_pname, ParamSetup.Integrated)

            else:  # a single param
                value = f'{info.default:.4g}'

                if info.fixed:
                    setup[model_pname] = (model_pname, ParamSetup.Fixed)
                else:
                    setup[model_pname] = (model_pname, ParamSetup.Free)
                    n += 1
                    idx = str(n)
                    sample_order.append(pid)

            model_info.append((idx, cname, pname, value, bound, prior))

    for pid, info in aux_params.items():  # iterate through aux params
        idx = ''
        bound = info.bound
        prior = info.prior

        if info.integrate:  # aux param is interval
            value = f'[{info.default[0]:.4g}, {info.default[1]:.4g}]'
        else:  # aux param is not interval, record its default value
            value = f'{info.default:.4g}'

        if not info.fixed:  # aux param is free to vary
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
        unit=unit_mapping,
        sample=sample,
        default=default,
        deterministic=deterministic,
        fixed=fixed,
        log=log,
        pid_to_comp_latex=comp_latex_mapping,
        cid_to_params=cid_to_params,
        cid_to_name=cid_to_name,
        cid_to_latex=cid_to_latex,
        integrate=integrate,
        setup=setup,
    )


class ModelInfo(NamedTuple):
    """Model information."""

    info: tuple[tuple[str, str, str, str, str, str], ...]
    """The model parameter information.

    Each row contains (No., Component, Parameter, Value, Bound, Prior).
    """

    name: ParamIDStrMapping
    """The mapping of parameter id to parameter name."""

    latex: ParamIDStrMapping
    r"""The mapping of component parameters name to :math:`\LaTeX` format."""

    unit: ParamIDStrMapping
    """The mapping of component parameters name to physical unit."""

    sample: dict[ParamID, Distribution]
    """The mapping of free parameter id to numpyro Distribution."""

    default: ParamIDValMapping
    """The mapping of free parameter id to default value."""

    deterministic: dict[ParamID, Callable[[ParamIDValMapping], JAXFloat]]
    """The mapping of deterministic parameters id to value getter."""

    fixed: ParamIDValMapping
    """The mapping of fixed parameter id to fixed parameter value."""

    log: dict[ParamID, bool]
    """The mapping of parameter id to logarithmic flag."""

    pid_to_comp_latex: dict[ParamID, str]
    """The mapping of parameter id to component LaTeX format."""

    cid_to_name: dict[CompID, CompName]
    """The mapping of component id to component name."""

    cid_to_latex: dict[CompID, str]
    """The mapping of component id to component LaTeX format."""

    cid_to_params: dict[CompID, Callable[[ParamIDValMapping], NameValMapping]]
    """The mapping of component id to parameter value getter function."""

    integrate: dict[ParamID, IntegralFactory]
    """The mapping of interval parameter id to integral operator."""

    setup: dict[CompParamName, tuple[ParamID, ParamSetup]]
    """The mapping of component parameter setup."""


class ParamSetup(Enum):
    """Model parameter setup."""

    Free = 0
    """Parameter is free to vary."""

    Composite = 1
    """Parameter is composed of other free parameters."""

    Forwarded = 2
    """Parameter is directly forwarded to another model parameter."""

    Fixed = 3
    """Parameter is fixed to a value."""

    Integrated = 4
    """Parameter is integrated out."""
