"""Handle model fitting."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from functools import partial, reduce
from typing import Any, Callable, Iterable, Literal, NamedTuple, Optional

import arviz as az
import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import numpyro
from iminuit import Minuit
from iminuit.util import FMin
# from numpyro.contrib.nested_sampling import NestedSampler
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import log_likelihood, constrain_fn, unconstrain_fn
from prettytable import PrettyTable
from scipy.stats import norm

from ..data.ogip import Data
from ..model.base import Model
from .likelihood import chi2, cstat, pstat, pgstat, wstat
from .nested_sampling import NestedSampler

__all__ = ['LikelihoodFit', 'BayesianFit']

Statistic = Literal['chi2', 'cstat', 'pstat', 'pgstat', 'wstat']


class BaseFit(ABC):
    """Model Fitting context.

    Parameters
    ----------
    data : Data or sequence of Data
        The observation data.
    model : Model or sequence of Model
        The model used  to fit the data.
    stat : str or sequence of str
        The likelihood option for the data and model. Available likelihood
        options are:
            * 'chi2' : Gaussian data
            * 'cstat' : Poisson data
            * 'pstat' : Poisson data with known background
            * 'pgstat' : Poisson data with Gaussian background
            * 'wstat' : Poisson data with Poisson background
    seed : int

    """

    _stat_options: set[str] = {
        'chi2',
        'cstat',
        'pstat',
        'pgstat',
        'wstat'
        # It should be noted that 'lstat' does not have long run coverage
        # property for source estimation, which is probably due to conjugate
        # prior choice of background. We may include 'lstat' at some point.
    }

    def __init__(
        self,
        data: Data | Iterable[Data],
        model: Model | Iterable[Model],
        stat: Statistic | Iterable[Statistic],
        seed: int = 42
    ):
        data, model, stat = self._sanity_check(data, model, stat)

        self._data = {d.name: d for d in data}
        self._model = {d.name: m for d, m in zip(data, model)}
        self._stat = {d.name: s for d, s in zip(data, stat)}
        self._seed = int(seed)

        # model information
        info_list = [m._model_info for m in model]
        info = {
            i: reduce(lambda x, y: x | y, (j[i] for j in info_list))
            for i in [
                'sample', 'composite',
                'default', 'min', 'max', 'dist_expr',
                'params', 'mname', 'mfmt', 'mpfmt', 'pname', 'pfmt'
            ]
        }

        # add data name to components if they are not fit with all data
        nd = len(data)
        comps = [j for i in model for j in i._comps]
        comps = np.unique(comps).tolist()
        name_suffix = {}
        fmt_suffix = {}
        for comp in comps:
            data_having_this_comp = []
            for name in self._data:
                if comp in self._model[name]._comps:
                    data_having_this_comp.append(name)

            if len(data_having_this_comp) == nd:
                name_suffix[comp] = ''
                fmt_suffix[comp] = ''
            else:
                name_suffix[comp] = '_' + '_'.join(data_having_this_comp)
                joined_name = '+'.join(data_having_this_comp)
                fmt_suffix[comp] = r'^\mathrm{' + joined_name + '}'

        info['mname'] = {
            k: v + name_suffix[k]
            for k, v in info['mname'].items()
        }
        info['mfmt'] = {
            k: r'$\big[' + v + fmt_suffix[k] + r'\big]$'
            for k, v in info['mfmt'].items()
        }

        # parameters directly input to the model, which are of interest
        params = [
            j  # sample site and composition directly input to the model
            for i in info['params'].values()
            for j in i.values()
        ]
        mname = info['mname']
        mfmt = info['mfmt']
        params_name = {}
        params_fmt = {}
        for m, ps in info['params'].items():
            for p, p_id in ps.items():
                params_name[p_id] = f'{mname[m]}_{p}'
                params_fmt[p_id] = f'{mfmt[m]}_{p}'

        # name/fmt of aux parameters
        aux_params = [
            i  # sample site not directly input to the model
            for i in info['sample']
            if i not in params and not isinstance(i, float)
        ]
        _name = [info['pname'][i] for i in aux_params]
        _name_suffix = [
            '' if (n := _name[:i+1].count(name)) == 1
            else str(n)
            for i, name in enumerate(_name)
        ]
        aux_params_name = {
            i: j + k for i, j, k in zip(aux_params, _name, _name_suffix)
        }
        _fmt = [info['pfmt'][i] for i in aux_params]
        _fmt_suffix = [
            '' if (n := _fmt[:i+1].count(fmt)) == 1
            else '_{%s}' % n
            for i, fmt in enumerate(_fmt)
        ]
        aux_params_fmt = {
            i: j + k for i, j, k in zip(aux_params, _fmt, _fmt_suffix)
        }

        id_mapping = params_name | aux_params_name | info['mname']

        self._info = _replace_by_mapping(id_mapping, info)

        self._model_info = _replace_by_mapping(
            id_mapping,
            {k: v._node.name for k, v in self._model.items()}
        )

        sample_dic = self._info['sample']
        default_dic = self._info['default']
        min_dic = self._info['min']
        max_dic = self._info['max']
        dist_dic = self._info['dist_expr']
        # params name having id as suffix
        params_dic = _replace_by_mapping(info['mname'], info['params'])
        idx = 1
        params_info = {}
        for comp, param_dic in self._info['params'].items():
            for i, j in param_dic.items():
                if j in sample_dic:
                    if not isinstance(sample_dic[j], float):  # free parameter
                        min_ = min_dic[j]
                        max_ = max_dic[j]
                        pidx = str(idx)
                        idx += 1
                    else:  # fixed parameter
                        min_ = max_ = ''
                        pidx = ''

                    params_info[f'{comp}_{i}'] = [
                        pidx, comp, i, default_dic[j], min_, max_, dist_dic[j]
                    ]

                else:  # composite parameter
                    mapping = {
                        k: v
                        for k, v in id_mapping.items()
                        if v != f'{comp}_{i}'
                    }
                    expr = _replace_by_mapping(mapping, params_dic[comp][i])
                    params_info[f'{comp}_{i}'] = [
                        '', comp, i, expr, '', '', ''
                    ]

        for k, v in self._info['sample'].items():
            if k in params_info:
                continue

            if not isinstance(v, float):  # free parameter
                min_ = min_dic[k]
                max_ = max_dic[k]
                pidx = str(idx)
                idx += 1
            else:  # fixed parameter
                min_ = max_ = ''
                pidx = ''

            params_info[k] = [
                pidx, '', k, default_dic[k], min_, max_, dist_dic[k]
            ]

        self._params_info = params_info

        # parameters Tex format
        self._params_fmt = _replace_by_mapping(
            id_mapping,
            params_fmt | aux_params_fmt
        )

        # likelihood function
        self._stat_func = {
            d.name: _likelihood_func(d, s)
            for d, s in zip(data, stat)
        }

        # spectral model function
        self._model_func = {
            i.name: j.pop('func')
            for i, j in zip(data, info_list)
        }

        # parameters of spectral model function
        model_func_params = {
            name: {minfo['mname'][k]: v for k, v in minfo['params'].items()}
            for name, minfo in zip(self._data, info_list)
        }
        self._model_func_params = _replace_by_mapping(
            id_mapping,
            model_func_params
        )

        # numpyro sample site
        self._sample = {
            k: v for k, v in self._info['sample'].items()
            if not isinstance(v, float)
        }

        # fixed parameters
        self._fixed_params = {
            k: v for k, v in self._info['sample'].items()
            if isinstance(v, float)
        }

        # composite parameter
        self._composite = _ordered_composite(
            self._info['sample'], self._info['composite']
        )

        # ordered free parameter
        self._free_params = tuple(
            k for k, v in self._params_info.items() if v[0]
        )

        # ordered parameters of interest,
        # which are directly input to model and not fixed
        self._interested_params = tuple(
            k for k, v in self._params_info.items()
            if v[1] and k not in self._fixed_params
        )

        # ordered parameters
        self._params = tuple(self._params_info.keys())

        # function containing numpyro primitives
        self._numpyro_model = self._generate_numpyro_model()

        # channel number of data
        self._nchan = {d.name: len(d.channel) for d in data}
        self._nchan['total'] = sum(self._nchan.values())

        # number of free parameters
        self._nparam = len(self._sample.keys())

        # degree of freedom
        self._dof = self._nchan['total'] - self._nparam

        self.print_info()

    @abstractmethod
    def print_info(self) -> None:
        """Print information about the fit."""
        pass

    def _generate_numpyro_model(self) -> Callable:
        """Return a function that contains numpyro primitives."""
        sample = self._sample
        fixed_params = self._fixed_params
        composite = self._composite
        deterministic = []  # record composite parameter input to model
        for i in self._info['params'].values():
            for j in i.values():
                if j not in sample and j not in deterministic:
                    deterministic.append(j)
        model_func = self._model_func
        model_params = self._model_func_params
        data = self._data
        names = list(data.keys())
        stat_func = self._stat_func

        def numpyro_model():
            """The numpyro model."""
            params = {
                name: numpyro.sample(name, dist)
                for name, dist in sample.items()
            } | {
                k: numpyro.deterministic(k, jnp.array(v))
                for k, v in fixed_params.items()
            }
            for name, (arg_names, func) in composite.items():
                args = (params[arg_name] for arg_name in arg_names)
                if name in deterministic:
                    v = numpyro.deterministic(name, func(*args))
                else:
                    v = func(*args)
                params[name] = v

            for name in names:
                d = data[name]
                ph_egrid = d.ph_egrid
                resp_matrix = d.resp_matrix
                expo = d.spec_effexpo
                mfunc = model_func[name]
                params_i = model_params[name]
                params_i = jax.tree_map(lambda k: params[k], params_i)
                model_counts = mfunc(ph_egrid, params_i) @ resp_matrix * expo
                stat_func[name](model_counts)

        return numpyro_model

    def _sanity_check(
        self,
        data: Data | Iterable[Data],
        model: Model | Iterable[Model],
        stat: Statistic | Iterable[Statistic]
    ):
        """Check if data, model, and stat are correct and return lists."""
        def get_list(inputs, name, itype, tname):
            """Check the model/data/stat, and return a list."""
            if isinstance(inputs, itype):
                input_list = [inputs]
            elif isinstance(inputs, (list, tuple)):
                if not inputs:
                    raise ValueError(f'{name} list is empty')
                if not all(isinstance(i, itype) for i in inputs):
                    raise ValueError(f'all {name} must be {tname} instance')
                input_list = list(inputs)
            else:
                raise ValueError(f'got wrong type {type(inputs)} for {name}')
            return input_list

        data_list = get_list(data, 'data', Data, 'Data')
        model_list = get_list(model, 'model', Model, 'Model')
        stat_list = get_list(stat, 'stat', str, 'str')

        # check if model type is additive
        flag = list(i.type == 'add' for i in model_list)
        if not all(flag):
            err = (j for i, j in enumerate(model_list) if not flag[i])
            err = ', '.join(f"'{i}'" for i in err)
            msg = f'got models which are not additive type: {err}'
            raise ValueError(msg)

        # check stat option
        flag = list(i in self._stat_options for i in stat_list)
        if not all(flag):
            err = (j for i, j in enumerate(stat_list) if not flag[i])
            err = ', '.join(f"'{i}'" for i in err)
            supported = ', '.join(f"'{i}'" for i in self._stat_options)
            msg = f'got unexpected stat: {err}; supported are {supported}'
            raise ValueError(msg)

        nd = len(data_list)
        nm = len(model_list)
        ns = len(stat_list)

        # check model number
        if nm == 1:
            model_list *= nd
        elif nm != nd:
            msg = f'number of model ({nm}) and data ({nd}) are not matched'
            raise ValueError(msg)

        # check stat number
        if ns == 1:
            stat_list *= nd
        elif ns != nd:
            msg = f'number of data ({nd}) and stat ({ns}) are not matched'
            raise ValueError(msg)

        # check if correctly using stat
        def check_data_stat(d, s):
            """Check if data type and likelihood are matched."""
            name = d.name
            if s != 'chi2' and not d.spec_poisson:
                stat_h = s[:s.index('stat')].upper()
                msg = f'Poisson data is required for using {stat_h}-statistics'
                raise ValueError(msg)

            if s == 'cstat' and d.has_back:
                back = 'Poisson' if d.back_poisson else 'Gaussian'
                stat1 = 'W' if d.back_poisson else 'PG'
                stat2 = 'w' if d.back_poisson else 'pg'
                msg = 'C-statistics is not valid for Poisson data with '
                msg += f'{back} background, use {stat1}-statistics'
                msg += f'({stat2}stat) for {name} instead'
                raise ValueError(msg)

            elif s == 'pstat' and not d.has_back:
                msg = 'background is required for P-statistics'
                raise ValueError(msg)

            elif s == 'pgstat' and not d.has_back:
                msg = 'background is required for PG-statistics'
                raise ValueError(msg)

            elif s == 'wstat':
                if not d.spec_poisson:
                    msg = 'Poisson data is required for W-statistics'
                    raise ValueError(msg)
                if not (d.has_back and d.back_poisson):
                    msg = 'Poisson background is required for W-statistics'
                    raise ValueError(msg)

        for d, s in zip(data_list, stat_list):
            check_data_stat(d, s)

        return data_list, model_list, stat_list


class MLEResult(NamedTuple):
    """Result of :meth:`LikelihoodFit.mle`."""

    params: dict[str, tuple[float, float]]
    stat: dict[str, float]
    nchan: dict[str, int]
    dof: int
    aic: float
    bic: float
    state: FMin


class ConfidenceInterval(NamedTuple):
    """Result of :meth:`LikelihoodFit.ci`."""

    mle: dict[str, float]
    interval: dict[str, tuple[float, float]]
    error: dict[str, tuple[float, float]]
    cl: float
    status: dict


class BootstrapResult(NamedTuple):
    """Result of :meth:`LikelihoodFit.boot`."""

    ...


class LikelihoodFit(BaseFit):
    def __init__(
        self,
        data: Data | Iterable[Data],
        model: Model | Iterable[Model],
        stat: Statistic | Iterable[Statistic],
        seed: int = 42
    ):
        super().__init__(data, model, stat, seed)

        params = self._params
        free_params = self._free_params
        numpyro_model = self._numpyro_model

        def to_dict(params_array):
            """Transform parameter array to a dict."""
            return {k: v for k, v in zip(free_params, params_array)}

        def to_constr_dict(unconstrained_params_array):
            """Transform unconstrained parameters array to constrained dict."""
            return constrain_fn(
                model=numpyro_model,
                model_args=(),
                model_kwargs={},
                params=to_dict(unconstrained_params_array)
            )

        def to_unconstr_dict(constrained_params_array):
            """Transform constrained parameters array to unconstrained dict."""
            return unconstrain_fn(
                model=numpyro_model,
                model_args=(),
                model_kwargs={},
                params=to_dict(constrained_params_array)
            )

        def to_unconstr_array(constr_array):
            """Transform constrained parameters array to unconstrained."""
            unconstr = to_unconstr_dict(constr_array)
            return jnp.array([unconstr[i] for i in free_params])

        def deviance_unconstr(unconstrained_params_array):
            """Deviance function in unconstrained parameter space."""
            p = to_constr_dict(unconstrained_params_array)
            return -2.0 * jax.tree_util.tree_reduce(
                lambda x, y: x + y,
                jax.tree_map(
                    lambda x: x.sum(),
                    log_likelihood(numpyro_model, p)
                )
            )

        def deviance_unconstr_group(unconstrained_params_array):
            """Deviance by data group in unconstrained parameter space."""
            p = to_constr_dict(unconstrained_params_array)
            deviance = jax.tree_map(
                lambda x: -2.0 * x.sum(),
                log_likelihood(numpyro_model, p)
            )
            deviance = {
                name: sum(
                    v for k, v in deviance.items()
                    if k in [f'{name}_Non', f'{name}_Noff']
                )
                for name in self._data.keys()
            }
            return deviance

        def to_params_dict(unconstr_dict):
            """Transform unconstrained dict array to constrained parameters."""
            return constrain_fn(
                model=numpyro_model,
                model_args=(),
                model_kwargs={},
                params=unconstr_dict,
                return_deterministic=True
            )

        def to_params_array(unconstrained_params_array):
            """Transform unconstrained parameters array to constrained."""
            unconstr = to_dict(unconstrained_params_array)
            constr = to_params_dict(unconstr)  # including deterministic site
            return jnp.array([constr[i] for i in params])

        free = [self._info['default'][k] for k in free_params]
        init_unconstr = np.array(to_unconstr_array(free))

        self._free_params = free_params
        self._init_unconstr = init_unconstr
        self._deviance = deviance_unconstr
        self._deviance_group = jax.jit(deviance_unconstr_group)
        self._grad_unconstr = jax.jit(jax.grad(deviance_unconstr))
        self._cov_unconstr = jax.jit(
            lambda x: 2.0 * jnp.linalg.inv(jax.hessian(deviance_unconstr)(x))
        )
        self._to_unconstr = jax.jit(to_unconstr_array)
        self._to_params_dict = jax.jit(to_params_dict)
        self._to_params_array = jax.jit(to_params_array)
        self._to_params_array_jac = jax.jit(jax.jacobian(to_params_array))
        self._minuit: Minuit | None = None
        self._ns: NestedSampler | None = None

    def print_info(self) -> None:
        tab1 = _make_text_table(
            ['Data', 'Model'],
            list((k, v) for k, v in self._model_info.items())
        )
        tab2 = _make_text_table(
            ['No.', 'Component', 'Parameter', 'Value', 'Minimum', 'Maximum'],
            list(i[:-1] for i in self._params_info.values())
        )
        print('Likelihood Fit')
        print(tab1)
        print(tab2)

    def mle(
        self,
        init: Optional[dict[str, float]] = None,
        full_params: bool = False,
        global_search: Optional[str] = None
    ) -> MLEResult:
        """Find the Maximum Likelihood Estimation (MLE) for the model.

        Migrad optimization of Minuit is used here to perform the MLE, as it is
        considered as a robust optimiser.

        Parameters
        ----------
        init : dict, optional
            Initial guess for the MLE. The default is None.
        full_params : bool, optional
            Whether to include the parameters in the MLEResult. The default is
            False, which means only includes those parameters directly input
            to spectral model.
        global_search : {'ns'}, optional
            Global search method to find the initial guess for the MLE.
            Available options are:
                * 'ns' : nested sampling of :mod:`jaxns`.
            The default is None.

        Returns
        -------
        MLEResult
            The MLE result.

        """
        if global_search == 'ns' and self._ns is None:
            ns = NestedSampler(
                self._numpyro_model,
                constructor_kwargs=dict(
                    num_live_points=100 * self._nparam,
                    max_samples=100000,
                    num_parallel_samplers=jax.device_count(),
                    # num_parallel_workers=jax.device_count(),
                ),
                termination_kwargs=dict(
                    live_evidence_frac=1e-5,
                )
            )

            t0 = time.time()
            ns.run(jax.random.PRNGKey(42))
            print(f'Global fit cost {time.time() - t0:.2f} s')
            mle_idx = ns._results.log_L_samples.argmax()
            mle_constr = jax.tree_map(
                lambda s: s[mle_idx], ns._results.samples
            )
            mle_constr = [mle_constr[i] for i in self._free_params]
            init_unconstr = np.array(self._to_unconstr(mle_constr))
            self._ns = ns

        elif init is not None:
            init_constr = [init[i] for i in self._free_params]
            init_unconstr = self._to_unconstr(init_constr)

        else:
            init_unconstr = self._init_unconstr

        minuit = Minuit(
            jax.jit(self._deviance),
            init_unconstr,
            grad=self._grad_unconstr,
            name=self._free_params
        )

        # max_it = 10
        # nit = 0
        # minuit.strategy = 0
        # minuit.migrad()
        # while (not minuit.fmin.is_valid) and nit < max_it:
        #     minuit.hesse()
        #     minuit.migrad()
        #     nit += 1
        # minuit.hesse()

        minuit.strategy = 1
        minuit.migrad(iterate=10)

        mle_unconstr = np.array(minuit.values)
        mle = np.array(self._to_params_array(mle_unconstr))
        jac = np.array(self._to_params_array_jac(mle_unconstr))
        cov = np.array(self._cov_unconstr(mle_unconstr))
        if np.isnan(cov).any() and minuit.fmin.has_covariance:
            cov = np.array(minuit.covariance)
        cov = jac @ cov @ jac.T
        err = np.sqrt(np.diagonal(cov))
        stat_group = self._deviance_group(mle_unconstr)
        stat = {k: float(stat_group[k]) for k in self._data.keys()}
        stat_total = minuit.fval
        stat |= {'total': stat_total}
        self._minuit = minuit

        if full_params:
            params_name = self._params
            params_idx = list(range(len(params_name)))
        else:
            params_name = self._interested_params
            params_idx = [self._params.index(i) for i in params_name]

        params = {p: (mle[i], err[i]) for p, i in zip(params_name, params_idx)}

        k = self._nparam
        n = self._nchan['total']

        return MLEResult(
            params=params,
            stat=stat,
            nchan=self._nchan,
            dof=self._dof,
            aic=stat_total + 2 * k + 2 * k * (k + 1) / (n - k - 1),
            bic=stat_total + k * np.log(n),
            state=minuit.fmin
        )

    def ci(
        self,
        params: Optional[str | Iterable[str]] = None,
        cl: float | int = 1,
        method: Literal['profile', 'boot'] = 'profile',
        n: Optional[int] = None,
    ) -> ConfidenceInterval:
        """Calculate confidence intervals for given parameters.

        If method is 'profile', Minos algorithm of Minuit is used here to find
        the profile likelihood based CIs, as it is considered as a robust error
        estimator.

        If method is 'boot', parametric bootstrap method is used to calculate
        the bootstrap CIs.

        Parameters
        ----------
        params : str or list of str, optional
            Parameters to calculate confidence intervals. If not provided,
            confidence intervals are calculated for all parameters.
        cl : float or int, optional
            Confidence level for the confidence interval. If 0 < `cl` < 1, the
            value is interpreted as the confidence level (a probability). If
            `cl` >= 1, it is interpreted as number of standard deviations.
            For example, ``cl=3`` produces a 3-sigma interval.
        method : {'profile', 'boot'}, optional
            Method used to calculate confidence. Either profile likelihood or
            parametric bootstrap method. The default is profile likelihood
            method.
        n : int, optional
            Number of bootstrap to calculate confidence intervals. Takes effect
            only if `method` is 'boot'. If None, set to the default value as in
            :meth:`LikelihoodFit.boot`.

        Returns
        -------
        ConfidenceInterval
            The confidence interval given the parameters and confidence level.

        """
        if self._minuit is None:
            msg = 'MLE must be performed before calculating CI'
            raise ValueError(msg)

        if not self._minuit.valid:
            msg = 'failed to calculate CI due to invalid fit'
            raise ValueError(msg)

        if params is None:
            params = self._interested_params

        elif isinstance(params, str):
            # check if params exist
            if params not in self._params:
                raise ValueError(f'parameter: {params} is not exist')

            params = [params]

        elif isinstance(params, Iterable):
            # check if params exist
            params = [str(i) for i in params]
            flag = [i in self._params for i in params]
            if not all(flag):
                params_err = ', '.join(
                    [i for i, j in zip(params, flag) if not j]
                )
                raise ValueError(f'parameters: {params_err} are not exist')

            params = list(str(i) for i in params)

        else:
            raise ValueError('params must be str, or sequence of str')

        free_params = [i for i in params if i in self._free_params]
        composite_params = [i for i in params if i in self._composite]

        cl_ = 1.0 - 2.0 * norm.sf(cl) if cl >= 1.0 else cl

        if method == 'profile':
            self._minuit.minos(*free_params, cl=cl)

            mle0 = self._minuit.values.to_dict()

            others = {  # set other unconstrained free parameter to mle
                i: mle0[i]
                for i in (set(mle0.keys()) - set(free_params))
            }

            ci = self._minuit.merrors
            l = self._to_params_dict(
                {k: mle0[k] + ci[k].lower for k in free_params} | others
            )
            u = self._to_params_dict(
                {k: mle0[k] + ci[k].upper for k in free_params} | others
            )

            mle = self._to_params_dict(mle0)
            interval = {k: (l[k], u[k]) for k in free_params}
            error = {k: (l[k] - mle[k], u[k] - mle[k]) for k in free_params}
            status = {
                k: {
                    'valid': (v.lower_valid, v.upper_valid),
                    'at_limit': (v.at_lower_limit, v.at_upper_limit),
                    'at_max_fcn': (v.at_lower_max_fcn, v.at_upper_max_fcn),
                    'new_min': (v.lower_new_min, v.upper_new_min),
                }
                for k, v in ci.items()
            }

            for p in composite_params:
                def loss(x):
                    """The loss when calculating CI of composite parameter.

                    Confidence interval of function of parameters, see, e.g.,
                    https://doi.org/10.1007/s11222-021-10012-y.
                    """
                    unconstr = {k: v for k, v in zip(self._free_params, x[1:])}
                    p0 = self._to_params_dict(unconstr)[p]
                    diff = (p0 / x[0] - 1) / 1e-3
                    return self._deviance(x[1:]) + diff*diff

                mle_p = mle[p]

                m = Minuit(
                    jax.jit(loss),
                    [mle_p, *self._minuit.values],
                    grad=jax.jit(jax.grad(loss)),
                )
                m.strategy = 2
                m.migrad()
                m.minos(0, cl=cl)
                ci = m.merrors['x0']
                interval[p] = (mle_p + ci.lower, mle_p + ci.upper)
                error[p] = (ci.lower, ci.upper)
                status[p] = {
                    'valid': (ci.lower_valid, ci.upper_valid),
                    'at_limit': (ci.at_lower_limit, ci.at_upper_limit),
                    'at_max_fcn': (ci.at_lower_max_fcn, ci.at_upper_max_fcn),
                    'new_min': (ci.lower_new_min, ci.upper_new_min),
                }

        elif method == 'boot':
            if n is None:
                self.boot()
            else:
                self.boot(n=n)
            mle = ...
            interval = ...
            error = ...
            status = ...

        else:
            raise ValueError('`method` must be either "profile" or "boot"')

        def format_result(v):
            """Order the result dict and use float as result type."""
            formatted = jax.tree_map(float, v)
            return {k: formatted[k] for k in params}

        return ConfidenceInterval(
            mle=format_result(mle),
            interval=format_result(interval),
            error=format_result(error),
            cl=cl_,
            status=status
        )

    def boot(self, n: int = 1000) -> BootstrapResult:
        # res = jaxopt.ScipyMinimize(
        #     fun=jax.jit(self._deviance),
        #     value_and_grad=jax.jit(jax.value_and_grad(self._deviance))
        # ).run(init_unconstr)
        # mle = np.array(self._to_constr(res.params))
        # jac = np.array(self._to_constr_jac(res.params))
        # cov_fun = jax.jit(
        #     lambda x: 2.0 * jnp.linalg.inv(jax.hessian(self._deviance)(x))
        # )
        # cov = np.array(cov_fun(res.params))
        # cov = jac @ cov @ jac.T
        # stat = float(res.state.fun_val)
        # state = {'success': res.state.success, 'status': res.state.status}
        ...


class CredibleInterval(NamedTuple):
    """Result of :meth:`BayesianFit.ci`."""
    ...


class PPCResult(NamedTuple):
    """Result of :meth:`BayesianFit.ppc`."""
    ...


class BayesianFit(BaseFit):
    def print_info(self) -> None:
        tab1 = _make_text_table(
            ['Data', 'Model'],
            list((k, v) for k, v in self._model_info.items())
        )
        tab2 = _make_text_table(
            ['No.', 'Component', 'Parameter', 'Value', 'Prior'],
            list(i[:4] + i[-1:] for i in self._params_info.values())
        )
        print('Bayesian Fit')
        print(tab1)
        print(tab2)

    def nuts(self, warmup=2000, samples=20000, chains=4, init=None):
        if init is None:
            init = {}

        sampler = MCMC(
            NUTS(
                self._numpyro_model,
                dense_mass=True,
                max_tree_depth=7,
                init_strategy=init_to_value(values=init)
            ),
            num_warmup=warmup,
            num_samples=samples,
            num_chains=chains,
            progress_bar=True,
        )

        sampler.run(jax.random.PRNGKey(self._seed))
        self._nuts_sampler = sampler
        self._nuts_idata = az.from_numpyro(
            sampler,
            # coords={'LE_channel': LE.channel, 'ME_channel': ME.channel, 'HE_channel': HE.channel},
            # dims={'LE_Non': ['LE_channel'], 'ME_Non': ['ME_channel'], 'HE_Non': ['HE_channel'],
            #       'LE_Noff': ['LE_channel'], 'ME_Noff': ['ME_channel'], 'HE_Noff': ['HE_channel']},
        )
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'Times New Roman'

    def ns(self):
        sampler = NestedSampler(
            self._numpyro_model,
            constructor_kwargs=dict(
                max_samples=100000,
                num_parallel_samplers=jax.device_count(),
                # num_parallel_workers=jax.device_count(),
            ),
            termination_kwargs=dict(
                live_evidence_frac=1e-5,
            )
        )

        t0 = time.time()
        print('Start nested sampling...')
        sampler.run(jax.random.PRNGKey(42))
        print(f'Nested sampling cost {time.time() - t0:.2f} s')
        samples = sampler.get_samples(jax.random.PRNGKey(self._seed), 80000)
        self._ns_sampler = sampler
        self._ns_idata = az.from_dict(samples)

    def ci(self):
        ...

    def ppc(self):
        ...


def _likelihood_func(data: Data, stat: str) -> Callable:
    """Wrap likelihood function."""
    name = data.name
    if stat == 'chi2':
        spec = data.net_counts
        error = data.net_error
        return partial(chi2, name=name, spec=spec, error=error)
    elif stat == 'cstat':
        return partial(cstat, name=name, spec=data.spec_counts)
    elif stat == 'pstat':
        spec = data.spec_counts
        back = data.back_counts
        ratio = data.spec_effexpo / data.back_effexpo
        return partial(pstat, name=name, spec=spec, back=back, ratio=ratio)
    elif stat == 'pgstat':
        spec = data.spec_counts
        back = data.back_counts
        back_error = data.back_error
        ratio = data.spec_effexpo / data.back_effexpo
        return partial(
            pgstat,
            name=name, spec=spec, back=back, back_error=back_error, ratio=ratio
        )
    elif stat == 'wstat':
        spec = data.spec_counts
        back = data.back_counts
        ratio = data.spec_effexpo / data.back_effexpo
        return partial(wstat, name=name, spec=spec, back=back, ratio=ratio)
    else:
        raise ValueError(f'stat "{stat}" is not supported')


def _ordered_composite(sample: dict, composite: dict) -> dict:
    ordered = {}

    remains = list(composite.items())
    while remains:
        s_and_d = sample | ordered
        i = remains.pop(0)
        name, (arg_names, func) = i
        if all(arg_name in s_and_d for arg_name in arg_names):
            ordered[name] = (arg_names, func)
        else:
            remains.append(i)

    return ordered


def _replace_by_mapping(mapping: dict[str, str], value: Any) -> Any:
    mapping = mapping.items()

    def replace_with_mapping(s: str):
        """Replace all k in s with v, as in mapping."""
        return reduce(lambda x, kv: x.replace(*kv), mapping, s)

    def replace_dict(d: dict):
        """Replace key and value of a dict."""
        return {replace(k): replace(v) for k, v in d.items()}

    def replace_iterable(it: tuple | list):
        """Replace element of a dict."""
        return type(it)(map(replace, it))

    def replace(v):
        """Main replace function."""
        if isinstance(v, dict):
            return replace_dict(v)
        elif isinstance(v, (list, tuple)):
            return replace_iterable(v)
        elif isinstance(v, str):
            return replace_with_mapping(v)
        else:
            return v

    return replace(value)


def _make_text_table(fields: Iterable[str], rows: Iterable) -> PrettyTable:
    table = PrettyTable(
        fields,
        align='c',
        hrules=1,  # 1 for all, 0 for frame
        vrules=1,
        padding_width=1,
        vertical_char='│',
        horizontal_char='─',
        junction_char='┼',
        top_junction_char='┬',
        bottom_junction_char='┴',
        right_junction_char='┤',
        left_junction_char='├',
        top_right_junction_char='┐',
        top_left_junction_char='┌',
        bottom_right_junction_char='┘',
        bottom_left_junction_char='└'
    )
    table.add_rows(rows)
    return table
