"""Handle model fitting."""
from __future__ import annotations

import time
from functools import partial, reduce
from typing import Callable, NamedTuple

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from iminuit import Minuit
from iminuit.util import FMin
import jaxopt
# from numpyro.contrib.nested_sampling import NestedSampler
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import log_likelihood, constrain_fn, unconstrain_fn

from ..data.ogip import Data
from ..model.base import Model
from .likelihood import chi2, cstat, pstat, pgstat, wstat
from .nested_sampling import NestedSampler


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


def _ordered_deterministic(sample: dict, deterministic: dict) -> dict:
    ordered = {}

    remains = list(deterministic.items())
    while remains:
        s_and_d = sample | ordered
        i = remains.pop(0)
        determ, (arg_names, func) = i
        if all(arg_name in s_and_d for arg_name in arg_names):
            ordered[determ] = (arg_names, func)
        else:
            remains.append(i)

    return ordered


class BaseFit:
    """Fitting context.

    Parameters
    ----------
    data : Data or list of Data
        The observation data.
    model : Model or list of Model
        The model used  to fit the data.
    stat : str or list of str
        The likelihood option for the data and model.
    seed : int

    """

    _stat_options = {
        'chi2',
        'cstat',
        'pstat',
        'pgstat',
        'wstat'
    }

    def __init__(
        self,
        data: Data | list[Data],
        model: Model | list[Model],
        stat: str | list[str],
        seed: int = 42
    ):
        data, model, stat = self._sanity_check(data, model, stat)

        self._data = {d.name: d for d in data}
        self._model = {d.name: m for d, m in zip(data, model)}
        self._stat = {d.name: s for d, s in zip(data, stat)}
        self._seed = int(seed)

        # model information
        info_list = [m._model_info for m in model]
        self._model_func = {
            i.name: j.pop('func')
            for i, j in zip(data, info_list)
        }

        self._model_info = {
            i: reduce(lambda x, y: x | y, (j[i] for j in info_list))
            for i in [
                'sample', 'deterministic', 'default', 'params', 'name', 'fmt'
            ]
        }
        self._sample = {
            k: v for k, v in self._model_info['sample'].items()
            if not isinstance(v, float)
        }
        self._frozen = {
            k: v for k, v in self._model_info['sample'].items()
            if isinstance(v, float)
        }
        self._deterministic = _ordered_deterministic(
            self._model_info['sample'], self._model_info['deterministic']
        )

        self._model_params = {
            name: {minfo['name'][k]: v for k, v in minfo['params'].items()}
            for name, minfo in zip(self._data, info_list)
        }

        # likelihood function
        self._stat_func = {
            d.name: _likelihood_func(d, s)
            for d, s in zip(data, stat)
        }

        self._numpyro_model = self._generate_numpyro_model()

        self._nchan = {d.name: len(d.channel) for d in data}
        self._nchan['total'] = sum(self._nchan.values())
        self._nparam = len(self._sample.keys())
        self._dof = self._nchan['total'] - self._nparam

        # add data name to components if they are not fit with all data
        # nd = len(data)
        # comps = [j for i in model for j in i._comps]
        # comps = np.unique(comps).tolist()
        # comp_suffix = {}
        # for comp in comps:
        #     data_having_this_comp = []
        #     for name in self._data:
        #         if comp in self._model[name]._comps:
        #             data_having_this_comp.append(name)
        #
        #     if len(data_having_this_comp) == nd:
        #         comp_suffix[comp] = ''
        #     else:
        #         joined = '+'.join(data_having_this_comp)
        #         comp_suffix[comp] = r'^\mathrm{' + joined + '}'
        #
        # comp_mapping = {
        #     k: '$' + v + comp_suffix[k] + '$'
        #     for k, v in self._model_info['fmt'].items()
        # }
        # params = self._model_params['params']

    def _generate_numpyro_model(self) -> Callable:
        sample = self._sample
        frozen = self._frozen
        deterministic = self._deterministic
        model_func = self._model_func
        model_params = self._model_params
        data = self._data
        names = list(data.keys())
        stat_func = self._stat_func

        def numpyro_model():
            """The numpyro model."""
            params = {
                name: numpyro.sample(name, dist)
                for name, dist in sample.items()
            }
            params |= frozen
            for determ, (arg_names, func) in deterministic.items():
                args = (params[arg_name] for arg_name in arg_names)
                params[determ] = numpyro.deterministic(determ, func(*args))

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

    @classmethod
    def stat_options(cls) -> list:
        """List of available likelihood options."""
        return list(cls._stat_options)

    def _sanity_check(
        self,
        data: Data | list[Data],
        model: Model | list[Model],
        stat: str | list[str]
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

    level: float
    interval: dict[str, tuple[float, float]]


class BootstrapResult(NamedTuple):
    """Result of :meth:`LikelihoodFit.boot`."""

    ...


class LikelihoodFit(BaseFit):
    def __init__(
        self,
        data: Data | list[Data],
        model: Model | list[Model],
        stat: str | list[str],
        seed: int = 42
    ):
        super().__init__(data, model, stat, seed)

        free_params = list(self._sample.keys())
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

        def to_constr_array(unconstrained_params_array):
            """Transform unconstrained parameters array to constrained."""
            constr = to_constr_dict(unconstrained_params_array)
            return jnp.array([constr[i] for i in free_params])

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

        free = [self._model_info['default'][k] for k in free_params]
        init_unconstr = np.array(to_unconstr_array(free))

        self._free_params = free_params
        self._init_unconstr = init_unconstr
        self._deviance = deviance_unconstr
        self._deviance_group = jax.jit(deviance_unconstr_group)
        self._grad_unconstr = jax.jit(jax.grad(deviance_unconstr))
        self._cov_unconstr = jax.jit(
            lambda x: 2.0 * jnp.linalg.inv(jax.hessian(deviance_unconstr)(x))
        )
        self._to_constr = jax.jit(to_constr_array)
        self._to_constr_jac = jax.jit(jax.jacobian(to_constr_array))
        self._to_unconstr = jax.jit(to_unconstr_array)
        self._minuit = None
        self._ns = None

    def mle(self, init=None, ns_init=False) -> MLEResult:
        """Maximum Likelihood Estimation (MLE) for the model.

        Parameters
        ----------
        init : dict, optional
            Initial guess for the MLE. The default is None.
        ns_init : bool, optional
            Whether to use nested sampling to find the initial guess for the
            MLE. The default is False.

        Returns
        -------
        MLEResult
            The MLE result.

        """
        if bool(ns_init) and self._ns is None:
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
            print(f'Global Fit: {time.time() - t0:.2f} s cost')
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

        max_it = 10
        nit = 0
        minuit.strategy = 0
        minuit.migrad()
        while (not minuit.fmin.is_valid) and nit < max_it:
            minuit.hesse()
            minuit.migrad()
            nit += 1
        minuit.hesse()

        mle_unconstr = np.array(minuit.values)
        mle = np.array(self._to_constr(mle_unconstr))
        jac = np.array(self._to_constr_jac(mle_unconstr))
        # cov = np.array(self._minuit.covariance)
        cov = np.array(self._cov_unconstr(mle_unconstr))
        cov = jac @ cov @ jac.T
        stat_group = self._deviance_group(mle_unconstr)
        stat = {k: float(stat_group[k]) for k in self._data.keys()}
        stat_total = minuit.fval
        stat |= {'total': stat_total}
        self._minuit = minuit

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

        err = np.sqrt(np.diagonal(cov))
        params = {p: (v, e) for p, v, e in zip(self._free_params, mle, err)}

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

    def ci(self) -> ConfidenceInterval:
        # confidence interval of function of parameters, see
        # https://doi.org/10.1007/s11222-021-10012-y
        ...

    def boot(self) -> BootstrapResult:
        ...


class CredibleInterval(NamedTuple):
    """Result of :meth:`BayesianFit.ci`."""
    ...


class PPCResult(NamedTuple):
    """Result of :meth:`BayesianFit.ppc`."""
    ...


class BayesianFit(BaseFit):
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
        self._nuts_idata = az.from_numpyro(sampler)

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
