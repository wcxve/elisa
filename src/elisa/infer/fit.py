"""Model fit in maximum likelihood or Bayesian way."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from functools import partial, reduce
from typing import Callable, Literal, NamedTuple, Optional, Sequence, TypeVar

import arviz as az
import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import numpyro
from iminuit import Minuit
from jax import lax
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import constrain_fn, unconstrain_fn, log_likelihood
from prettytable import PrettyTable

from ..data.ogip import Data
from ..model.base import Model
from .analysis import MLEResult, PosteriorResult
from .likelihood import chi2, cstat, pstat, pgstat, wstat
from .nested_sampling import NestedSampler
from .util import (
    make_pretty_table, order_composite, progress_bar_factory, replace_string
)

__all__ = ['LikelihoodFit', 'BayesianFit']

Statistic = Literal['chi2', 'cstat', 'pstat', 'pgstat', 'wstat']
T = TypeVar('T')


class BaseFit(ABC):
    """Model fitting.

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
    seed : int, optional
        Random number generator seed. The default is 42.

    """
    # TODO: introduce background model that directly fit the background data

    _stat_options: set[str] = {
        'chi2',
        'cstat',
        'pstat',
        'pgstat',
        'wstat'
        # It should be noted that 'lstat' does not have long run coverage
        # property for source estimation, which is probably due to the choice
        # of conjugate prior of Poisson background data.
        # 'lstat' will be included here with a proper prior at some point.
    }

    _stat_with_back: set[str] = {
        'pgstat',
        'wstat'
    }

    def __init__(
        self,
        data: Data | Sequence[Data],
        model: Model | Sequence[Model],
        stat: Statistic | Sequence[Statistic],
        seed: int = 42,
    ):
        data, model, stat = self._sanity_check(data, model, stat)

        self._seed = int(seed)
        self._PRNGKey = jax.random.PRNGKey(self._seed)

        data_dict: dict[str, Data] = {}
        model_dict: dict[str, Model] = {}
        stat_dict: dict[str, Statistic] = {}
        for d, m, s in zip(data, model, stat):
            data_name = d.name
            data_dict[data_name] = d
            model_dict[data_name] = m
            stat_dict[data_name] = s
        self._data = data_dict
        self._model = model_dict
        self._stat = stat_dict

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
            # k: r'$\big[' + v + fmt_suffix[k] + r'\big]$'
            k: r'$[' + v + fmt_suffix[k] + r']$'
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
        mpfmt = info['mpfmt']
        params_name = {}
        params_fmt = {}
        for m, ps in info['params'].items():
            for p, p_id in ps.items():
                params_name[p_id] = f'{mname[m]}_{p}'
                params_fmt[p_id] = rf'{mfmt[m]}\ ${mpfmt[m][p]}$'

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

        # first avoid renaming conflict then create the mapping
        composite_order = order_composite(info['sample'], info['composite'])
        name_order = list(reversed(composite_order.keys()))
        name_order += list(info['sample'].keys())
        name_order += list(info['mname'].keys())
        id_mapping = params_name | aux_params_name | info['mname']
        id_mapping = {k: id_mapping[k] for k in name_order}

        self._info = replace_string(id_mapping, info)

        # model information will be displayed
        self._model_info = replace_string(
            id_mapping,
            {k: (v._node.name, self._stat[k]) for k, v in self._model.items()}
        )
        # model information will be displayed

        # parameter information will be displayed
        sample_dic = self._info['sample']
        default_dic = self._info['default']
        min_dic = self._info['min']
        max_dic = self._info['max']
        dist_dic = self._info['dist_expr']
        # params name having id as suffix
        params_dic = replace_string(info['mname'], info['params'])
        idx = 1
        params_info = {}
        for comp, param_dic in self._info['params'].items():
            for i, j in param_dic.items():
                if j in sample_dic:
                    if not isinstance(sample_dic[j], float):  # free parameter
                        bound = f'({min_dic[j]}, {max_dic[j]})'
                        pidx = str(idx)
                        idx += 1
                    else:  # fixed parameter
                        bound = ''
                        pidx = ''

                    params_info[f'{comp}_{i}'] = [
                        pidx, comp, i, default_dic[j], bound, dist_dic[j]
                    ]

                else:  # composite parameter
                    mapping = {
                        k: v
                        for k, v in id_mapping.items()
                        if v != f'{comp}_{i}'
                    }
                    expr = replace_string(mapping, params_dic[comp][i])
                    params_info[f'{comp}_{i}'] = [
                        '', comp, i, expr, '', ''
                    ]

        for k, v in self._info['sample'].items():
            if k in params_info:
                continue

            if not isinstance(v, float):  # free parameter
                bound = f'({min_dic[k]}, {max_dic[k]})'
                pidx = str(idx)
                idx += 1
            else:  # fixed parameter
                bound = ''
                pidx = ''

            params_info[k] = [
                pidx, '', k, default_dic[k], bound, dist_dic[k]
            ]
        self._params_info = params_info

        # parameters Tex format
        self._params_fmt = replace_string(
            id_mapping,
            params_fmt | aux_params_fmt
        )

        # parameters of spectral model function
        spec_params = {
            name: {minfo['mname'][k]: v for k, v in minfo['params'].items()}
            for name, minfo in zip(self._data, info_list)
        }
        self._spec_params = replace_string(id_mapping, spec_params)

        # free parameters
        self._free = {
            k: v for k, v in self._info['sample'].items()
            if not isinstance(v, float)
        }

        # fixed parameters
        self._fixed = {
            k: v for k, v in self._info['sample'].items()
            if isinstance(v, float)
        }

        # composite parameter
        self._composite = order_composite(
            self._info['sample'], self._info['composite']
        )

        # ordered free parameters names
        self._free_names = tuple(
            k for k, v in self._params_info.items() if v[0]
        )

        # ordered parameters of interest,
        # which are directly input to model and not fixed
        self._interest_names = tuple(
            k for k, v in self._params_info.items()
            if v[1] and k not in self._fixed
        )

        # ordered parameters
        self._params_names = tuple(self._params_info.keys())

        # channel number of data
        self._ndata = {d.name: len(d.channel) for d in data}
        self._ndata['total'] = sum(self._ndata.values())

        # number of free parameters
        self._nparam = len(self._free.keys())

        # degree of freedom
        self._dof = self._ndata['total'] - self._nparam

        # a collection of functions will be used in optimization
        self._helper = generate_helper(self)
        # tell iminuit the number of data points
        self._helper.deviance_unconstr.ndata = self._ndata['total']

        # the numpyro model
        self._numpyro_model = self._helper.numpyro_model

        # default initial parameter in unconstrained space
        free = [self._info['default'][k] for k in self._free_names]
        self._init_unconstr = np.array(self._helper.to_unconstr_array(free))

        # self._simfit = SimFit()

        # make some pretty tables to display model information
        self._tab1: PrettyTable
        self._tab2: PrettyTable
        self._make_info_table()

    @abstractmethod
    def _repr_html_(self) -> str:
        """The repr call in Jupyter notebook environment."""
        pass

    @abstractmethod
    def _make_info_table(self) -> None:
        """Make the model and params info table."""
        pass

    def _sanity_check(
        self,
        data: Data | Sequence[Data],
        model: Model | Sequence[Model],
        stat: Statistic | Sequence[Statistic]
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
                    raise ValueError(f'all {name} must be a valid {tname}')
                input_list = list(inputs)
            else:
                raise ValueError(f'got wrong type {type(inputs)} for {name}')
            return input_list

        data_list = get_list(data, 'data', Data, 'Data')
        model_list = get_list(model, 'model', Model, 'Model')
        stat_list = get_list(stat, 'stat', str, 'str')

        # check if data name is unique
        name_list = list(d.name for d in data_list)
        if len(set(name_list)) != len(data_list):
            msg = f'data names are not unique: {", ".join(name_list)}'
            raise ValueError(msg)

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
            if not d.spec_poisson and s != 'chi2':
                msg = f'{name} is Gaussian data, use stat "chi2" but not "{s}"'
                raise ValueError(msg)

            if s == 'cstat' and d.has_back:
                back = 'Poisson' if d.back_poisson else 'Gaussian'
                stat1 = 'W' if d.back_poisson else 'PG'
                stat2 = 'w' if d.back_poisson else 'pg'
                msg = 'C-statistic (cstat) is not valid for Poisson data '
                msg += f'with {back} background, use {stat1}-statistic'
                msg += f'({stat2}stat) for {name} instead'
                raise ValueError(msg)

            elif s == 'pstat' and not d.has_back:
                msg = f'P-statistic (pstat) is not valid for {name}, which '
                msg += 'requires background file, use C-statistic (cstat) '
                msg += 'instead'
                raise ValueError(msg)

            elif s == 'pgstat' and not d.has_back:
                msg = f'PG-statistic is not valid for {name}, which requires '
                msg += 'Gaussian background data, use C-statistic (cstat) '
                msg += 'instead'
                raise ValueError(msg)

            elif s == 'wstat' and not (d.has_back and d.back_poisson):
                msg = f'W-statistic is not valid for {name}, which requires '
                msg += 'Poisson background data, use C-statistic (cstat) '
                msg += 'instead'
                raise ValueError(msg)

        for d, s in zip(data_list, stat_list):
            check_data_stat(d, s)

        return data_list, model_list, stat_list


class LikelihoodFit(BaseFit):
    _ns: NestedSampler | None = None

    def __repr__(self) -> str:
        s = 'Likelihood Fit\n'
        s += self._tab1.get_string() + '\n'
        s += self._tab2.get_string()
        return s

    def _repr_html_(self) -> str:
        s = '<b>Likelihood Fit</b>\n'
        s += self._tab1.get_html_string(format=True) + '\n'
        s += self._tab2.get_html_string(format=True)
        return s

    def _make_info_table(self) -> None:
        self._tab1 = make_pretty_table(
            ['Data', 'Model', 'Statistic'],
            list((k, *v) for k, v in self._model_info.items())
        )
        self._tab2 = make_pretty_table(
            ['No.', 'Component', 'Parameter', 'Value', 'Bound'],
            list(i[:-1] for i in self._params_info.values())
        )

    def mle(
        self,
        init: Optional[dict[str, float]] = None,
        lopt: Literal['minuit', 'lm'] = 'minuit',
        strategy: Literal[0, 1, 2] = 1,
        gopt: Optional[str] = None
    ) -> MLEResult:
        """Find the Maximum Likelihood Estimation (MLE) for the model.

        Migrad optimization of :mod:`iminuit` is the default algorithm here to
        perform the MLE, as it is considered as a robust optimiser.
        Levenberg-Marquardt algorithm implemented in :mod:`jaxopt` is the
        optional method.

        Parameters
        ----------
        init : dict, optional
            Initial guess for the MLE. The default is None.
        lopt : {'minuit', 'lm'}, optional
            Local optimization algorithm to use.
            Available options are:
                * 'minuit': Migrad algorithm of Minuit.
                * 'lm': Levenberg-Marquardt algorithm of :mod:`jaxopt`.
            The default is 'minuit'.
        strategy : {0, 1, 2}, optional
            Optimization strategy to use in Minuit.
            Available options are:
                * 0: Fast.
                * 1: Default.
                * 2: Careful. This improves accuracy at the cost of time.
        gopt : {'ns'}, optional
            Global optimization algorithm to find the initial guess for MLE.
            Available options are:
                * 'ns' : nested sampling of :mod:`jaxns`.
            The default is None.

        Returns
        -------
        MLEResult
            The MLE result.

        """
        if gopt == 'ns' and self._ns is None:
            ns = NestedSampler(
                self._numpyro_model,
                constructor_kwargs=dict(
                    num_live_points=max(800, 100 * self._nparam),
                    max_samples=100000,
                    # num_parallel_samplers=jax.device_count(),
                    num_parallel_workers=1,
                ),
                termination_kwargs=dict(
                    live_evidence_frac=1e-5,
                )
            )

            t0 = time.time()
            print('Start nested sampling...')
            ns.run(jax.random.PRNGKey(42))
            print(f'Global optimization cost {time.time() - t0:.2f} s')
            mle_idx = ns._results.log_L_samples.argmax()
            mle_constr = jax.tree_map(
                lambda s: s[mle_idx], ns._results.samples
            )
            mle_constr = [mle_constr[i] for i in self._free_names]
            init_unconstr = self._helper.to_unconstr_array(mle_constr)
            self._ns = ns

        elif init is not None:
            init_constr = [init[i] for i in self._free_names]
            init_unconstr = self._helper.to_unconstr_array(init_constr)

        else:
            init_unconstr = self._init_unconstr

        if lopt == 'lm':
            res = jax.jit(jaxopt.LevenbergMarquardt(
                self._helper.residual,
                stop_criterion='grad-l2-norm'
            ).run)(jnp.array(self._init_unconstr))
            init_unconstr = res.params
        elif lopt != 'minuit':
            raise ValueError(f'invalid local optimization method {lopt}')

        minuit = Minuit(
            self._helper.deviance_unconstr,
            np.array(init_unconstr),
            grad=self._helper.deviance_unconstr_grad,
            name=self._free_names
        )

        # TODO: use simplex to "polish" the initial guess?
        if strategy == 0:
            max_it = 10
            nit = 0
            minuit.strategy = 0
            minuit.migrad()
            while (not minuit.fmin.is_valid) and nit < max_it:
                minuit.hesse()
                minuit.migrad()
                nit += 1
            minuit.hesse()

        elif strategy in (1, 2):
            minuit.strategy = strategy
            minuit.migrad(iterate=10)
        else:
            raise ValueError(f'invalid strategy {strategy}')

        return MLEResult(minuit, self)


class BayesianFit(BaseFit):
    def __repr__(self) -> str:
        s = 'Bayesian Fit\n'
        s += self._tab1.get_string() + '\n'
        s += self._tab2.get_string()
        return s

    def _repr_html_(self) -> str:
        s = '<b>Bayesian Fit</b>\n'
        s += self._tab1.get_html_string(format=True) + '\n'
        s += self._tab2.get_html_string(format=True)
        return s

    def _make_info_table(self) -> None:
        self._tab1 = make_pretty_table(
            ['Data', 'Model', 'Statistic'],
            list((k, *v) for k, v in self._model_info.items())
        )
        self._tab2 = make_pretty_table(
            ['No.', 'Component', 'Parameter', 'Value', 'Prior'],
            list(i[:4] + i[-1:] for i in self._params_info.values())
        )

    def nuts(
        self,
        warmup=2000,
        samples=20000,
        chains: Optional[int] = None,
        init: Optional[dict[str, float]] = None,
        progress: bool = True,
        nuts_kwargs: Optional[dict] = None
    ) -> PosteriorResult:
        """Run the No-U-Turn Sampler (NUTS) of :mod:`numpyro`.

        Parameters
        ----------
        warmup : int, optional
            Number of warmup steps.
        samples : int, optional
            Number of samples to generate from each chain.
        chains : int, optional
            Number of MCMC chains to run. If there are not enough devices
            available, chains will be run in sequence. If None, will run
            ``jax.device_count()`` chains.
        init : dict, optional
            Initial parameter for sampler to start from.
        progress : bool, optional
            Whether to show progress bar during sampling. The default is True.
        nuts_kwargs : dict
            Other parameters for :class:`numpyro.infer.NUTS`.

        Returns
        -------
        inference_data : az.InferenceData
            The inference data.

        """
        if chains is None:
            chains = jax.device_count()

        if init is None:
            init = {}

        if nuts_kwargs is None:
            nuts_kwargs = {}
        else:
            nuts_kwargs.pop('model', None)
            nuts_kwargs.pop('init_strategy', None)

        dense_mass = nuts_kwargs.pop('dense_mass', True)
        max_tree_depth = nuts_kwargs.pop('max_tree_depth', 10)

        sampler = MCMC(
            NUTS(
                self._numpyro_model,
                dense_mass=dense_mass,
                max_tree_depth=max_tree_depth,
                init_strategy=init_to_value(values=init)
            ),
            num_warmup=warmup,
            num_samples=samples,
            num_chains=chains,
            progress_bar=progress,
        )

        sampler.run(self._PRNGKey)

        idata = self._generate_inference_idata(sampler)

        # import matplotlib.pyplot as plt
        # plt.rcParams['text.usetex'] = True
        # plt.rcParams['font.family'] = 'serif'

        return idata

    def ns(self, *args, **kwargs) -> PosteriorResult:
        """Run the Nested Sampler of :mod:`jaxns`.

        Returns
        -------

        """
        sampler = NestedSampler(
            self._numpyro_model,
            constructor_kwargs=dict(
                max_samples=100000,
                # num_parallel_samplers=jax.device_count(),
                num_parallel_workers=jax.device_count(),
            ),
            termination_kwargs=dict(
                live_evidence_frac=1e-5,
            )
        )

        t0 = time.time()
        print('Start nested sampling...')
        sampler.run(self._PRNGKey)
        print(f'Nested sampling cost {time.time() - t0:.2f} s')

        idata = self._generate_inference_idata(sampler)

        return idata

    def _generate_inference_idata(self, sampler) -> az.InferenceData:
        if not isinstance(sampler, (MCMC, NestedSampler)):
            raise ValueError(f'unknown sampler type {type(sampler)}')

        coords = {
            f'{k}_channel': v.channel
            for k, v in self._data.items()
        }

        dims = {
           f'{k}_Non': [f'{k}_channel']
           for k, v in self._data.items()
        } | {
           f'{k}_Noff': [f'{k}_channel']
           for k, v in self._data.items()
           if v.has_back and self._stat[k] in self._stat_with_back
        }

        if isinstance(sampler, MCMC):
            idata = az.from_numpyro(sampler, coords=coords, dims=dims)
        else:
            # TODO
            samples = sampler.get_samples(self._PRNGKey, 80000)
            idata = az.from_dict(samples)

        ln_likelihood = idata['log_likelihood']
        observation = idata['observed_data']

        for k, v in self._data.items():
            # channel-wise log likelihood of data group
            if f'{k}_Noff' in ln_likelihood:
                ln_likelihood[k] = ln_likelihood[f'{k}_Non'] \
                                   + ln_likelihood[f'{k}_Noff']
            else:
                ln_likelihood[k] = ln_likelihood[f'{k}_Non']

            # net counts of data group
            observation[k] = ((f'{k}_channel',), v.net_counts)

        # channel-wise log likelihood
        ln_likelihood['channels'] = (
            ('chain', 'draw', 'channel'),
            np.concatenate([ln_likelihood[i] for i in self._data], axis=-1)
        )

        # channel-wise net counts
        observation['channels'] = (
            ('channel',),
            np.concatenate([observation[i] for i in self._data], axis=-1)
        )

        # total log likelihood
        ln_likelihood['total'] = ln_likelihood['channels'].sum('channel')

        # total net counts
        observation['total'] = observation['channels'].sum('channel')

        channel_coords = np.hstack([d.channel for d in self._data.values()])
        idata = idata.assign_coords({'channel': channel_coords})

        return idata


class HelperFn(NamedTuple):
    """A collection of helper functions."""
    numpyro_model: Callable
    to_dict: Callable
    to_constr_dict: Callable
    to_unconstr_dict: Callable
    to_unconstr_array: Callable
    to_params_dict: Callable
    to_params_array: Callable
    deviance_unconstr: Callable
    deviance_unconstr_info: Callable
    deviance_unconstr_grad: Callable
    unconstr_covar: Callable
    params_covar: Callable
    residual: Callable
    params_by_group: Callable
    net_counts: Callable
    model_counts: Callable
    sim_residual: Callable
    sim_deviance_unconstr_info: Callable
    sim_result_container: Callable
    sim_fit_one: Callable
    sim_sequence_fit: Callable
    sim_parallel_fit: Callable


def generate_helper(fit: BaseFit) -> HelperFn:
    """Generates a collection of helper functions."""
    # ====================== function to get parameters =======================
    spec_params = fit._spec_params

    @jax.jit
    def params_by_group(constr_dict: dict) -> dict:
        """Get parameters of spectral model for each data group."""
        return {
            k: jax.tree_map(lambda n: constr_dict[n], v)
            for k, v in spec_params.items()
        }

    # ================== function to calculate model counts ===================
    data = fit._data
    spec_model = {k: v._wrapped_fn for k, v in fit._model.items()}

    egrid = {k: v.ph_egrid for k, v in data.items()}
    resp = {k: v.resp_matrix for k, v in data.items()}
    expo = {k: v.spec_effexpo for k, v in data.items()}

    @jax.jit
    def model_counts(constr_dict: dict) -> dict:
        """Calculate model counts for each data group."""
        p = params_by_group(constr_dict)
        return jax.tree_map(
            lambda mi, pi, ei, ri, ti: mi(ei, pi) @ ri * ti,
            spec_model, p, egrid, resp, expo
        )

    # ========================= create numpyro model ==========================
    free = fit._free
    fixed = fit._fixed
    composite = fit._composite
    stat = fit._stat
    stat_fn = jax.tree_map(_likelihood_fn, data, stat)

    deterministic = []  # record composite params directly input to model
    for i in fit._info['params'].values():
        for j in i.values():
            if j not in free and j not in deterministic:
                deterministic.append(j)

    def numpyro_model(predictive=False):
        """The numpyro model."""
        params = {
            name: numpyro.sample(name, dist)
            for name, dist in free.items()
        } | {
            k: numpyro.deterministic(k, jnp.array(v))
            for k, v in fixed.items()
        }
        for name, (arg_names, fn) in composite.items():
            args = (params[arg_name] for arg_name in arg_names)
            if name in deterministic:
                v = numpyro.deterministic(name, fn(*args))
            else:
                v = fn(*args)
            params[name] = v

        jax.tree_map(
            lambda f, m: f(m, predictive=predictive),
            stat_fn, model_counts(params)
        )

    # ============================ other functions ============================
    free_names = fit._free_names
    params_names = fit._params_names
    data_names = tuple(data.keys())
    data_stat = fit._stat
    stat_with_back = fit._stat_with_back

    net_fn = {}  # function to calculate net counts
    group_name = {}  # data name of each observation
    tmp = {}
    for name, d in data.items():
        names = [f'{name}_Non']
        code = f'{name} = lambda data: data["{name}_spec"]'

        if data_stat[name] in stat_with_back:
            names.append(f'{name}_Noff')
            ratio = d.spec_effexpo / d.back_effexpo
            code = f'{code} - {ratio} * data["{name}_back"]'

        group_name[name] = names
        exec(code, tmp)
        net_fn[name] = tmp[name]

    @jax.jit
    def net_counts(data_dict: dict) -> dict:
        """Calculate net counts for each data group."""
        return jax.tree_map(lambda f: f(data_dict), net_fn)

    @jax.jit
    def to_dict(array: Sequence[T]) -> dict[str, T]:
        """Transform parameter array to a dict."""
        return {k: v for k, v in zip(free_names, array)}

    @jax.jit
    def to_constr_dict(unconstr_array: Sequence) -> dict:
        """Transform unconstrained parameter array to constrained dict."""
        return constrain_fn(
            model=numpyro_model,
            model_args=(),
            model_kwargs={},
            params=to_dict(unconstr_array)
        )

    @jax.jit
    def to_unconstr_dict(constr_array: Sequence) -> dict:
        """Transform constrained parameter array to unconstrained dict."""
        return unconstrain_fn(
            model=numpyro_model,
            model_args=(),
            model_kwargs={},
            params=to_dict(constr_array)
        )

    @jax.jit
    def to_unconstr_array(constr_array: Sequence) -> jnp.ndarray:
        """Transform constrained parameter array to unconstrained."""
        unconstr = to_unconstr_dict(constr_array)
        return jnp.array([unconstr[i] for i in free_names])

    @jax.jit
    def deviance_unconstr(unconstr_array: Sequence) -> float:
        """Deviance in unconstrained parameter space."""
        p = to_constr_dict(unconstr_array)
        return -2.0 * jax.tree_util.tree_reduce(
            lambda x, y: x + y,
            jax.tree_map(
                lambda x: x.sum(),
                log_likelihood(numpyro_model, p)
            )
        )

    # deviance_unconstr_info will be used in simulation,
    # therefore we do not jit it, otherwise data substitution will fail
    def deviance_unconstr_info(unconstr_array: Sequence) -> dict:
        """Deviance of data group and data point in unconstrained space."""
        p = to_constr_dict(unconstr_array)

        log_like = log_likelihood(numpyro_model, p)
        deviance = jax.tree_map(lambda x: -2.0 * x, log_like)

        group = {
            k: sum(deviance[i].sum() for i in v)
            for k, v in group_name.items()
        }

        point = {
            k: jnp.asarray(list(deviance[i] for i in v)).sum(axis=0)
            for k, v in group_name.items()
        }

        return {'group': group, 'point': point}

    @jax.jit
    def to_params_dict(unconstr_dict: dict) -> dict:
        """Transform unconstrained dict array to constrained parameters."""
        return constrain_fn(
            model=numpyro_model,
            model_args=(),
            model_kwargs={},
            params=unconstr_dict,
            return_deterministic=True
        )

    @jax.jit
    def to_params_array(unconstr_array: Sequence) -> jnp.ndarray:
        """Transform unconstrained parameters array to constrained."""
        unconstr = to_dict(unconstr_array)
        constr = to_params_dict(unconstr)  # including deterministic site
        return jnp.array([constr[i] for i in params_names])

    @jax.jit
    def unconstr_covar(unconstr_array: Sequence) -> jnp.ndarray:
        """Covariance matrix in unconstrained space."""
        cov = jnp.linalg.inv(jax.hessian(deviance_unconstr)(unconstr_array))
        return 2 * cov

    @jax.jit
    def params_covar(
        unconstr_array: Sequence,
        cov_unconstr: Sequence
    ) -> jnp.ndarray:
        """Covariance matrix in constrained space."""
        jac = jax.jacobian(to_params_array)(unconstr_array)
        return jac @ cov_unconstr @ jac.T

    # residual will be used in simulation,
    # therefore we do not jit it, otherwise data substitution will fail
    def residual(unconstr_array: Sequence) -> jnp.ndarray:
        """Likelihood residual (i.e. sqrt deviance) function."""
        p = to_constr_dict(unconstr_array)
        log_like = log_likelihood(numpyro_model, p)
        log_like_array = jnp.hstack(list(log_like.values()))
        return jnp.sqrt(-2.0 * log_like_array)

    # ===================== functions used in simulation ======================
    ndata = {k: v for k, v in fit._ndata.items() if k != 'total'}

    def sim_result_container(n: int):
        """Make a fitting result container for simulation data."""
        return {
            'params_rep': {k: jnp.empty(n) for k in params_names},
            'model_rep': {k: jnp.empty((n, v)) for k, v in ndata.items()},
            'stat_rep': {
                'total': jnp.empty(n),
                'group': {k: jnp.empty(n) for k in data_names},
                'point': {k: jnp.empty((n, v)) for k, v in ndata.items()}
            },
            'params_fit': {k: jnp.empty(n) for k in params_names},
            'model_fit': {k: jnp.empty((n, v)) for k, v in ndata.items()},
            'stat_fit': {
                'total': jnp.empty(n),
                'group': {k: jnp.empty(n) for k in data_names},
                'point': {k: jnp.empty((n, v)) for k, v in ndata.items()}
            },
            'valid': jnp.full(n, True, bool)
        }

    @jax.jit
    def sim_fit_one(i, args):
        """Fit the i-th simulation data."""
        sim_data, result, init = args

        new_data = jax.tree_map(lambda x: x[i], sim_data)
        new_residual = handlers.substitute(
            fn=residual,
            data=new_data
        )
        new_deviance_info = handlers.substitute(
            fn=deviance_unconstr_info,
            data=new_data
        )

        # update best fit params to result
        params = to_params_dict(to_dict(init[i]))
        for k in result['params_rep']:
            result['params_rep'][k] = \
                result['params_rep'][k].at[i].set(params[k])

        # update unfit model to result
        model = model_counts(to_constr_dict(init[i]))
        for k in data_names:
            result['model_rep'][k] = result['model_rep'][k].at[i].set(model[k])

        # update unfit deviance to result
        stat_info = new_deviance_info(init[i])
        stat_group = stat_info['group']
        stat_point = stat_info['point']
        res = result['stat_rep']
        group = res['group']
        point = res['point']
        for k in data_names:
            group[k] = group[k].at[i].set(stat_group[k])
            point[k] = point[k].at[i].set(stat_point[k])
        res['total'] = res['total'].at[i].set(sum(stat_group.values()))

        # fit simulation data
        res = jaxopt.LevenbergMarquardt(
            residual_fun=new_residual,
            stop_criterion='grad-l2-norm'
        ).run(init[i])
        state = res.state

        # update best fit params to result
        params = to_params_dict(to_dict(res.params))
        for k in result['params_fit']:
            result['params_fit'][k] = \
                result['params_fit'][k].at[i].set(params[k])

        # update best fit model to result
        model = model_counts(to_constr_dict(res.params))
        for k in data_names:
            result['model_fit'][k] = result['model_fit'][k].at[i].set(model[k])

        stat_info = new_deviance_info(res.params)
        stat_group = stat_info['group']
        stat_point = stat_info['point']
        res = result['stat_fit']
        group = res['group']
        point = res['point']
        for k in data_names:
            group[k] = group[k].at[i].set(stat_group[k])
            point[k] = point[k].at[i].set(stat_point[k])

        res['total'] = res['total'].at[i].set(2.0 * state.value)

        valid = jnp.bitwise_not(
            jnp.isnan(state.value)
            | jnp.isnan(state.error)
            | jnp.greater(state.error, 1e-3)
        )
        result['valid'] = result['valid'].at[i].set(valid)

        return sim_data, result, init

    @jax.jit
    def sim_sequence_fit(sim_data, result, init, run_str):
        """Fit simulation data in sequence."""
        raise NotImplementedError

    def sim_parallel_fit(sim_data, result, init, run_str):
        """Fit simulation data in parallel."""
        neval = len(result['valid'])
        ncores = jax.device_count()

        reshape = lambda x: x.reshape((ncores, -1) + x.shape[1:])
        sim_data_ = jax.tree_map(reshape, sim_data)
        result_ = jax.tree_map(reshape, result)
        init_ = jax.tree_map(reshape, init)

        fn = progress_bar_factory(neval, ncores, run_str=run_str)(sim_fit_one)

        fit_results = jax.pmap(
            lambda *args: lax.fori_loop(0, neval//ncores, fn, args)[1]
        )(sim_data_, result_, init_)

        return jax.tree_map(lambda x: jnp.hstack(x), fit_results)

    return HelperFn(
        numpyro_model=numpyro_model,
        to_dict=to_dict,
        to_constr_dict=to_constr_dict,
        to_unconstr_dict=to_unconstr_dict,
        to_unconstr_array=to_unconstr_array,
        to_params_dict=to_params_dict,
        to_params_array=to_params_array,
        deviance_unconstr=jax.jit(deviance_unconstr),
        deviance_unconstr_info=jax.jit(deviance_unconstr_info),
        deviance_unconstr_grad=jax.jit(jax.grad(deviance_unconstr)),
        unconstr_covar=unconstr_covar,
        params_covar=params_covar,
        residual=jax.jit(residual),
        params_by_group=params_by_group,
        net_counts=net_counts,
        model_counts=model_counts,
        sim_residual=residual,
        sim_deviance_unconstr_info=deviance_unconstr_info,
        sim_result_container=sim_result_container,
        sim_fit_one=sim_fit_one,
        sim_sequence_fit=sim_sequence_fit,
        sim_parallel_fit=sim_parallel_fit
    )


def _likelihood_fn(data: Data, stat: str) -> Callable:
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
