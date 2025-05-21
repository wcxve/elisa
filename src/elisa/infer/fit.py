"""Model fit in a maximum likelihood or Bayesian way."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from importlib import metadata
from typing import TYPE_CHECKING

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import xarray as xr
from iminuit import Minuit
from numpyro.infer import init_to_value
from numpyro.infer.barker import BarkerMH, BarkerMHState
from numpyro.infer.ensemble import (
    AIES,
    AIESState,
    EnsembleSamplerState,
    ESSState,
)
from numpyro.infer.hmc import HMC, NUTS, HMCState
from numpyro.infer.mcmc import MCMC, MCMCKernel
from numpyro.infer.sa import SA, SAState

from elisa import __version__ as elisa_version
from elisa.data.base import FixedData, ObservationData
from elisa.infer.helper import Helper, get_helper
from elisa.infer.likelihood import _STATISTIC_OPTIONS
from elisa.infer.results import MLEResult, PosteriorResult
from elisa.infer.samplers.blackjax.nuts import BlackJAXNUTS, BlackJAXNUTSState
from elisa.infer.samplers.ensemble.emcee import EmceeSampler
from elisa.infer.samplers.ensemble.numpyro import (
    NumPyroAIES,
    NumpyroEnsembleSampler,
    NumPyroESS,
)
from elisa.infer.samplers.ensemble.zeus import ZeusSampler
from elisa.infer.samplers.ns.jaxns import JAXNSSampler
from elisa.models.model import Model, get_model_info
from elisa.util.config import get_parallel_number
from elisa.util.misc import (
    add_suffix,
    build_namespace,
    make_pretty_table,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Literal

    from jaxlib.xla_client import Device
    from prettytable import PrettyTable

    from elisa.infer.likelihood import Statistic
    from elisa.models.model import ModelInfo
    from elisa.util.typing import Array, ArrayLike, JAXArray, JAXFloat


class Fit(ABC):
    """Abstract base class for model fitting.

    Parameters
    ----------
    data : Data or sequence of Data
        The observation data.
    model : Model or sequence of Model
        The model used to fit the data.
    stat : {'chi2', 'cstat', 'pstat', 'pgstat', 'wstat'} or sequence, optional
        The likelihood option for the data and model. Available likelihood
        options are:

            * ``'chi2'``: Gaussian data
            * ``'cstat'``: Poisson data
            * ``'pstat'``: Poisson data with known background
            * ``'pgstat'``: Poisson data with Gaussian background
            * ``'wstat'``: Poisson data with Poisson background

        The default is None, which means automatically choosing the suitable
        likelihood options for the datasets and models.
    seed : int, optional
        Seed of random number generator used for fit. The default is 42.
    """

    # TODO:
    #  - fit multiple sources to one dataset (with multiple responses)
    #  - fit data background given response and model

    _lm: Callable[[JAXArray], JAXArray] | None = None
    _ns: JAXNSSampler | None = None

    def __init__(
        self,
        data: ObservationData | Sequence[ObservationData],
        model: Model | Sequence[Model],
        stat: Statistic | Sequence[Statistic] | None = None,
        seed: int = 42,
    ):
        inputs = self._parse_input(data, model, stat)
        data: list[FixedData] = inputs[0]
        models: list[Model] = inputs[1]
        stats: list[Statistic] = inputs[2]

        # if a component is not fit with all datasets,
        # add names of data sets to be fit with it as its name/latex suffix
        data_names = [d.name for d in data]
        data_to_cid = {
            n: m._comps_id for n, m in zip(data_names, models, strict=True)
        }
        cid_to_comp = {c._id: c for m in models for c in m._comps}
        cid = list(cid_to_comp.keys())
        comps = list(cid_to_comp.values())
        cid_to_data_suffix = {
            i: (
                '+'.join(i for i in data_names if i not in names)  # keep order
                if (
                    names := [n for n in data_names if i not in data_to_cid[n]]
                )
                else ''
            )
            for i in cid
        }
        data_suffix = list(cid_to_data_suffix.values())
        cname = [comp.name for comp in comps]
        name_with_data_suffix = list(
            map(''.join, zip(cname, data_suffix, strict=True))
        )
        num_suffix = build_namespace(name_with_data_suffix)['suffix_num']
        cname = add_suffix(cname, num_suffix, True)
        cname = add_suffix(cname, data_suffix, False)
        cid_to_name = dict(zip(cid, cname, strict=True))
        latex = [comp.latex for comp in comps]
        latex = add_suffix(latex, num_suffix, True, latex=True)
        latex = add_suffix(latex, data_suffix, False, latex=True, mathrm=True)
        cid_to_latex = dict(zip(cid, latex, strict=True))

        # get model info
        self._model_info: ModelInfo = get_model_info(
            comps, cid_to_name, cid_to_latex
        )

        # first filter out duplicated models then compile the remaining models,
        # this is intended to avoid re-compilation of the same model
        models_id = [id(m) for m in models]
        mid_to_model = dict(zip(models_id, models, strict=True))
        compiled_model = {
            mid: m.compile(model_info=self._model_info)
            for mid, m in mid_to_model.items()
        }
        data_to_mid = dict(zip(data_names, models_id, strict=True))
        self._model = {
            name: compiled_model[mid] for name, mid in data_to_mid.items()
        }

        # store data, stat, seed
        self._data: dict[str, FixedData] = dict(
            zip(data_names, data, strict=True)
        )
        self._stat: dict[str, Statistic] = dict(
            zip(data_names, stats, strict=True)
        )
        self._seed: int = int(seed)

        # make model information table
        self._make_info_table()

        self.__helper: Helper | None = None

    def _optimize_lm(
        self,
        unconstr_init: JAXArray,
        max_steps: int = 131072,
        throw: bool = True,
        verbose: bool = False,
    ) -> tuple[JAXArray, JAXFloat]:
        """Search MLE by Levenberg-Marquardt algorithm of :mod:`optimistix`."""

        if verbose:
            verbose = frozenset({'step', 'loss'})
        else:
            verbose = frozenset()

        if getattr(self, '_lm_verbose', verbose) != verbose:
            self._lm = None

        if self._lm is None:
            lm_solver = optx.LevenbergMarquardt(
                rtol=0.0, atol=1e-6, verbose=verbose
            )
            residual = jax.jit(lambda x, aux: self._helper.residual(x))

            def lm(init):
                res = optx.least_squares(
                    fn=residual,
                    solver=lm_solver,
                    y0=init,
                    max_steps=max_steps,
                    throw=throw,
                )
                grad_norm = jnp.linalg.norm(res.state.f_info.compute_grad())
                return res.value, grad_norm

            self._lm = jax.jit(lm)

        return self._lm(jnp.asarray(unconstr_init, float))

    def _optimize_ns(self, max_steps=131072, verbose=False) -> JAXArray:
        """Search MLE using nested sampling of :mod:`jaxns`."""
        if self._ns is None:
            self._ns = JAXNSSampler(
                self._helper.numpyro_model,
                constructor_kwargs={
                    'max_samples': max_steps,
                    'parameter_estimation': True,
                    'verbose': verbose,
                },
            )
            t0 = time.time()
            print('Start searching MLE...')
            self._ns.run(rng_key=jax.random.PRNGKey(self._helper.seed['mcmc']))
            print(f'Search completed in {time.time() - t0:.2f} s')

        ns = self._ns

        samples = ns._results.samples
        loglike = [
            samples[f'{i}_loglike'].sum(axis=-1) for i in self._data.keys()
        ]
        mle_idx = np.sum(loglike, axis=0).argmax()
        mle = jax.tree.map(lambda s: s[mle_idx], samples)
        mle = {i: mle[i] for i in self._helper.params_names['free']}
        return self._helper.constr_dic_to_unconstr_arr(mle)

    @property
    def _helper(self) -> Helper:
        if self.__helper is None:
            self.__helper = get_helper(self)

        return self.__helper

    def summary(self, file=None) -> None:
        """Print the summary of fitting setup.

        Parameters
        ----------
        file : file-like
            An object with a ``write(string)`` method. This is passed to
            :py:func:`print`.
        """
        print(repr(self), file=file)

    @property
    @abstractmethod
    def _tab_config(self) -> tuple[str, frozenset[str]]:
        """Model information table's title and excluded table fields."""
        pass

    def __repr__(self) -> str:
        return (
            f'\n{self._tab_config[0]}\n\n'
            f'{self._tab_likelihood.get_string()}\n\n'
            f'{self._tab_params.get_string()}\n'
        )

    def _repr_html_(self) -> str:
        """The repr in Jupyter notebook environment."""
        return (
            f'<details open><summary><b>{self._tab_config[0]}</b></summary>'
            f'<br/>{self._tab_likelihood.get_html_string(format=True)}'
            f'<br/>{self._tab_params.get_html_string(format=True)}'
            '</details>'
        )

    def _make_info_table(self):
        fields = ('Data', 'Model', 'Statistic')
        rows = tuple(
            zip(
                self._data,
                (m.name for m in self._model.values()),
                self._stat.values(),
                strict=True,
            )
        )
        self._tab_likelihood: PrettyTable = make_pretty_table(fields, rows)

        fields = ('No.', 'Component', 'Parameter', 'Value', 'Bound', 'Prior')
        mask = np.isin(fields, tuple(self._tab_config[1]))
        fields = np.array(fields)[~mask].tolist()
        rows = np.array(self._model_info.info)[:, ~mask].tolist()
        self._tab_params: PrettyTable = make_pretty_table(fields, rows)

    @staticmethod
    def _parse_input(
        data: ObservationData | Sequence[ObservationData],
        model: Model | Sequence[Model],
        stat: Statistic | Sequence[Statistic] | None,
    ) -> tuple[list[FixedData], list[Model], list[Statistic]]:
        """Check if data, model, and stat are correct and return lists."""

        # ====================== some helper functions ========================
        def get_list(
            inputs: Any, name: str, expect_type, type_name: str
        ) -> list:
            """Check the model/data/stat, and return a list."""
            if isinstance(inputs, expect_type):
                input_list = [inputs]
            elif isinstance(inputs, Sequence):
                if not inputs:
                    raise ValueError(f'{name} list is empty')
                if not all(isinstance(i, expect_type) for i in inputs):
                    raise ValueError(f'all {name} must be a valid {type_name}')
                input_list = list(inputs)
            else:
                raise ValueError(f'got wrong type {type(inputs)} for {name}')
            return input_list

        def get_stat(d: FixedData) -> Statistic:
            """Get the default stat for the data."""
            # 'pstat' is used only when specified explicitly by user
            if d.spec_poisson:
                if d.has_back:
                    if d.back_poisson:
                        return 'wstat'
                    return 'pgstat'
                else:
                    return 'cstat'
            else:
                return 'chi2'

        def check_stat(d: FixedData, s: Statistic):
            """Check if data type and likelihood are matched."""
            name = d.name
            if not d.spec_poisson and s != 'chi2':
                raise ValueError(
                    f'{name} data has Gaussian uncertainties, '
                    'use Gaussian statistic (chi2) instead'
                )

            if s == 'chi2':
                if np.any(d.net_errors == 0.0):
                    raise ValueError(
                        f'{name} data has zero uncertainties, '
                        'and Gaussian statistic (chi2) will be invalid; '
                        'grouping the data may fix this error'
                    )

            elif s == 'cstat':
                if d.has_back:
                    back = 'Poisson' if d.back_poisson else 'Gaussian'
                    stat1 = 'W' if d.back_poisson else 'PG'
                    stat2 = 'w' if d.back_poisson else 'pg'
                    raise ValueError(
                        f'{name} data has {back} background, '
                        'and using C-statistic (cstat) is invalid; '
                        f'use {stat1}-statistic ({stat2}stat) instead'
                    )

            elif s == 'pstat':
                if not d.has_back:
                    raise ValueError(
                        f'{name} data has no background, '
                        'and using P-statistic (pstat) is invalid; '
                        'use C-statistic (cstat) instead'
                    )

            elif s == 'pgstat':
                if not d.has_back:
                    raise ValueError(
                        f'{name} data has no background, '
                        'and using PG-statistic (pgstat) is invalid; '
                        'use C-statistic (cstat) instead'
                    )

                if np.any(d.back_errors == 0.0):
                    raise ValueError(
                        f'{name} data has zero background uncertainties, '
                        'and PG-statistic (pgstat) will be invalid; '
                        'grouping the data may fix this error'
                    )

            elif s == 'wstat' and not (d.has_back and d.back_poisson):
                if not d.has_back:
                    raise ValueError(
                        f'{name} data has no background, '
                        'and using W-statistic (wstat) is invalid; '
                        'use C-statistic (cstat) instead'
                    )

                if not d.back_poisson:
                    raise ValueError(
                        f'{name} data has Gaussian background, '
                        'and using W-statistic (wstat) is invalid; '
                        'use PG-statistic (pgstat) instead'
                    )

        # ====================== some helper functions ========================

        # get data
        data_list: list[FixedData] = [
            d.get_fixed_data()
            for d in get_list(data, 'data', ObservationData, 'Data')
        ]

        # check if data are used multiple times
        if len(list(map(id, data_list))) != len(data_list):
            count = {d: data_list.count(d) for d in set(data_list)}
            raise ValueError(
                'data cannot be used multiple times: '
                + ', '.join(
                    f'{k.name} ({v})' for k, v in count.items() if v > 1
                )
            )

        # check if data name is unique
        name_list = [d.name for d in data_list]
        if len(set(name_list)) != len(data_list):
            raise ValueError(
                f'data names are not unique: {", ".join(set(name_list))}, '
                "please give a unique name in Data(..., name='NAME'), "
                "or set data.name='NAME'"
            )

        # get model
        model_list: list[Model] = get_list(model, 'model', Model, 'Model')

        # check if the model type is additive
        flag = [i.type == 'add' for i in model_list]
        if not all(flag):
            err = (j for i, j in enumerate(model_list) if not flag[i])
            err = ', '.join(f"'{i}'" for i in err)
            msg = f'got models which are not additive type: {err}'
            raise TypeError(msg)

        # get stat
        stat_list: list[Statistic]
        if stat is None:
            stat_list: list[Statistic] = [get_stat(d) for d in data_list]
        else:
            stat_list: list[Statistic] = get_list(stat, 'stat', str, 'str')

            # check the stat option
            flag = [i in _STATISTIC_OPTIONS for i in stat_list]
            if not all(flag):
                err = ', '.join(
                    f"'{j}'" for i, j in enumerate(stat_list) if not flag[i]
                )
                supported = ', '.join(f"'{i}'" for i in _STATISTIC_OPTIONS)
                msg = f'unexpected stat: {err}; supported are {supported}'
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
        for d, s in zip(data_list, stat_list, strict=True):
            check_stat(d, s)

        return data_list, model_list, stat_list


class MaxLikeFit(Fit):
    _tab_config = ('Maximum Likelihood Fit', frozenset({'Prior'}))

    def _optimize_minuit(
        self,
        unconstr_init: JAXArray,
        ncall: int | None = None,
        throw: bool = True,
        verbose: int | bool = False,
    ) -> Minuit:
        """Search MLE using Minuit algorithm of :mod:`iminuit`."""
        deviance = jax.jit(self._helper.deviance_total)
        deviance.ndata = self._helper.ndata['total']
        minuit = Minuit(
            deviance,
            np.array(unconstr_init),
            grad=jax.jit(jax.grad(deviance)),
            name=self._helper.params_names['free'],
        )

        if throw:
            minuit.throw_nan = True

        minuit.print_level = int(verbose)

        # TODO: test if simplex can be used to "polish" the initial guess
        minuit.strategy = 2
        minuit.migrad(ncall=ncall, iterate=10)

        return minuit

    def mle(
        self,
        init: ArrayLike | dict | None = None,
        method: Literal['minuit', 'lm', 'ns'] = 'minuit',
        max_steps: int = None,
        throw: bool = True,
        verbose: int | bool = False,
    ) -> MLEResult:
        """Search Maximum Likelihood Estimation (MLE) for the model.

        Parameters
        ----------
        init : dict, optional
            Initial guess for the maximum likelihood estimation.
        method : {'minuit', 'lm', 'ns'}, optional
            Optimization algorithm used to find the MLE.
            Available options are:

                * ``'minuit'``: Migrad algorithm of :mod:`iminuit`.
                * ``'lm'``: Levenberg-Marquardt algorithm of :mod:`optimistix`.
                * ``'ns'``: Nested sampling of :mod:`jaxns`. This option first
                  search MLE globally, then polish it with local minimization.

            The default is 'minuit'.

        Other Parameters
        ----------------
        max_steps : int, optional
            The maximum number of steps the solver can take. The default is
            131072.
        throw : bool, optional
            Whether to report any failures of the solver. Defaults to True.
        verbose : int or bool, optional
            Whether to print fit progress information. The default is False.

        Returns
        -------
        MLEResult
            The MLE result.
        """
        if init is None:
            init = self._helper.free_default['constr_dic']
        else:
            init = self._helper.free_default['constr_dic'] | dict(init)
        init_unconstr = self._helper.constr_dic_to_unconstr_arr(init)

        max_steps = 131072 if max_steps is None else int(max_steps)

        if method == 'lm':  # use Levenberg-Marquardt algorithm to find MLE
            init_unconstr, _ = self._optimize_lm(
                init_unconstr, max_steps, throw, bool(verbose)
            )
        elif method == 'ns':  # use nested sampling to find MLE
            init_unconstr = self._optimize_ns(max_steps, verbose)
        else:
            if method != 'minuit':
                raise ValueError(f'unsupported optimization method {method}')

        minuit = self._optimize_minuit(
            init_unconstr, max_steps, throw, verbose
        )

        return MLEResult(minuit, self._helper)


class BayesFit(Fit):
    _tab_config = ('Bayesian Fit', frozenset({'Bound'}))

    def _generate_results(
        self,
        samples: dict[str, Array],
        ess: dict[str, int],
        reff: float,
        lnZ: tuple[float | None, float | None] = (None, None),
        sample_stats: dict[str, Any] | None = None,
        sampler_state: Any | None = None,
        attrs: dict[str, Any] | None = None,
        inference_library: str | None = None,
    ) -> PosteriorResult:
        helper = self._helper
        samples = jax.device_get(samples)
        params = helper.get_params(samples)
        models = helper.get_models(samples)
        posterior = params | models
        posterior_predictive = helper.simulate(helper.seed['pred'], models, 1)
        loglike = helper.get_loglike(samples)
        group = {f'{k}_total': v for k, v in loglike['group'].items()}
        loglike = (
            loglike['data']
            | loglike['point']
            | group
            | {'channels': loglike['channels']}
            | {'total': loglike['total']}
        )

        # get observation counts data
        obs_data = helper.obs_data

        # coords and dims of arviz.InferenceData
        coords = dict(helper.channels)

        dims = {'channels': ['channel']}
        for i in helper.data_names:
            dim = [f'{i}_channel']
            dims[i] = dims[f'{i}_Non'] = dims[f'{i}_Non_model'] = dim

            if f'{i}_Noff' in obs_data:
                dims[f'{i}_Noff'] = dims[f'{i}_Noff_model'] = dim

        # additional attrs for each group of arviz.InferenceData
        if attrs is None:
            attrs = {}
        else:
            attrs = dict(attrs)
        attrs['elisa_version'] = elisa_version
        attrs |= {
            'inference_library': inference_library,
            'inference_library_version': metadata.version(inference_library),
        }

        # create InferenceData
        idata = az.from_dict(
            posterior=posterior,
            posterior_predictive=posterior_predictive,
            sample_stats=sample_stats,
            log_likelihood=loglike,
            observed_data=obs_data,
            coords=coords,
            dims=dims,
            posterior_attrs=attrs,
            posterior_predictive_attrs=attrs,
            sample_stats_attrs=attrs,
            log_likelihood_attrs=attrs,
            observed_data_attrs=attrs,
        )
        # add extra statistics to idata
        ess = ess | {'reff': reff}
        evidence = {
            'lnZ': lnZ[0] if lnZ[0] is not None else np.nan,
            'lnZ_error': lnZ[1] if lnZ[1] is not None else np.nan,
        }
        ess = xr.Dataset(ess, attrs=attrs)
        evidence = xr.Dataset(evidence, attrs=attrs)
        idata.add_groups(
            group_dict={'ess': ess, 'evidence': evidence},
            warn_on_custom_groups=False,
        )

        return PosteriorResult(
            helper=self._helper,
            idata=idata,
            ml_optimize=self._optimize_lm,
            sampler_state=sampler_state,
        )

    def _check_init(self, init: dict[str, float] | None) -> dict[str, float]:
        if init is None:
            init = self._helper.free_default['constr_dic']
        else:
            init = self._helper.free_default['constr_dic'] | dict(init)
        return init

    @staticmethod
    def _set_numpyro_mcmc_post_warmup_state(mcmc: MCMC, state: Any) -> None:
        if state is None:
            return

        assert isinstance(mcmc, MCMC)
        kernel: MCMCKernel = mcmc.sampler
        kernel_state_types = {
            BarkerMH: BarkerMHState,
            BlackJAXNUTS: BlackJAXNUTSState,
            NumpyroEnsembleSampler: EnsembleSamplerState,
            HMC: HMCState,
            SA: SAState,
            NUTS: HMCState,
        }
        ensemble_state_types = {
            NumPyroAIES: AIESState,
            NumPyroESS: ESSState,
        }
        for kt, st in kernel_state_types.items():
            if isinstance(kernel, kt):
                if not isinstance(state, st):
                    raise ValueError(
                        f'post_warmup_state must be {st.__name__}'
                    )
                break
        if isinstance(kernel, NumpyroEnsembleSampler):
            kernel_type = kernel.__class__
            is_type = ensemble_state_types.get(kernel_type, object)
            if not isinstance(state.inner_state, is_type):
                raise ValueError(
                    f'post_warmup_state must be state for {kernel_type}'
                )
        mcmc.post_warmup_state = state

    def _get_ess(
        self,
        samples: dict[str, Array],
        chains: int,
    ) -> tuple[dict[str, int], float]:
        helper = self._helper
        params_names = helper.params_names

        # effective sample size
        params = helper.get_params(samples)
        ess = az.ess(params)
        ess = {k: int(ess[k].values) for k in params.keys()}

        # relative mcmc efficiency
        # the calculation of reff is according to arviz loo:
        # https://github.com/arviz-devs/arviz/blob/1b0b9cb050e3b757e1551d3a1f7a8f8e2773bc36/arviz/stats/stats.py#L776
        if chains == 1:
            reff = 1.0
        else:
            # use only free parameters to calculate reff
            free = {k: params[k] for k in params_names['free']}
            reff_p = az.ess(free, method='mean', relative=True)
            reff = np.hstack(list(reff_p.data_vars.values())).mean()

        return ess, reff

    def _generate_result_from_numpyro(
        self,
        sampler: MCMC,
        kernel_library: str = 'numpyro',
    ) -> PosteriorResult:
        samples = sampler.get_samples(group_by_chain=True)

        if isinstance(sampler.sampler, NumpyroEnsembleSampler):
            samples = jax.tree.map(lambda x: jnp.swapaxes(x, 1, 2), samples)
            samples = jax.tree.map(
                lambda x: jnp.reshape(
                    x,
                    (x.shape[0], x.shape[1] * x.shape[2], *x.shape[3:]),
                ),
                samples,
            )

        # sample stats
        rename = {'num_steps': 'n_steps'}
        sample_stats = {}
        for k, v in sampler.get_extra_fields(group_by_chain=True).items():
            name = rename.get(k, k)
            value = jax.device_get(v).copy()
            sample_stats[name] = value

        if 'tree_depth' not in sample_stats and 'num_steps' in sample_stats:
            num_steps = sample_stats['num_steps']
            sample_stats['tree_depth'] = np.log2(num_steps).astype(int) + 1

        ess, reff = self._get_ess(samples, sampler.num_chains)

        return self._generate_results(
            samples=samples,
            ess=ess,
            reff=reff,
            sample_stats=sample_stats,
            sampler_state=sampler.last_state,
            inference_library=kernel_library,
        )

    def _run_numpyro_mcmc(
        self,
        kernel: type[MCMCKernel],
        warmup: int,
        steps: int,
        chains: int | None = None,
        thinning: int = 1,
        init: dict[str, float] | None = None,
        chain_method: str = 'parallel',
        progress: bool = True,
        post_warmup_state: Any = None,
        extra_fields: tuple[str, ...] = (),
        kernel_library: str = 'numpyro',
        **kernel_kwargs: dict,
    ):
        """Run the regular sampler of numpyro."""
        if not issubclass(kernel, MCMCKernel):
            raise ValueError('kernel must be a subclass of numpyro MCMCKernel')

        warmup = int(warmup)
        steps = int(steps)
        thinning = int(thinning)

        device_count = jax.local_device_count()

        chains = int(chains) if chains is not None else device_count

        kernel_kwargs['model'] = self._helper.numpyro_model

        rng_key = jax.random.PRNGKey(self._helper.seed['mcmc'])

        # TODO: option to let sampler starting from MLE
        if issubclass(kernel, NumpyroEnsembleSampler):
            # set randomize_split to True to improve mixing
            kernel_kwargs.setdefault('randomize_split', True)
            init = self._check_init(init)
            init = self._helper.constr_dic_to_unconstr_arr(init)
            n_params = len(init)
            default_walkers = 4 * n_params
            walkers = kernel_kwargs.pop('walkers', default_walkers)
            if walkers is None:
                walkers = default_walkers
            else:
                walkers = int(walkers)
            kernel_kwargs['walkers'] = walkers
            jitter = 0.1 * jnp.abs(init)
            low = init - jitter
            high = init + jitter
            rng_key, init_key = jax.random.split(rng_key, 2)
            init = jax.random.uniform(
                init_key,
                shape=(n_params, chains, walkers),
                minval=low[:, None, None],
                maxval=high[:, None, None],
            )
            if chains == 1:  # remove the chains dim if run a single sampler
                init = jnp.squeeze(init, axis=1)
            init = dict(
                zip(self._helper.params_names['free'], init, strict=True)
            )
        else:
            init_strategy = init_to_value(values=self._check_init(init))
            if init is not None:
                kernel_kwargs['init_strategy'] = init_strategy
                init = None
            else:
                kernel_kwargs.setdefault('init_strategy', init_strategy)

        sampler = MCMC(
            kernel(**kernel_kwargs),
            num_warmup=warmup,
            num_samples=steps * thinning,
            num_chains=chains,
            thinning=thinning,
            chain_method=chain_method,
            progress_bar=progress,
        )
        self._set_numpyro_mcmc_post_warmup_state(sampler, post_warmup_state)
        sampler.run(rng_key, extra_fields=extra_fields, init_params=init)
        return self._generate_result_from_numpyro(
            sampler=sampler,
            kernel_library=kernel_library,
        )

    def nuts(
        self,
        warmup: int = 2000,
        steps: int = 5000,
        chains: int | None = None,
        thinning: int = 1,
        init: dict[str, float] | None = None,
        chain_method: str = 'parallel',
        progress: bool = True,
        post_warmup_state: HMCState | None = None,
        **kwargs: dict,
    ) -> PosteriorResult:
        """Run :mod:`numpyro`'s implementation of No-U-Turn Sampler (NUTS).

        .. note::
            If the chains are not converged well, see ref [2]_ for more
            information on how to fine-tune NUTS.

        Parameters
        ----------
        warmup : int, optional
            Number of warmup steps. The default is 2000.
        steps : int, optional
            Number of steps to run for each chain. The default is 5000.
        chains : int, optional
            Number of MCMC chains to run. If there are not enough devices
            available, chains will run in sequence. Defaults to the number of
            ``jax.local_device_count()``.
        thinning: int, optional
            For each chain, every `thinning` step is retained, and the other
            steps are discarded. The total steps for each chain are
            `steps` * `thinning`. The default is 1.
        init : dict, optional
            Initial parameter for sampler to start from.
        chain_method : str, optional
            The chain method passed to :class:`numpyro.infer.MCMC`.
        progress : bool, optional
            Whether to show progress bars during sampling. The default is True.
        post_warmup_state : HMCState, optional
            The state before the sampling phase. The sampling will start from
            the given state if provided.
        **kwargs : dict
            Extra parameters passed to :class:`numpyro.infer.NUTS`.
            The default for `dense_mass` is ``True``.

        Returns
        -------
        PosteriorResult
            The posterior sampling result.

        References
        ----------
        .. [1] The No-U-Turn Sampler: Adaptively Setting Path Lengths in
               Hamiltonian Monte Carlo
               (https://www.jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf)
        .. [2] NumPyro tutorial: `Bad posterior geometry and how to deal with
               it <https://num.pyro.ai/en/stable/tutorials/bad_posterior_geometry.html>`__
        """
        kwargs.setdefault('dense_mass', True)
        return self._run_numpyro_mcmc(
            kernel=NUTS,
            warmup=warmup,
            steps=steps,
            chains=chains,
            thinning=thinning,
            init=init,
            chain_method=chain_method,
            progress=progress,
            post_warmup_state=post_warmup_state,
            extra_fields=('energy', 'num_steps'),
            **kwargs,
        )

    def barkermh(
        self,
        warmup: int = 5000,
        steps: int = 5000,
        chains: int | None = None,
        thinning: int = 1,
        init: dict[str, float] | None = None,
        chain_method: str = 'parallel',
        progress: bool = True,
        post_warmup_state: BarkerMHState | None = None,
        **kwargs: dict,
    ) -> PosteriorResult:
        """Run :mod:`numpyro`'s implementation of ``BarkerMH`` sampler.

        .. note::
            This is a gradient-based MCMC algorithm of Metropolis-Hastings
            type that uses a skew-symmetric proposal distribution that depends
            on the gradient of the potential (the Barker proposal [1]_).
            In particular the proposal distribution is skewed in the direction
            of the gradient at the current sample. This algorithm is expected
            to be particularly effective for low to moderate dimensional
            models, where it may be competitive with HMC and NUTS.

        Parameters
        ----------
        warmup : int, optional
            Number of warmup steps. The default is 5000.
        steps : int, optional
            Number of steps to run for each chain. The default is 10000.
        chains : int, optional
            Number of MCMC chains to run. If there are not enough devices
            available, chains will run in sequence. Defaults to the number of
            ``jax.local_device_count()``.
        thinning: int, optional
            For each chain, every `thinning` step is retained, and the other
            steps are discarded. The total steps for each chain are
            `steps` * `thinning`. The default is 1.
        init : dict, optional
            Initial parameter for sampler to start from.
        chain_method : str, optional
            The chain method passed to :class:`numpyro.infer.MCMC`.
        progress : bool, optional
            Whether to show progress bars during sampling. The default is True.
        post_warmup_state : BarkerMHState, optional
            The state before the sampling phase. The sampling will start from
            the given state if provided.
        **kwargs : dict
            Extra parameters passed to :class:`numpyro.infer.BarkerMH`.
            The default for `dense_mass` is ``True``.

        Returns
        -------
        PosteriorResult
            The posterior sampling result.

        References
        ----------
        .. [1] The Barker proposal: combining robustness and efficiency in
               gradient-based MCMC (https://doi.org/10.1111/rssb.12482),
               Samuel Livingstone and Giacomo Zanella.
        """
        kwargs.setdefault('dense_mass', True)
        return self._run_numpyro_mcmc(
            kernel=BarkerMH,
            warmup=warmup,
            steps=steps,
            chains=chains,
            thinning=thinning,
            init=init,
            chain_method=chain_method,
            progress=progress,
            post_warmup_state=post_warmup_state,
            **kwargs,
        )

    def sa(
        self,
        warmup: int = 70000,
        steps: int = 5000,
        chains: int | None = None,
        thinning: int = 2,
        init: dict[str, float] | None = None,
        chain_method: str = 'parallel',
        progress: bool = True,
        post_warmup_state: SAState | None = None,
        **kwargs: dict,
    ) -> PosteriorResult:
        """Run :mod:`numpyro`'s implementation of Sample Adaptive (SA) MCMC.

        .. note::
            This is a gradient-free sampler. It is fast in terms of n_eff / s,
            but requires **many** warmup (burn-in) steps.

            If the result does not converge satisfactorily, consider increasing
            the values of `warmup` and/or `adapt_state_size`, or providing
            better initial parameter estimates via the `init` argument.

        Parameters
        ----------
        warmup : int, optional
            Number of warmup steps. The default is 70000.
        steps : int, optional
            Number of steps to run. The default is 5000.
        chains : int, optional
            Number of MCMC chains to run. If there are not enough devices
            available, chains will run in sequence. Defaults to the number of
            ``jax.local_device_count()``.
        thinning: int, optional
            For each chain, every `thinning` step is retained, and the other
            steps are discarded. The total steps for each chain are
            `steps` * `thinning`. The default is 2.
        init : dict, optional
            Initial parameter for sampler to start from.
        chain_method : str, optional
            The chain method passed to :class:`numpyro.infer.MCMC`.
        progress : bool, optional
            Whether to show progress bars during sampling. The default is True.
        post_warmup_state : SAState, optional
            The state before the sampling phase. The sampling will start from
            the given state if provided.
        **kwargs : dict
            Extra parameters passed to :class:`numpyro.infer.SA`.
            The default for `adapt_state_size` is ``5 * D``, where `D` is the
            dimension of model parameters.

        Returns
        -------
        PosteriorResult
            The posterior sampling result.

        References
        ----------
        .. [1] Sample Adaptive MCMC
               (https://papers.nips.cc/paper/9107-sample-adaptive-mcmc),
               Michael Zhu
        """
        nparams = len(self._helper.params_names['free'])
        kwargs.setdefault('adapt_state_size', 5 * nparams)
        return self._run_numpyro_mcmc(
            kernel=SA,
            warmup=warmup,
            steps=steps,
            chains=chains,
            thinning=thinning,
            init=init,
            chain_method=chain_method,
            progress=progress,
            post_warmup_state=post_warmup_state,
            **kwargs,
        )

    def blackjax_nuts(
        self,
        warmup: int = 2000,
        steps: int = 5000,
        chains: int | None = None,
        thinning: int = 1,
        init: dict[str, float] | None = None,
        chain_method: str = 'parallel',
        progress: bool = True,
        post_warmup_state: HMCState | None = None,
        **kwargs: dict,
    ) -> PosteriorResult:
        """Run :mod:`blackjax`'s implementation of No-U-Turn Sampler (NUTS).

        .. note::
            If the chains are not converged well, see ref [2]_ for more
            information on how to fine-tune NUTS.

        Parameters
        ----------
        warmup : int, optional
            Number of warmup steps. The default is 2000.
        steps : int, optional
            Number of steps to run for each chain. The default is 5000.
        chains : int, optional
            Number of MCMC chains to run. If there are not enough devices
            available, chains will run in sequence. Defaults to the number of
            ``jax.local_device_count()``.
        thinning: int, optional
            For each chain, every `thinning` step is retained, and the other
            steps are discarded. The total steps for each chain are
            `steps` * `thinning`. The default is 1.
        init : dict, optional
            Initial parameter for sampler to start from.
        chain_method : str, optional
            The chain method passed to :class:`numpyro.infer.MCMC`.
        progress : bool, optional
            Whether to show progress bars during sampling. The default is True.
        post_warmup_state : HMCState, optional
            The state before the sampling phase. The sampling will start from
            the given state if provided.
        **kwargs : dict
            Extra parameters passed to :class:`BlackJAXNUTS`.
            The default for `dense_mass` is ``True``.

        Returns
        -------
        PosteriorResult
            The posterior sampling result.

        References
        ----------
        .. [1] The No-U-Turn Sampler: Adaptively Setting Path Lengths in
               Hamiltonian Monte Carlo
               (https://www.jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf)
        .. [2] NumPyro tutorial: `Bad posterior geometry and how to deal with
               it <https://num.pyro.ai/en/stable/tutorials/bad_posterior_geometry.html>`__
        """
        return self._run_numpyro_mcmc(
            kernel=BlackJAXNUTS,
            warmup=warmup,
            steps=steps,
            chains=chains,
            thinning=thinning,
            init=init,
            chain_method=chain_method,
            progress=progress,
            post_warmup_state=post_warmup_state,
            extra_fields=('energy', 'num_steps'),
            kernel_library='blackjax',
            **kwargs,
        )

    def aies(
        self,
        warmup: int = 5000,
        steps: int = 5000,
        chains: int | None = None,
        thinning: int = 1,
        init: dict[str, float] | None = None,
        chain_method: str = 'parallel',
        n_parallel: int | None = None,
        progress: bool = True,
        post_warmup_state: EnsembleSamplerState | None = None,
        **kwargs: dict,
    ) -> PosteriorResult:
        """Run :mod:`numpyro`'s Affine-Invariant Ensemble Sampling (AIES).

        Affine-invariant ensemble sampling [1]_ is a gradient-free method
        that informs Metropolis-Hastings proposals by sharing information
        between chains. Suitable for low to moderate dimensional models.
        Generally, `chains` should be at least twice the dimensionality
        of the model.

        .. note::
            This sampler must be used with even number `chains` > 1.

        Parameters
        ----------
        warmup : int, optional
            Number of warmup steps. The default is 5000.
        steps : int, optional
            Number of steps to run for each chain. The default is 5000.
        chains : int, optional
            Number of MCMC chains to run. Defaults to 4 * `D`, where `D` is
            the dimension of model parameters.
        thinning: int, optional
            For each chain, every `thinning` step is retained, and the other
            steps are discarded. The total steps for each chain are
            `steps` * `thinning`. The default is 1.
        init : dict, optional
            Initial parameter for sampler to start from.
        chain_method : str, optional
            The chain method passed to :class:`numpyro.infer.MCMC`.
        n_parallel : int, optional
            Number of parallel samplers to run.
            The default is ``jax.local_device_count()``.
        progress : bool, optional
            Whether to show progress bars during sampling. The default is True.
        post_warmup_state : EnsembleSamplerState, optional
            The state before the sampling phase. The sampling will start from
            the given state if provided. This does not take effect when
            `n_parallel`>=2.
        **kwargs : dict
            Extra parameters passed to :class:`numpyro.infer.AIES`.
            The default for `moves` is ``{AIES.StretchMove(): 1.0}``.

        Returns
        -------
        PosteriorResult
            The posterior sampling result.

        References
        ----------
        .. [1] *emcee: The MCMC Hammer*
               (https://iopscience.iop.org/article/10.1086/670067),
               Daniel Foreman-Mackey, David W. Hogg, Dustin Lang,
               and Jonathan Goodman.
        """
        kwargs['walkers'] = chains
        # use the same default moves as in emcee
        kwargs.setdefault('moves', {AIES.StretchMove(): 1.0})
        return self._run_numpyro_mcmc(
            kernel=NumPyroAIES,
            warmup=warmup,
            steps=steps,
            chains=n_parallel,
            thinning=thinning,
            init=init,
            chain_method=chain_method,
            progress=progress,
            post_warmup_state=post_warmup_state,
            **kwargs,
        )

    def ess(
        self,
        warmup: int = 5000,
        steps: int = 5000,
        chains: int | None = None,
        thinning: int = 1,
        init: dict[str, float] | None = None,
        chain_method: str = 'parallel',
        n_parallel: int | None = None,
        progress: bool = True,
        post_warmup_state: EnsembleSamplerState | None = None,
        **kwargs: dict,
    ) -> PosteriorResult:
        """Run :mod:`numpyro`'s Ensemble Slice Sampling (ESS).

        Ensemble slice sampling [1]_ is a gradient free method
        that finds better slice sampling directions by sharing information
        between chains. Suitable for low to moderate dimensional models.
        Generally, `chains` should be at least twice the dimensionality
        of the model.

        .. note::
            This sampler must be used with even number `chains` > 1.

        Parameters
        ----------
        warmup : int, optional
            Number of warmup steps. The default is 5000.
        steps : int, optional
            Number of steps to run for each chain. The default is 5000.
        chains : int, optional
            Number of MCMC chains to run. Defaults to 4 * `D`, where `D` is
            the dimension of model parameters.
        thinning: int, optional
            For each chain, every `thinning` step is retained, and the other
            steps are discarded. The total steps for each chain are
            `steps` * `thinning`. The default is 1.
        init : dict, optional
            Initial parameter for sampler to start from.
        chain_method : str, optional
            The chain method passed to :class:`numpyro.infer.MCMC`.
        n_parallel : int, optional
            Number of parallel samplers to run.
            The default is ``jax.local_device_count()``.
        progress : bool, optional
            Whether to show progress bars during sampling. The default is True.
        post_warmup_state : EnsembleSamplerState, optional
            The state before the sampling phase. The sampling will start from
            the given state if provided. This does not take effect when
            `n_parallel`>=2.
        **kwargs : dict
            Extra parameters passed to :class:`numpyro.infer.ESS`.

        Returns
        -------
        PosteriorResult
            The posterior sampling result.

        References
        ----------
        .. [1] zeus: a PYTHON implementation of ensemble slice sampling
                for efficient Bayesian parameter inference
                (https://academic.oup.com/mnras/article/508/3/3589/6381726),
                Minas Karamanis, Florian Beutler, and John A. Peacock.
        .. [2] Ensemble slice sampling
               (https://link.springer.com/article/10.1007/s11222-021-10038-2),
               Minas Karamanis, Florian Beutler.
        """
        kwargs['walkers'] = chains
        return self._run_numpyro_mcmc(
            kernel=NumPyroESS,
            warmup=warmup,
            steps=steps,
            chains=n_parallel,
            thinning=thinning,
            init=init,
            chain_method=chain_method,
            progress=progress,
            post_warmup_state=post_warmup_state,
            **kwargs,
        )

    def emcee(
        self,
        warmup: int = 5000,
        steps: int = 5000,
        chains: int | None = None,
        thinning: int = 1,
        init: dict[str, float] | None = None,
        n_parallel: int | None = None,
        progress: bool = True,
        post_warmup_state: Sequence | None = None,
        tune: bool = False,
        ignore_nan: bool = False,
        warmup_kwargs: dict | None = None,
        sampling_kwargs: dict | None = None,
    ) -> PosteriorResult:
        """Run :mod:`emcee`'s affine-invariant ensemble sampling.

        Affine-invariant ensemble sampling [1]_ is a gradient-free method
        that informs Metropolis-Hastings proposals by sharing information
        between chains. Suitable for low to moderate dimensional models.
        Generally, `chains` should be at least twice the dimensionality
        of the model.

        .. note::
            This sampler must be used with even `chains` > 1.

        Parameters
        ----------
        warmup : int, optional
            Number of warmup steps. The default is 5000.
        steps : int, optional
            Number of steps to run for each chain. The default is 5000.
        chains : int, optional
            Number of MCMC chains to run. Defaults to 4 * `D`, where `D` is
            the dimension of model parameters.
        thinning: int, optional
            For each chain, every `thinning` step is retained, and the other
            steps are discarded. The total steps for each chain are
            `steps` * `thinning`. The default is 1.
        init : dict, optional
            Initial parameter for sampler to start from.
        n_parallel : int, optional
            Number of parallel samplers to run.
            The default is ``jax.local_device_count()``.
        progress : bool, optional
            Whether to show progress bars during sampling. The default is True.
        post_warmup_state : sequence, optional
            The state before the sampling phase. The sampling will start from
            the given state if provided.
        tune : bool, optional
            If True, the parameters of some moves will be automatically tuned.
        ignore_nan : bool, optional
            Whether to transform a NaN log probability to a large negative
            number (-1e300). The default is False.

            .. warning::
                Setting ``ignore_nan=True`` may fail to spot potential issues
                with model computation.
        warmup_kwargs: dict, optional
            Extra parameters passed to :class:`emcee.EnsembleSampler` for
            warm-up phase.
        sampling_kwargs: dict | None = None,
            Extra parameters passed to :class:`emcee.EnsembleSampler` for
            sampling phase.

        Returns
        -------
        PosteriorResult
            The posterior sampling result.

        References
        ----------
        .. [1] *emcee: The MCMC Hammer*
               (https://iopscience.iop.org/article/10.1086/670067),
               Daniel Foreman-Mackey, David W. Hogg, Dustin Lang,
               and Jonathan Goodman.
        """
        init = self._check_init(init)
        n_parallel = get_parallel_number(n_parallel)
        sampler = EmceeSampler(
            numpyro_model=self._helper.numpyro_model,
            init_params=init,
            ignore_nan=ignore_nan,
            seed=self._helper.seed['mcmc'],
        )
        samples, states = sampler.run(
            warmup=warmup,
            steps=steps,
            chains=chains,
            thinning=thinning,
            n_parallel=n_parallel,
            tune=tune,
            progress=progress,
            states=post_warmup_state,
            warmup_kwargs=warmup_kwargs,
            sampling_kwargs=sampling_kwargs,
        )
        ess, reff = self._get_ess(samples, n_parallel)
        return self._generate_results(
            samples=samples,
            ess=ess,
            reff=reff,
            sampler_state=states,
            inference_library='emcee',
        )

    def zeus(
        self,
        warmup: int = 3000,
        steps: int = 5000,
        chains: int | None = None,
        thinning: int = 1,
        init: dict[str, float] | None = None,
        n_parallel: int | None = None,
        progress: bool = True,
        post_warmup_state: Sequence | None = None,
        tune: bool = True,
        ignore_nan: bool = False,
        warmup_kwargs: dict | None = None,
        sampling_kwargs: dict | None = None,
    ) -> PosteriorResult:
        """Run :mod:`zeus`' ensemble slice sampling.

        Ensemble slice sampling [1]_ is a gradient free method
        that finds better slice sampling directions by sharing information
        between chains. Suitable for low to moderate dimensional models.
        Generally, `chains` should be at least twice the dimensionality
        of the model.

        .. note::
            This sampler must be used with even number `chains` > 1.

        Parameters
        ----------
        warmup : int, optional
            Number of warmup steps. The default is 5000.
        steps : int, optional
            Number of steps to run for each chain. The default is 5000.
        chains : int, optional
            Number of MCMC chains to run. Defaults to 4 * `D`, where `D` is
            the dimension of model parameters.
        thinning: int, optional
            For each chain, every `thinning` step is retained, and the other
            steps are discarded. The total steps for each chain are
            `steps` * `thinning`. The default is 1.
        init : dict, optional
            Initial parameter for sampler to start from.
        n_parallel : int, optional
            Number of parallel samplers to run.
            The default is ``jax.local_device_count()``.
        progress : bool, optional
            Whether to show progress bars during sampling. The default is True.
        post_warmup_state : sequence, optional
            The state before the sampling phase. The sampling will start from
            the given state if provided.
        tune : bool, optional
            If True, the parameters of some moves will be automatically tuned.
        ignore_nan : bool, optional
            Whether to transform a NaN log probability to a large negative
            number (-1e300). The default is False.

            .. warning::
                Setting ``ignore_nan=True`` may fail to spot potential issues
                with model computation.
        warmup_kwargs: dict, optional
            Extra parameters passed to :class:`zeus.EnsembleSampler` for
            warm-up phase.
        sampling_kwargs: dict | None = None,
            Extra parameters passed to :class:`zeus.EnsembleSampler` for
            sampling phase.

        Returns
        -------
        PosteriorResult
            The posterior sampling result.

        References
        ----------
        .. [1] zeus: a PYTHON implementation of ensemble slice sampling
                for efficient Bayesian parameter inference
                (https://academic.oup.com/mnras/article/508/3/3589/6381726),
                Minas Karamanis, Florian Beutler, and John A. Peacock.
        .. [2] Ensemble slice sampling
               (https://link.springer.com/article/10.1007/s11222-021-10038-2),
               Minas Karamanis, Florian Beutler.
        """
        init = self._check_init(init)
        n_parallel = get_parallel_number(n_parallel)
        sampler = ZeusSampler(
            numpyro_model=self._helper.numpyro_model,
            init_params=init,
            ignore_nan=ignore_nan,
            seed=self._helper.seed['mcmc'],
        )
        samples, states = sampler.run(
            warmup=warmup,
            steps=steps,
            chains=chains,
            thinning=thinning,
            n_parallel=n_parallel,
            tune=tune,
            progress=progress,
            states=post_warmup_state,
            warmup_kwargs=warmup_kwargs,
            sampling_kwargs=sampling_kwargs,
        )
        ess, reff = self._get_ess(samples, n_parallel)
        return self._generate_results(
            samples=samples,
            ess=ess,
            reff=reff,
            sampler_state=states,
            inference_library='zeus-mcmc',
        )

    def jaxns(
        self,
        max_samples: int = 2**17,
        num_live_points: int | None = None,
        s: int | None = None,
        k: int | None = None,
        c: int | None = None,
        devices: list[Device] | None = None,
        difficult_model: bool = False,
        parameter_estimation: bool = False,
        verbose: bool = False,
        term_cond: dict | None = None,
        **kwargs: dict,
    ) -> PosteriorResult:
        """Run :mod:`jaxns`'s implementation of nested sampling.

        .. note::
            Parameters `s`, `k`, and `c` are defined in the paper [1]_.
            For more information of the sampler parameters, see ref [1]_ [2]_.

        Parameters
        ----------
        max_samples : int, optional
            Maximum number of posterior samples. The default is 131072.
        num_live_points : int, optional
            Approximate number of live points. The default is `c` * (`k` + 1).
        s : int, optional
            Number of slices per dimension. The default is 5.
        k : int, optional
            Number of phantom samples. The default is 0.
        c : int, optional
            Number of parallel Markov chains. The default is 30 * `D`, where
            `D` is the dimension of model parameters. It takes effect only
            for num_live_points=None.
        devices : list, optional
            Devices to use. Defaults to all available devices.
        difficult_model : bool, optional
            If True, uses more robust default settings (`s` = 10 and
            `c` = 50 * `D`). It takes effect only for `num_live_points` = None,
            `s` = None or `c` = None. Defaults to False.
        parameter_estimation : bool, optional
            If True, uses more robust default settings for parameter estimation
            (`k` = `D`). It takes effect only for `k` = None.
            Defaults to False.
        verbose : bool, optional
            Print progress information. The default is False.
        term_cond : dict, optional
            Termination conditions for the sampling. The default is as in
            :class:`jaxns.TermCondition`.
        **kwargs : dict
            Extra parameters passes to :class:`jaxns.DefaultNestedSampler`.

        Returns
        -------
        PosteriorResult
            The posterior sampling result.

        References
        ----------
        .. [1] `Phantom-Powered Nested Sampling <https://arxiv.org/abs/2312.11330>`__
        .. [2] `JAXNS API doc <https://jaxns.readthedocs.io/en/latest/api/jaxns/index.html#jaxns.DefaultNestedSampler>`__
        """
        constructor_kwargs = {
            'max_samples': max_samples,
            'num_live_points': num_live_points,
            's': s,
            'k': k,
            'c': c,
            'devices': devices,
            'difficult_model': difficult_model,
            'parameter_estimation': parameter_estimation,
            'verbose': verbose,
        }
        constructor_kwargs.update(kwargs)

        termination_kwargs = {'dlogZ': 1e-4}
        if term_cond is not None:
            termination_kwargs.update(term_cond)

        sampler = JAXNSSampler(
            self._helper.numpyro_model,
            constructor_kwargs=constructor_kwargs,
            termination_kwargs=termination_kwargs,
        )

        print('Running nested sampling of JAXNS...')
        t0 = time.time()
        sampler.run(rng_key=jax.random.PRNGKey(self._helper.seed['mcmc']))
        print(f'Sampling completed in {time.time() - t0:.2f} s')

        helper = self._helper
        result = sampler._results

        # get posterior samples
        total = result.total_num_samples
        rng_key = jax.random.PRNGKey(helper.seed['mcmc'])
        samples = jax.tree.map(
            lambda x: x[None, ...],
            sampler.get_samples(rng_key, total),
        )
        # effective sample size
        overall_ess = int(result.ESS)
        ess = dict.fromkeys(self._helper.params_names['all'], overall_ess)
        # relative mcmc efficiency
        reff = float(overall_ess / result.total_num_samples)
        # model evidence
        lnZ = (float(result.log_Z_mean), float(result.log_Z_uncert))
        return self._generate_results(
            samples=samples,
            ess=ess,
            reff=reff,
            lnZ=lnZ,
            inference_library='jaxns',
        )

    def nautilus(
        self,
        ess: int = 3000,
        ignore_nan: bool = False,
        *,
        constructor_kwargs: dict | None = None,
        termination_kwargs: dict | None = None,
    ) -> PosteriorResult:
        """Run :mod:`nautilus`'s implementation of nested sampling.

        Parameters
        ----------
        ess : int, optional
            The desired effective sample size.
        ignore_nan : bool, optional
            Whether to transform a NaN log probability to a large negative
            number (-1e300). The default is False.

            .. warning::
                Setting ``ignore_nan=True`` may fail to spot potential issues
                with model computation.
        constructor_kwargs : dict, optional
            Extra parameters passed to
            :class:`nautilus.Sampler`.
        termination_kwargs : dict, optional
            Extra parameters passed to
            :class:`nautilus.Sampler.run()`.
        """
        from elisa.infer.samplers.ns.nautilus import NautilusSampler

        if constructor_kwargs is None:
            constructor_kwargs = {}
        else:
            constructor_kwargs = dict(constructor_kwargs)
        constructor_kwargs.setdefault('pool', get_parallel_number(None))

        if termination_kwargs is None:
            termination_kwargs = {}
        else:
            termination_kwargs = dict(termination_kwargs)
        termination_kwargs['n_eff'] = int(ess)

        print('Running nested sampling of Nautilus...')
        sampler = NautilusSampler(
            numpyro_model=self._helper.numpyro_model,
            seed=self._helper.seed['mcmc'],
            ignore_nan=ignore_nan,
            **constructor_kwargs,
        )
        t0 = time.time()
        samples = sampler.run(**termination_kwargs)
        print(f'Sampling completed in {time.time() - t0:.2f} s')

        # format posterior samples
        samples = jax.tree.map(lambda x: x[None], samples)

        # effective sample size
        ess_overall = sampler.ess
        ess = dict.fromkeys(self._helper.params_names['all'], ess_overall)
        # relative mcmc efficiency
        total_sample = len(samples[self._helper.params_names['all'][0]])
        reff = float(ess_overall / total_sample)
        # model evidence
        lnZ = (sampler.lnZ, None)
        return self._generate_results(
            samples=samples,
            ess=ess,
            reff=reff,
            lnZ=lnZ,
            inference_library='nautilus-sampler',
        )

    def ultranest(
        self,
        ess: int = 3000,
        ignore_nan: bool = False,
        viz_params: list[str] | None = None,
        print_result: bool = True,
        *,
        constructor_kwargs: dict | None = None,
        termination_kwargs: dict | None = None,
        read_file_config: dict | None = None,
    ) -> PosteriorResult:
        """Run :mod:`ultranest`'s implementation of nested sampling.

        Parameters
        ----------
        ess : int, optional
            The desired effective sample size.
        ignore_nan : bool, optional
            Whether to transform a NaN log probability to a large negative
            number (-1e300). The default is False.

            .. warning::
                Setting ``ignore_nan=True`` may fail to spot potential issues
                with model computation.
        viz_params : list, optional
            Parameters to visualize during sampling. The default is all.
        print_result : bool, optional
            Whether to print sampling result. The default is True.
        constructor_kwargs : dict, optional
            Extra parameters passed to
            :class:`ultranest.ReactiveNestedSampler`.
        termination_kwargs : dict, optional
            Extra parameters passed to
            :class:`ultranest.ReactiveNestedSampler.run()`.
        read_file_config : dict, optional
            Read the log file from a previous run. The dictionary should
            contain the log directory and other optional parameters. It
            should be noted that when providing this keyword argument, the
            sampler will not run, but read the log file instead and make
            sure the data and model settings are the same as the previous run.
        """
        from elisa.infer.samplers.ns.ultranest import UltraNestSampler

        if constructor_kwargs is None:
            constructor_kwargs = {}
        else:
            constructor_kwargs = dict(constructor_kwargs)

        if termination_kwargs is None:
            termination_kwargs = {}
        else:
            termination_kwargs = dict(termination_kwargs)
        termination_kwargs['min_ess'] = ess

        if viz_params is None:
            viz_params = self._helper.params_names['all']

        print('Running nested sampling of UltraNest...')
        t0 = time.time()
        sampler = UltraNestSampler(
            numpyro_model=self._helper.numpyro_model,
            seed=self._helper.seed['mcmc'],
            ignore_nan=ignore_nan,
            **constructor_kwargs,
        )
        samples = sampler.run(
            viz_sample_names=viz_params,
            read_file_config=read_file_config,
            **termination_kwargs,
        )
        if print_result:
            sampler.print_results()
        print(f'Sampling completed in {time.time() - t0:.2f} s')

        # format posterior samples
        samples = jax.tree.map(lambda x: x[None], samples)

        # effective sample size
        ess_overall = sampler.ess
        ess = dict.fromkeys(self._helper.params_names['all'], ess_overall)
        # relative mcmc efficiency
        total_sample = len(samples[self._helper.params_names['all'][0]])
        reff = float(ess_overall / total_sample)
        # model evidence
        lnZ = sampler.lnZ
        return self._generate_results(
            samples=samples,
            ess=ess,
            reff=reff,
            lnZ=lnZ,
            inference_library='ultranest',
        )
