"""Model fit in maximum likelihood or Bayesian way."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from iminuit import Minuit
from numpyro.infer import MCMC, NUTS, init_to_value

from elisa.data.ogip import Data, FitData
from elisa.infer.helper import Helper, get_helper
from elisa.infer.likelihood import _STATISTIC_OPTIONS
from elisa.infer.nested_sampling import NestedSampler
from elisa.infer.results import MLEResult, PosteriorResult
from elisa.models.model import Model, get_model_info
from elisa.util.misc import add_suffix, build_namespace, make_pretty_table

if TYPE_CHECKING:
    from typing import Any, Callable, Literal

    from prettytable import PrettyTable

    from elisa.infer.likelihood import Statistic
    from elisa.models.model import ModelInfo
    from elisa.util.typing import ArrayLike, JAXArray, JAXFloat


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
    _ns: NestedSampler | None = None

    def __init__(
        self,
        data: Data | Sequence[Data],
        model: Model | Sequence[Model],
        stat: Statistic | Sequence[Statistic] | None = None,
        seed: int = 42,
    ):
        inputs = self._parse_input(data, model, stat)
        data: list[FitData] = inputs[0]
        models: list[Model] = inputs[1]
        stats: list[Statistic] = inputs[2]

        # if a component is not fit with all datasets,
        # add names of data sets to be fit with it as its name/latex suffix
        data_names = [d.name for d in data]
        data_to_cid = {n: m._comps_id for n, m in zip(data_names, models)}
        cid_to_comp = {c._id: c for m in models for c in m._comps}
        cid = list(cid_to_comp.keys())
        comps = list(cid_to_comp.values())
        cid_to_data_suffix = {
            i: '+'.join(i for i in data_names if i not in names)  # keep order
            if (names := [n for n in data_names if i not in data_to_cid[n]])
            else ''
            for i in cid
        }
        data_suffix = list(cid_to_data_suffix.values())
        cname = [comp.name for comp in comps]
        name_with_data_suffix = list(map(''.join, zip(cname, data_suffix)))
        num_suffix = build_namespace(name_with_data_suffix)['suffix_num']
        cname_num = add_suffix(cname, num_suffix, True, True)
        cname_num_data = add_suffix(cname_num, data_suffix, False, True)
        cid_to_name = dict(zip(cid, cname_num_data))

        # get model info
        self._model_info: ModelInfo = get_model_info(comps, cid_to_name)

        # first filter out duplicated models then compile the remaining models,
        # this is intended to avoid re-compilation of the same model
        models_id = [id(m) for m in models]
        mid_to_model = dict(zip(models_id, models))
        compiled_model = {
            mid: m.compile(model_info=self._model_info)
            for mid, m in mid_to_model.items()
        }
        data_to_mid = dict(zip(data_names, models_id))
        self._model = {
            name: compiled_model[mid] for name, mid in data_to_mid.items()
        }

        # store data, stat, seed
        self._data: dict[str, FitData] = dict(zip(data_names, data))
        self._stat: dict[str, Statistic] = dict(zip(data_names, stats))
        self._seed: int = int(seed)

        # make model information table
        self._make_info_table()

        self.__helper: Helper | None = None

    def _optimize_lm(
        self, unconstr_init: JAXArray
    ) -> tuple[JAXArray, JAXFloat]:
        """Search MLE by Levenberg-Marquardt algorithm of :mod:`optimistix`."""
        if self._lm is None:
            lm_solver = optx.LevenbergMarquardt(rtol=0.0, atol=1e-6)
            residual = jax.jit(lambda x, aux: self._helper.residual(x))

            def lm(init):
                res = optx.least_squares(residual, lm_solver, init)
                return res.value, jnp.linalg.norm(res.state.f_info.grad)

            self._lm = jax.jit(lm)

        return self._lm(jnp.asarray(unconstr_init, float))

    def _optimize_ns(self) -> JAXArray:
        """Search MLE using nested sampling of :mod:`jaxns`."""
        if self._ns is None:
            self._ns = NestedSampler(
                self._helper.numpyro_model,
                constructor_kwargs={
                    'max_samples': 100000,
                    'num_live_points': max(800, 100 * self._helper.nparam),
                },
            )
            t0 = time.time()
            print('Start searching MLE...')
            self._ns.run(rng_key=jax.random.PRNGKey(self._helper.seed['mcmc']))
            print(f'Search cost {time.time() - t0:.2f} s')

        ns = self._ns

        samples = ns._results.samples
        loglike = [
            samples[f'{i}_loglike'].sum(axis=-1) for i in self._data.keys()
        ]
        mle_idx = np.sum(loglike, axis=0).argmax()
        mle = jax.tree_map(lambda s: s[mle_idx], samples)
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
            '</details><br/>'
        )

    def _make_info_table(self):
        fields = ('Data', 'Model', 'Statistic')
        rows = tuple(
            zip(
                self._data,
                (m.name for m in self._model.values()),
                self._stat.values(),
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
        data: Data | Sequence[Data],
        model: Model | Sequence[Model],
        stat: Statistic | Sequence[Statistic] | None,
    ) -> tuple[list[FitData], list[Model], list[Statistic]]:
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

        def get_stat(d: FitData) -> Statistic:
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

        def check_stat(d: FitData, s: Statistic):
            """Check if data type and likelihood are matched."""
            name = d.name
            if not d.spec_poisson and s != 'chi2':
                msg = f'{name} is Gaussian data, use stat "chi2" instead'
                raise ValueError(msg)

            if s == 'chi2' and np.any(d.back_error == 0.0):
                raise ValueError(
                    f'"chi2" is not valid for {name} data, which has zero '
                    'uncertainties; grouping the data may fix this error'
                )

            elif s == 'cstat' and d.has_back:
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

            elif s == 'pgstat':
                if not d.has_back:
                    msg = f'PG-statistic is not valid for {name}, which '
                    msg += 'requires Gaussian background data, '
                    msg += 'use C-statistic instead (cstat)'
                    raise ValueError(msg)

                if np.any(d.back_error == 0.0):
                    raise ValueError(
                        f'PG-statistic is not valid for {name} data, '
                        'which has zero background uncertainties; '
                        'grouping the data may fix this error'
                    )

            elif s == 'wstat' and not (d.has_back and d.back_poisson):
                msg = f'W-statistic is not valid for {name}, which requires '
                msg += 'Poisson background data, use C-statistic (cstat) '
                msg += 'instead'
                raise ValueError(msg)

        # ====================== some helper functions ========================

        # get data
        data_list: list[FitData] = [
            FitData.from_data(d) for d in get_list(data, 'data', Data, 'Data')
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
            msg = f'data names are not unique: {", ".join(name_list)}'
            raise ValueError(msg)

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
        for d, s in zip(data_list, stat_list):
            check_stat(d, s)

        return data_list, model_list, stat_list


class MaxLikeFit(Fit):
    _tab_config = ('Maximum Likelihood Fit', frozenset({'Prior'}))

    def _optimize_minuit(
        self,
        unconstr_init: JAXArray,
        strategy: Literal[0, 1, 2],
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

        # TODO: test if simplex can be used to "polish" the initial guess
        # if strategy == 0:
        #     max_it = 10
        #     nit = 0
        #     minuit.strategy = 0
        #     minuit.migrad()
        #     while (not minuit.fmin.is_valid) and nit < max_it:
        #         minuit.hesse()
        #         minuit.migrad()
        #         nit += 1
        #     minuit.hesse()
        #
        # elif strategy in {1, 2}:
        #     minuit.strategy = strategy
        #     minuit.migrad(iterate=10)
        #
        # else:
        #     raise ValueError(f'invalid strategy {strategy}')

        minuit.strategy = strategy
        minuit.migrad(iterate=10)
        if strategy == 0:
            # refine hessian matrix
            minuit.hesse()

        # set strategy to 2 for accuracy of confidence interval calculation
        minuit.strategy = 2

        return minuit

    def mle(
        self,
        init: ArrayLike | dict | None = None,
        method: Literal['minuit', 'lm', 'ns'] = 'minuit',
        strategy: Literal[0, 1, 2] = 1,
    ) -> MLEResult:
        """Search Maximum Likelihood Estimation (MLE) for the model.

        Parameters
        ----------
        init : array_like or dict, optional
            Initial guess for the maximum likelihood estimation. The default is
            as parameters' default values.
        method : {'minuit', 'lm', 'ns'}, optional
            Optimization algorithm used to find the MLE.
            Available options are:

                * ``'minuit'``: Migrad algorithm of :mod:`iminuit`.
                * ``'lm'``: Levenberg-Marquardt algorithm of :mod:`optimistix`.
                * ``'ns'``: Nested sampling of :mod:`jaxns`. This option first
                  search MLE globally, then polish it with local minimization.

            The default is 'minuit'.
        strategy : {0, 1, 2}, optional
            Minuit optimization strategy, available options are:

                * ``0``: Fast.
                * ``1``: Default.
                * ``2``: Careful, which improves accuracy at the cost of time.

        Returns
        -------
        MLEResult
            The MLE result.
        """
        if isinstance(init, (np.ndarray, jax.Array, Sequence)):
            init_unconstr = self._helper.constr_arr_to_unconstr_arr(init)
        elif isinstance(init, dict):
            init_unconstr = self._helper.constr_dic_to_unconstr_arr(init)
        elif init is None:
            init_unconstr = self._helper.free_default['unconstr_arr']
        else:
            raise TypeError('params must be a array, sequence, or mapping')

        if method == 'lm':  # use Levenberg-Marquardt algorithm to find MLE
            init_unconstr, grad_l2_norm = self._optimize_lm(init_unconstr)
        elif method == 'ns':  # use nested sampling to find MLE
            init_unconstr = self._optimize_ns()
        else:
            if method != 'minuit':
                raise ValueError(f'unsupported optimization method {method}')

        minuit = self._optimize_minuit(init_unconstr, strategy)

        return MLEResult(minuit, self._helper)


class BayesFit(Fit):
    _tab_config = ('Bayesian Fit', frozenset({'Bound'}))

    def nuts(
        self,
        warmup=2000,
        samples=20000,
        chains: int | None = None,
        init: dict[str, float] | None = None,
        progress: bool = True,
        **nuts_kwargs: dict,
    ) -> PosteriorResult:
        """Run the No-U-Turn Sampler of :mod:`numpyro`.

        .. note::
            If the chains are not converged well, see ref [1]_ for more
            information on how to finetune the NUTS sampler.

        Parameters
        ----------
        warmup : int, optional
            Number of warmup steps.
        samples : int, optional
            Number of samples to generate from each chain.
        chains : int, optional
            Number of MCMC chains to run. If there are not enough devices
            available, chains will run in sequence. Defaults to the number of
            ``jax.local_device_count()``.
        init : dict, optional
            Initial parameter for sampler to start from.
        progress : bool, optional
            Whether to show progress bar during sampling. The default is True.
        **nuts_kwargs : dict
            Extra parameters passed to :class:`numpyro.infer.NUTS`.

        Returns
        -------
        PosteriorResult
            The posterior sampling result.

        References
        ----------
        .. [1] NumPyro tutorial: `Bad posterior geometry and how to deal with
               it <https://num.pyro.ai/en/stable/tutorials/bad_posterior_geometry.html>`__
        """
        device_count = jax.local_device_count()

        if chains is None:
            chains = device_count
        else:
            chains = int(chains)

        samples = int(samples)

        # the total samples number should be multiple of the device number
        if chains * samples % device_count != 0:
            samples += device_count - samples % device_count

        # TODO: option to let sampler starting from MLE
        if init is None:
            init = self._helper.free_default['constr_dic']

        default_nuts_kwargs = {
            'dense_mass': True,
            'target_accept_prob': 0.8,
            'max_tree_depth': 10,
            'init_strategy': init_to_value(values=init),
        }
        nuts_kwargs = default_nuts_kwargs | nuts_kwargs
        nuts_kwargs['model'] = self._helper.numpyro_model

        sampler = MCMC(
            NUTS(**nuts_kwargs),
            num_warmup=warmup,
            num_samples=samples,
            num_chains=chains,
            progress_bar=progress,
        )

        sampler.run(
            rng_key=jax.random.PRNGKey(self._helper.seed['mcmc']),
            extra_fields=('energy', 'num_steps'),
        )
        return PosteriorResult(sampler, self._helper, self)

    def ns(
        self,
        max_samples: int = 100000,
        num_live_points: int | None = None,
        num_parallel_workers: int = 1,
        difficult_model: bool = False,
        parameter_estimation: bool = False,
        verbose: bool = False,
        term_cond: dict | None = None,
        **ns_kwargs: dict,
    ) -> PosteriorResult:
        """Run the Nested Sampler of :mod:`jaxns`.

        .. note::
            For more information of the sampler parameters, see ref [1]_ [2]_.

        Parameters
        ----------
        max_samples : int, optional
            Maximum number of posterior samples. The default is 100000.
        num_live_points : int, optional
            Approximate number of live points.
        num_parallel_workers : int, optional
            Parallel workers number. The default is 1.
        difficult_model : bool, optional
            Use more robust default settings when True. The default is False.
        parameter_estimation : bool, optional
            Use more robust default settings for parameter estimation. The
            default is False.
        verbose : bool, optional
            Print progress information. The default is False.
        term_cond : dict, optional
            Termination conditions for the sampling. The default is as in
            :class:`jaxns.TermCondition`.
        **ns_kwargs : dict
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
        num_parallel_workers = int(num_parallel_workers)

        constructor_kwargs = {
            'max_samples': max_samples,
            'num_live_points': num_live_points,
            'num_parallel_workers': num_parallel_workers,
            'difficult_model': difficult_model,
            'parameter_estimation': parameter_estimation,
            'verbose': verbose,
        }
        constructor_kwargs |= ns_kwargs

        termination_kwargs = {'dlogZ': 1e-4}
        if term_cond is not None:
            termination_kwargs |= dict(term_cond)

        sampler = NestedSampler(
            self._helper.numpyro_model,
            constructor_kwargs=constructor_kwargs,
            termination_kwargs=termination_kwargs,
        )

        print('Start nested sampling...')
        t0 = time.time()
        sampler.run(rng_key=jax.random.PRNGKey(self._helper.seed['mcmc']))
        print(f'Sampling cost {time.time() - t0:.2f} s')
        return PosteriorResult(sampler, self._helper, self)
