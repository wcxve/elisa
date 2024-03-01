"""Subsequent analysis of likelihood or Bayesian fit."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, NamedTuple

import arviz as az
import jax
import numpy as np
from iminuit import Minuit
from iminuit.util import FMin
from scipy.stats import norm

from elisa.util.misc import make_pretty_table

from .simulation import SimFit

if TYPE_CHECKING:
    from .fit import BayesianFit, LikelihoodFit


class ConfidenceInterval(NamedTuple):
    """Confidence interval result."""

    mle: dict[str, float]
    interval: dict[str, tuple[float, float]]
    error: dict[str, tuple[float, float]]
    cl: float
    method: str
    status: dict


class BootstrapResult(NamedTuple):
    """Parametric bootstrap result."""

    mle: dict[str, float]
    params: dict[str, np.ndarray]
    stat: dict
    p_value: dict
    n: int
    n_valid: int
    seed: int
    result: dict


class MLEResult:
    """MLE result obtained from likelihood fit."""

    def __init__(self, minuit: Minuit, fit: LikelihoodFit, result: dict):
        self._minuit = minuit
        self._helper = fit._helper
        self._free_names = fit._free_names
        self._params_names = fit._params_names
        self._interest_params = fit._interest_names
        self._composite_params = fit._composite
        self._ndata = fit._ndata
        self._dof = fit._dof
        self._stat_type = fit._stat
        self._simfit = SimFit(fit)
        self._seed = fit._seed
        self._result = result
        self._mle = result['mle']
        self._fit_statistic = result['fit_stat']
        self._aic = result['aic']
        self._bic = result['bic']
        self._status = result['status']
        self._obs_data = result['obs_data']
        self._simulation = result['simulation']
        self._boot: BootstrapResult | None = None

    def __repr__(self):
        tab = make_pretty_table(
            ['Parameter', 'Value', 'Error'],
            [(k, f'{v[0]:.4g}', f'{v[1]:.4g}') for k, v in self._mle.items()],
        )
        s = 'MLE:\n' + tab.get_string() + '\n'

        stat_type = self._stat_type
        stat_value = self._fit_statistic
        ndata = self._ndata
        stat = [
            f'{i}: {stat_type[i]}={stat_value[i]:.2f}, ndata={ndata[i]}'
            for i in self._ndata.keys()
            if i != 'total'
        ]
        total_stat = stat_value['total']
        dof = self._dof
        stat += [
            f'Total: stat/dof={total_stat/dof:.2f} ({total_stat:.2f}/{dof})'
        ]
        s += '\nStatistic:\n' + '\n'.join(stat) + '\n'
        s += f'AIC: {self.aic:.2f}\n'
        s += f'BIC: {self.bic:.2f}\n'

        s += f'\nFit Status:\n{self.status}'

        return s

    def _repr_html_(self) -> str:
        # TODO
        return self.__repr__()

    @property
    def mle(self) -> dict[str, tuple[float, float]]:
        """MLE and error of parameters."""
        return self._mle

    @property
    def statistic(self) -> dict[str, float]:
        """Fit statistic."""
        return self._fit_statistic

    @property
    def ndata(self) -> dict[str, int]:
        """Number of data points."""
        return self._ndata

    @property
    def dof(self) -> int:
        """Degree of freedom."""
        return self._dof

    @property
    def aic(self) -> float:
        """Akaike information criterion with sample size correction."""
        return self._aic

    @property
    def bic(self) -> float:
        """Bayesian information criterion."""
        return self._bic

    @property
    def status(self) -> FMin:
        """Fit status of Minuit."""
        return self._status

    def print_summary(self) -> None:
        """Print summary of MLE."""
        print(repr(self))

    def ci(
        self,
        params: str | Sequence[str] | None = None,
        cl: float | int = 1,
        method: Literal['profile', 'boot'] = 'profile',
        **kwargs: dict,
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
            Parameters to calculate confidence intervals. Will calculate for
            all parameters if not provided.
        cl : float or int, optional
            Confidence level for the confidence interval. If 0 < `cl` < 1, the
            value is interpreted as the confidence level. If `cl` >= 1, it is
            interpreted as number of standard deviations. For example, ``cl=1``
            produces a 1-sigma or 68.3% confidence interval. The default is 1.
        method : {'profile', 'boot'}, optional
            Method used to calculate confidence. Either profile likelihood or
            parametric bootstrap method. The default is profile likelihood
            method.
        **kwargs : dict, optional
            Other kwargs forwarded to :meth:`MLEResult.boot`. Takes effect only
            if `method` is 'boot'.

        Returns
        -------
        ConfidenceInterval
            The confidence interval given the parameters and confidence level.

        """
        if not self._minuit.valid:
            msg = 'fit must be valid to calculate confidence interval'
            raise RuntimeError(msg)

        if params is None:
            params = self._interest_params

        elif isinstance(params, str):
            # check if params exist
            if params not in self._params_names:
                raise ValueError(f'parameter: {params} is not exist')

            params = [params]

        elif isinstance(params, Sequence):
            # check if params exist
            params = [str(i) for i in params]
            flag = [i in self._params_names for i in params]
            if not all(flag):
                params_err = ', '.join(
                    [i for i, j in zip(params, flag) if not j]
                )
                raise ValueError(f'parameters: {params_err} are not exist')

            params = [str(i) for i in params]

        else:
            raise ValueError('params must be str, or sequence of str')

        free_params = [i for i in params if i in self._free_names]
        composite_params = [i for i in params if i in self._composite_params]

        if cl <= 0.0:
            raise ValueError('cl must be non-negative')

        cl_ = 1.0 - 2.0 * norm.sf(cl) if cl >= 1.0 else cl

        mle = {k: v for k, v in self._result['params'].items() if k in params}

        helper = self._helper

        if method == 'profile':
            self._minuit.minos(*free_params, cl=cl)

            mle0 = self._minuit.values.to_dict()

            others = {  # set other unconstrained free parameter to mle
                i: mle0[i] for i in (set(mle0.keys()) - set(free_params))
            }

            ci = self._minuit.merrors
            lo = helper.to_params_dict(
                {k: mle0[k] + ci[k].lower for k in free_params} | others
            )
            up = helper.to_params_dict(
                {k: mle0[k] + ci[k].upper for k in free_params} | others
            )

            interval = {k: (lo[k], up[k]) for k in free_params}
            error = {k: (lo[k] - mle[k], up[k] - mle[k]) for k in free_params}
            status = {
                k: {
                    'valid': (v.lower_valid, v.upper_valid),
                    'at_limit': (v.at_lower_limit, v.at_upper_limit),
                    'at_max_fcn': (v.at_lower_max_fcn, v.at_upper_max_fcn),
                    'new_min': (v.lower_new_min, v.upper_new_min),
                }
                for k, v in ci.items()
            }

            def loss_factory(p):
                """Factory to create loss function for composite parameter."""

                @jax.jit
                def _(x):
                    """The loss when calculating CI of free parameter."""
                    unconstr = dict(zip(free_params, x))
                    p0 = helper.to_params_dict(unconstr)[p]
                    return helper.deviance_unconstr(x) + (p0 / mle[p] - 1) ** 2

                return _

            # confidence interval of function of parameters,
            # see, e.g. https://doi.org/10.1007/s11222-021-10012-y
            for p in composite_params:
                loss = loss_factory(p)
                mle_p = mle[p]
                m = Minuit(
                    loss,
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
            self.boot(**kwargs)
            boot_result = self._boot

            interval = jax.tree_map(
                lambda x: tuple(
                    np.quantile(x, q=(0.5 - cl_ / 2, 0.5 + cl_ / 2))
                ),
                {k: v for k, v in boot_result.params.items() if k in params},
            )

            error = {
                k: (interval[k][0] - mle[k], interval[k][1] - mle[k])
                for k in params
            }

            status = {
                'n': boot_result.n,
                'n_valid': boot_result.n_valid,
                'seed': boot_result.seed,
            }

        else:
            raise ValueError('method must be either "profile" or "boot"')

        return ConfidenceInterval(
            mle=_format_result(mle, params),
            interval=_format_result(interval, params),
            error=_format_result(error, params),
            cl=cl_,
            method=method,
            status=status,
        )

    def boot(
        self,
        n: int = 10000,
        parallel: bool = True,
        seed: int | None = None,
        progress: bool = True,
    ):
        """Parametric bootstrap.

        Parameters
        ----------
        n : int, optional
            The number of bootstrap. The default is 10000.
        parallel : bool, optional
            Whether to run the fit in parallel. The default is True.
        seed : int, optional
            The random seed used in parametric bootstrap. Defaults to the seed
            as in the fitting context.
        progress : bool, optional
            Whether to display progress bar. The default is True.

        """
        if not self._minuit.valid:
            msg = 'fit must be valid to perform bootstrap'
            raise RuntimeError(msg)

        if seed is None:
            seed = self._seed
        else:
            seed = int(seed)

        # reuse previous result if all setup is the same
        if self._boot and self._boot.n == n and self._boot.seed == seed:
            return

        boot_result = self._simfit.run_one_set(
            self._result['constr'],
            n,
            seed,
            parallel,
            run_str='Bootstrap',
            progress=progress,
        )

        p_value = {
            'rep': jax.tree_map(
                lambda obs, sim: np.sum(sim >= obs, axis=0) / len(sim),
                self._result['stat'],
                boot_result['stat_rep'],
            ),
            'fit': jax.tree_map(
                lambda obs, sim: np.sum(sim >= obs, axis=0) / len(sim),
                self._result['stat'],
                boot_result['stat_fit'],
            ),
        }

        self._boot = BootstrapResult(
            mle=self._result['unconstr'],
            params=boot_result['params_fit'],
            stat=boot_result['stat_fit'],
            p_value=p_value,
            n=n,
            n_valid=boot_result['n_valid'],
            seed=seed,
            result=boot_result,
        )

    def plot_data(
        self, plots='data ldata chi pchi deviance pit ne ene eene fv vfv'
    ):
        ...

    def plot_corner(self):
        # correlation map or bootstrap distribution
        ...


class CredibleInterval(NamedTuple):
    """Credible interval result."""

    median: dict[str, float]
    interval: dict[str, tuple[float, float]]
    error: dict[str, tuple[float, float]]
    prob: float
    method: str


class PPCResult(NamedTuple):
    """Posterior predictive check result."""

    params_rep: dict[str, np.ndarray]
    params_fit: dict[str, np.ndarray]
    p_value: dict
    n: int
    n_valid: int
    seed: int
    idata: az.InferenceData
    result: dict


class PosteriorResult:
    """Posterior sampling result obtained from Bayesian fit."""

    def __init__(
        self,
        idata: az.InferenceData,
        sampler,
        mle: MLEResult,
        fit: BayesianFit,
    ):
        self._idata = idata
        self._ess = idata.attrs['ess']
        self._reff = idata.attrs['reff']
        self._lnZ = idata.attrs['lnZ']
        self._rhat = None
        self._divergence = None
        self._waic = None
        self._loo = None
        self._mle = mle
        self._sampler = sampler
        self._helper = fit._helper
        self._free_names = fit._free_names
        self._params_names = fit._params_names
        self._interest_params = fit._interest_names
        self._composite_params = fit._composite
        self._ndata = fit._ndata
        self._dof = fit._dof
        self._stat_type = fit._stat
        self._simfit = SimFit(fit)
        self._seed = fit._seed
        self._ppc: PPCResult | None = None

    def __repr__(self):
        # TODO
        return super().__repr__()

    def _repr_html_(self) -> str:
        # TODO
        return self.__repr__()

    @property
    def ndata(self) -> dict[str, int]:
        """Number of data points."""
        return self._ndata

    @property
    def dof(self) -> int:
        """Degree of freedom."""
        return self._dof

    @property
    def reff(self) -> float:
        """Relative MCMC efficiency."""
        return self._reff

    @property
    def ess(self) -> dict[str, int]:
        """Effective MCMC sample size."""
        return self._ess

    @property
    def rhat(self) -> dict[str, float]:
        """Computes split R-hat over MCMC chains.

        In general, only fully trust the sample if R-hat is less than 1.01. In
        early workflow, R-hat below 1.1 is often sufficient. See [1]_ for more
        information.

        References
        ----------
        .. [1] : https://arxiv.org/abs/1903.08008

        """
        if self._rhat is None:
            posterior = self._idata['posterior']

            if len(posterior['chain']) == 1:
                rhat = {k: float('nan') for k in posterior.data_vars.keys()}
            else:
                rhat = {
                    k: float(v.values)
                    for k, v in az.rhat(posterior).data_vars.items()
                }

            self._rhat = rhat

        return self._rhat

    @property
    def divergence(self) -> int:
        """Number of divergent samples."""
        if self._divergence is None:
            if 'sample_stats' in self._idata:
                n = int(self._idata['sample_stats']['diverging'].sum())
            else:
                n = 0

            self._divergence = n

        return self._divergence

    @property
    def waic(self) -> float:
        """The widely applicable information criterion (WAIC).

        Estimates the expected log point-wise predictive density (elpd) using
        WAIC. Also calculates the WAIC's standard error and the effective
        number of parameters. See [1]_ and [2]_ for more information.

        References
        ----------
        .. [1] https://arxiv.org/abs/1507.04544
        .. [2] https://arxiv.org/abs/1004.2316

        """
        if self._waic is None:
            self._waic = az.waic(
                self._idata, var_name='channels', scale='deviance'
            )

        return self._waic

    @property
    def loo(self) -> float:
        """Pareto-smoothed importance sampling leave-one-out cross-validation
        (PSIS-LOO-CV).

        Estimates the expected log point-wise predictive density (elpd) using
        PSIS-LOO-CV. Also calculates LOO's standard error and the effective
        number of parameters. For more information, see [1]_, [2]_ and [3]_.

        References
        ----------
        .. [1] https://avehtari.github.io/modelselection/CV-FAQ.html
        .. [2] https://arxiv.org/abs/1507.04544
        .. [3] https://arxiv.org/abs/1507.02646

        """
        if self._loo is None:
            self._loo = az.loo(
                self._idata,
                var_name='channels',
                reff=self._reff,
                scale='deviance',
            )

        return self._loo

    @property
    def lnZ(self) -> tuple[float, float]:
        """Log model evidence."""
        return self._lnZ

    def print_summary(self) -> None:
        """Print summary of posterior result."""
        print(self)

    def ci(
        self,
        params: str | Sequence[str] | None = None,
        prob: float | int = 1,
        hdi: bool = False,
    ) -> CredibleInterval:
        """Calculate credible intervals for given parameters.

        Parameters
        ----------
        params : str or list of str, optional
            Parameters to calculate credible intervals. Will calculate for all
            parameters if not provided.
        prob : float or int, optional
            The probability mass of samples within the credible interval. If
            0 < `prob` < 1, the value is interpreted as the probability mass.
            If `prob` >= 1, it is interpreted as number of standard deviations.
            For example, ``prob=1`` produces a 1-sigma or 68.3% credible
            interval. The default is 1.
        hdi : bool, optional
            Whether to return the highest density interval. The default is
            False, which means an equal tailed interval is returned.

        Returns
        -------
        CredibleInterval
            The credible interval given the parameters and probability mass.

        """
        if params is None:
            params = list(self._interest_params)

        elif isinstance(params, str):
            # check if params exist
            if params not in self._params_names:
                raise ValueError(f'parameter: {params} is not exist')

            params = [params]

        elif isinstance(params, Sequence):
            # check if params exist
            params = [str(i) for i in params]
            flag = [i in self._params_names for i in params]
            if not all(flag):
                params_err = ', '.join(
                    [i for i, j in zip(params, flag) if not j]
                )
                raise ValueError(f'parameters: {params_err} are not exist')

            params = [str(i) for i in params]

        else:
            raise ValueError('params must be str, or sequence of str')

        if prob <= 0.0:
            raise ValueError('prob must be non-negative')

        prob_ = 1.0 - 2.0 * norm.sf(prob) if prob >= 1.0 else prob

        if hdi:
            median = self._idata['posterior'].median()
            median = {
                k: float(v) for k, v in median.data_vars.items() if k in params
            }
            interval = az.hdi(self._idata, prob_, var_names=params)
            interval = {
                k: (float(v[0]), float(v[1]))
                for k, v in interval.data_vars.items()
            }
        else:
            q = [0.5, 0.5 - prob_ / 2.0, 0.5 + prob_ / 2.0]
            quantile = self._idata['posterior'].quantile(q)
            quantile = {
                k: v for k, v in quantile.data_vars.items() if k in params
            }
            median = {k: float(v[0]) for k, v in quantile.items()}
            interval = {
                k: (float(v[1]), float(v[2])) for k, v in quantile.items()
            }

        error = {
            k: (interval[k][0] - median[k], interval[k][1] - median[k])
            for k in params
        }

        return CredibleInterval(
            median=_format_result(median, params),
            interval=_format_result(interval, params),
            error=_format_result(error, params),
            prob=prob_,
            method='HDI' if hdi else 'ETI',
        )

    def ppc(
        self,
        n: int = 10000,
        parallel: bool = True,
        seed: int | None = None,
        progress: bool = True,
    ):
        """Perform posterior predictive check.

        Parameters
        ----------
        n : int, optional
            The number of posterior predictions. The default is 10000.
        parallel : bool, optional
            Whether to run the fit in parallel. The default is True.
        seed : int, optional
            The random seed used in prediction. Defaults to the seed as in the
            fitting context.
        progress : bool, optional
            Whether to display progress bar. The default is True.

        """
        if seed is None:
            seed = self._seed
        else:
            seed = int(seed)

        # directly return previous results if the configuration is the same
        if self._ppc and self._ppc.n == n and self._ppc.seed == seed:
            return self._ppc

        idata = self._idata

        # random select n samples from posterior
        rng = np.random.default_rng(seed)
        i = rng.integers(0, idata['posterior'].chain.size, n)
        j = rng.integers(0, idata['posterior'].draw.size, n)

        posterior = {
            k: v.values[i, j] for k, v in idata['posterior'].data_vars.items()
        }

        ppc_result = self._simfit.run_multi_sets(
            posterior, seed, parallel, run_str='PPC', progress=progress
        )

        # filter out invalid results
        i = i[ppc_result['valid']]
        j = j[ppc_result['valid']]

        posterior = {
            k: v.values[i, j] for k, v in idata['posterior'].data_vars.items()
        }

        # coords and dims for creating idata for ppc
        chain_str = np.char.add(i.astype(str), '_')
        draw_str = np.char.add(chain_str, j.astype(str))
        coords = {
            k: v.values for k, v in idata['observed_data'].coords.items()
        } | {'chain': [0], 'draw': draw_str}
        dims = {
            k: list(v.dims)
            for k, v in idata['observed_data'].items()
            if k != 'total'
        }

        log_likelihood = {
            k: v.values[None, i, j]
            for k, v in idata['log_likelihood'].data_vars.items()
        }

        posterior_predictive = {
            k.replace('_spec', '_Non').replace('_back', '_Noff'): v[None, ...]
            for k, v in ppc_result['data'].items()
        }
        posterior_predictive |= {
            'channels': np.concatenate(
                [
                    posterior_predictive[k]
                    for k in self._ndata.keys()
                    if k != 'total'
                ],
                axis=-1,
            )
        }

        observed_data = {
            k: v.values
            for k, v in idata['observed_data'].data_vars.items()
            if k != 'total'
        }

        idata_ppc = az.from_dict(
            posterior=posterior,
            posterior_predictive=posterior_predictive,
            log_likelihood=log_likelihood,
            observed_data=observed_data,
            coords=coords,
            dims=dims,
        )
        idata_ppc['observed_data']['total'] = (
            (),
            idata['observed_data']['total'].values,
        )

        # stat for observed data and params for computing ppp
        stat = -2.0 * idata_ppc['log_likelihood']
        stat_obs = {
            'total': stat['total'].values[0],
            'group': {
                k: stat[k].sum(dim=f'{k}_channel').values[0]
                for k in self._ndata.keys()
                if k != 'total'
            },
            'point': {
                k: stat[k].values[0]
                for k in self._ndata.keys()
                if k != 'total'
            },
        }
        p_value = {
            'rep': jax.tree_map(
                lambda obs, sim: np.sum(sim >= obs, axis=0) / len(sim),
                stat_obs,
                ppc_result['stat_rep'],
            ),
            'fit': jax.tree_map(
                lambda obs, sim: np.sum(sim >= obs, axis=0) / len(sim),
                self._mle._result['stat'],
                ppc_result['stat_fit'],
            ),
        }

        self._ppc = PPCResult(
            params_rep=posterior,
            params_fit=ppc_result['params_fit'],
            p_value=p_value,
            n=n,
            n_valid=ppc_result['n_valid'],
            seed=seed,
            idata=idata_ppc,
            result=ppc_result,
        )


def _format_result(v, params_order):
    """Order the result dict and use float type."""
    formatted = jax.tree_map(float, v)
    return {k: formatted[k] for k in params_order}
