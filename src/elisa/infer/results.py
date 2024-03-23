"""Subsequent analysis of maximum likelihood or Bayesian fit."""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import metadata
from typing import TYPE_CHECKING, NamedTuple

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import scipy.stats as stats
from iminuit import Minuit
from numpyro.infer import MCMC

from elisa.__about__ import __version__
from elisa.infer.nested_sampling import NestedSampler
from elisa.util.misc import make_pretty_table

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Literal

    from iminuit.util import FMin

    from elisa.infer.fit import Fit
    from elisa.infer.helper import Helper


class FitResult(ABC):
    """Fit result."""

    _helper: Helper

    def __init__(self, helper: Helper):
        self._helper = helper

    def _check_ci_params(self, params) -> list[str]:
        params_names = self._helper.params_names

        all_params = set(params_names['all']) | set(self._helper.params_setup)
        forwarded = {
            k: v[0]
            for k, v in self._helper.params_setup.items()
            if v[1].name == 'Forwarded'
        }
        fixed = [
            k
            for k, v in self._helper.params_setup.items()
            if v[1].name == 'Fixed'
        ]
        integrated = [
            k
            for k, v in self._helper.params_setup.items()
            if v[1].name == 'Integrated'
        ]

        if params is None:
            params = set(params_names['interest'])

        elif isinstance(params, str):
            # check if params exist
            if params not in all_params:
                raise ValueError(f'parameter {params} is not exist')

            params = {params}

        elif isinstance(params, Iterable):
            # check if params exist
            params = {str(i) for i in params}
            if not params.issubset(all_params):
                params_err = params - set(params_names['all'])
                raise ValueError(f'parameters: {params_err} are not exist')

        else:
            raise ValueError('params must be str, or sequence of str')

        if params_err := params.intersection(forwarded):
            forwarded = {i: forwarded[i] for i in params_err}
            info = ', '.join(f'{k} to {v}' for k, v in forwarded.items())
            raise RuntimeError(
                f"parameters are linked: {info}; corresponding parameters' "
                'name should be used to calculate CIs'
            )

        if params_err := params.intersection(fixed):
            info = ', '.join(params_err)
            raise RuntimeError(
                f'cannot calculate CIs of fixed parameters: {info}'
            )

        if params_err := params.intersection(integrated):
            info = ', '.join(params_err)
            raise RuntimeError(
                'cannot calculate CIs of parameters being integrated-out: '
                f'{info}'
            )

        return sorted(params, key=params_names['all'].index)

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def _repr_html_(self):
        pass

    # def plot_data(
    #     self, plots='data ldata chi pchi deviance pit ne ene eene fv vfv'
    # ):
    #     ...
    #
    # def plot_corner(self):
    #     # correlation map, bootstrap distribution, posterior distribution
    #     ...

    def summary(self, file=None) -> None:
        """Print the summary of fitting setup.

        Parameters
        ----------
        file: file-like
            An object with a ``write(string)`` method. This is passed to
            :py:func:`print`.
        """
        print(repr(self), file=file)

    @property
    def ndata(self) -> dict[str, int]:
        """Data points number."""
        return self._helper.ndata

    @property
    def dof(self) -> int:
        """Degree of freedom."""
        return self._helper.dof


class MLEResult(FitResult):
    """Result of maximum likelihood fit."""

    def __init__(self, minuit: Minuit, helper: Helper):
        super().__init__(helper)

        self._minuit = minuit

        self._mle_unconstr = jnp.array(minuit.values, float)
        mle, cov = helper.get_mle(self._mle_unconstr)

        if np.isnan(cov).any() and minuit.covariance is not None:
            cov_unconstr = jnp.array(minuit.covariance, float)
            cov = helper.params_covar(self._mle_unconstr, cov_unconstr)

        err = jnp.sqrt(jnp.diagonal(cov))

        # MLE of model params in constrained space
        self._mle = dict(zip(helper.params_names['all'], zip(mle, err)))

        # model deviance at MLE
        self._deviance = jax.jit(helper.deviance)(self._mle_unconstr)

        # model values at MLE
        sites = jax.jit(helper.get_sites)(self._mle_unconstr)
        self._model_values = sites['models']

        # model comparison statistics
        k = self._helper.nparam
        n = self._helper.ndata['total']
        stat = self._deviance['total']
        self._aic = float(stat + k * 2 * (1 + (k + 1) / (n - k - 1)))
        self._bic = float(stat + k * np.log(n))

        # parametric bootstrap result
        self._boot: BootstrapResult | None = None

    def __repr__(self):
        tab = make_pretty_table(
            ['Parameter', 'Value', 'Error'],
            [(k, f'{v[0]:.4g}', f'{v[1]:.4g}') for k, v in self.mle.items()],
        )

        stat_type = self._helper.statistic
        deviance = self.deviance
        ndata = self.ndata
        stat = [
            f'{i}: {stat_type[i]}={deviance[i]:.2f}, ndata={ndata[i]}'
            for i in self.ndata.keys()
            if i != 'total'
        ]
        total_stat = deviance['total']
        dof = self.dof
        stat += [
            f'Total: stat/dof={total_stat/dof:.2f} ({total_stat:.2f}/{dof})'
        ]
        s = 'MLE:\n' + tab.get_string() + '\n'
        s += '\nStatistic:\n' + '\n'.join(stat) + '\n'
        s += f'AIC: {self.aic:.2f}\n'
        s += f'BIC: {self.bic:.2f}\n'
        s += f'\nFit Status:\n{self.status}'

        return s

    def _repr_html_(self):
        return self.__repr__()

    def ci(
        self,
        params: str | Iterable[str] | None = None,
        cl: float | int = 1,
        method: Literal['profile', 'boot'] = 'profile',
    ) -> ConfidenceInterval:
        """Calculate confidence intervals.

        Parameters
        ----------
        params : str or sequence of str, optional
            Parameters to calculate confidence intervals. If not specified,
            calculate for parameters of interest.
        cl : float or int, optional
            Confidence level for the confidence interval. If 0 < `cl` < 1, the
            value is interpreted as the confidence level. If `cl` >= 1, it is
            interpreted as the number of standard deviations. For example,
            ``cl=1`` produces a 1-sigma or 68.3% confidence interval.
            The default is 1.
        method : {'profile', 'boot'}, optional
            Method used to calculate confidence. Available options are:

                * ``'profile'``: use Minos algorithm of Minuit to find the
                  confidence intervals based on the profile likelihood
                * ``'boot'``: use parametric bootstrap method to calculate
                  the confidence intervals. :meth:`MLEResult.boot` must be
                  called before using this method.

            The default is ``'profile'``.

        Returns
        -------
        ConfidenceInterval
            The confidence intervals.
        """
        if not self._minuit.valid:
            msg = 'the fit must be valid to calculate confidence interval'
            raise RuntimeError(msg)

        if cl <= 0.0:
            raise ValueError('cl must be non-negative')

        if method not in {'profile', 'boot'}:
            raise ValueError(f'unsupported method: {method}')

        params_names = self._helper.params_names

        params = self._check_ci_params(params)

        params_set = set(params)
        free = params_set.intersection(params_names['free'])
        composite = params_set.intersection(params_names['deterministic'])
        assert free | composite == params_set

        if method == 'profile':
            empty = ({}, {})
            res1 = self._ci_free(free, cl) if free else empty
            res2 = self._ci_composite(composite, cl) if composite else empty
            intervals, status = (r1 | r2 for r1, r2 in zip(res1, res2))

        elif method == 'boot':
            if self._boot is None:
                raise RuntimeError(
                    'before using the bootstrap method to calculate confidence'
                    ' intervals, MLEResult.boot(...) must be called'
                )
            intervals, status = self._ci_boot(params, cl)

        else:
            raise ValueError("method must be either 'profile' or 'boot'")

        mle = {k: v[0] for k, v in self.mle.items()}
        intervals = format_result(intervals, params)
        errors = {
            k: (intervals[k][0] - mle[k], intervals[k][1] - mle[k])
            for k in params
        }

        return ConfidenceInterval(
            mle=mle,
            intervals=intervals,
            errors=errors,
            cl=1.0 - 2.0 * stats.norm.sf(cl) if cl >= 1.0 else cl,
            method=method,
            status=status,
        )

    def boot(
        self,
        n: int = 10000,
        seed: int | None = None,
        parallel: bool = True,
        progress: bool = True,
    ):
        """Preform parametric bootstrap.

        Parameters
        ----------
        n : int, optional
            Number of parametric bootstraps based on the MLE. The default is
            10000.
        seed : int, optional
            The seed of random number generator used in parametric bootstrap.
        parallel : bool, optional
            Whether to run simulation fit in parallel. The default is True.
        progress : bool, optional
            Whether to display progress bar. The default is True.
        """
        # reuse the previous result if all setup is the same
        if self._boot and self._boot.n == n and self._boot.seed == seed:
            return

        helper = self._helper
        n = int(n)
        seed = helper.seed['pred'] if seed is None else int(seed)
        params = {i: self._mle[i][0] for i in helper.params_names['free']}
        models = self._model_values

        # perform parametric bootstrap
        result = helper.simulate_and_fit(
            seed, params, models, n, parallel, progress, 'Bootstrap'
        )
        valid = result.pop('valid')
        result = jax.tree_map(lambda x: x[valid], result)

        self._boot = BootstrapResult(
            mle={k: v[0] for k, v in self.mle.items()},
            data=result['data'],
            models=result['models'],
            params=result['params'],
            deviance=result['deviance'],
            p_value=jax.tree_map(
                lambda obs, sim: np.sum(sim >= obs, axis=0) / len(sim),
                self._deviance,
                result['deviance'],
            ),
            n=n,
            n_valid=np.sum(valid),
            seed=seed,
        )

    def _ci_free(self, names: Iterable[str], cl: float | int):
        """Confidence interval of free parameters."""
        self._minuit.minos(*names, cl=cl)
        mle_unconstr = self._minuit.values.to_dict()
        ci_unconstr = self._minuit.merrors

        # values of uninterested free parameters, in unconstrained space
        others = {k: v for k, v in mle_unconstr.items() if k not in names}

        # lower bound
        lower = self._helper.unconstr_dic_to_params_dic(
            {k: mle_unconstr[k] + ci_unconstr[k].lower for k in names} | others
        )

        # upper bound
        upper = self._helper.unconstr_dic_to_params_dic(
            {k: mle_unconstr[k] + ci_unconstr[k].upper for k in names} | others
        )

        interval = {k: (lower[k], upper[k]) for k in names}
        status = {
            k: {
                'valid': (v.lower_valid, v.upper_valid),
                'at_limit': (v.at_lower_limit, v.at_upper_limit),
                'at_max_fcn': (v.at_lower_max_fcn, v.at_upper_max_fcn),
                'new_min': (v.lower_new_min, v.upper_new_min),
            }
            for k, v in ci_unconstr.items()
            if k in names
        }
        return interval, status

    def _ci_composite(self, names: Iterable[str], cl: float | int):
        """Confidence intervals of function of free parameters.

        References
        ----------
        .. [1] Eq.24 of https://doi.org/10.1007/s11222-021-10012-y
        .. [2] https://github.com/vemomoto/vemomoto/blob/master/ci_rvm/ci_rvm/ci_rvm.py#L1455
        """

        def loss_factory(name, mle):
            """Factory to create loss function for composite parameter."""
            rtol = 1e-3
            atol = mle * rtol
            atol_inv = 1.0 / atol

            @jax.jit
            def _(x: np.ndarray):
                assert len(x) - 1 == len(free_params)
                unconstr_dic = dict(zip(free_params, x[1:]))
                value = helper.unconstr_dic_to_params_dic(unconstr_dic)[name]
                s = (value - x[0]) * atol_inv
                return helper.deviance_total(x[1:]) + s * s

            return _

        helper = self._helper
        free_params = helper.params_names['free']

        interval = {}
        error = {}
        status = {}
        for i in names:
            mle_i = self._mle[i][0]
            loss = loss_factory(i, mle_i)
            init = np.array([mle_i, *self._minuit.values], float)
            grad = jax.jit(jax.grad(loss))
            minuit = Minuit(loss, init, grad=grad)
            minuit.strategy = 1
            minuit.migrad()
            minuit.minos(0, cl=cl)
            ci = minuit.merrors[0]
            interval[i] = (mle_i + ci.lower, mle_i + ci.upper)
            error[i] = (ci.lower, ci.upper)
            status[i] = {
                'valid': (ci.lower_valid, ci.upper_valid),
                'at_limit': (ci.at_lower_limit, ci.at_upper_limit),
                'at_max_fcn': (ci.at_lower_max_fcn, ci.at_upper_max_fcn),
                'new_min': (ci.lower_new_min, ci.upper_new_min),
            }

        return interval, error, status

    def _ci_boot(self, names: Iterable[str], cl: float | int):
        """Bootstrap confidence interval."""
        cl = 1.0 - 2.0 * stats.norm.sf(cl) if cl >= 1.0 else cl
        boot = self._boot
        interval = {
            k: np.quantile(v, q=(0.5 - 0.5 * cl, 0.5 + 0.5 * cl)).tolist()
            for k, v in boot.params.items()
            if k in names
        }
        status = {'nboot': boot.n_valid, 'seed': boot.seed}
        return interval, status

    @property
    def mle(self) -> dict[str, tuple[float, float]]:
        """MLE and error of parameters."""
        return format_result(self._mle, self._helper.params_names['all'])

    @property
    def deviance(self) -> dict[str, float]:
        """Deviance of the model at MLE."""
        stat = self._deviance['group'] | {'total': self._deviance['total']}
        stat = {i: float(stat[i]) for i in (*self._helper.data_names, 'total')}
        return stat

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
        return self._minuit.fmin


class ConfidenceInterval(NamedTuple):
    """Confidence interval result."""

    mle: dict[str, float]
    """MLE of the model parameters."""

    intervals: dict[str, tuple[float, float]]
    """The confidence intervals."""

    errors: dict[str, tuple[float, float]]
    """The confidence intervals in error form."""

    cl: float
    """The confidence level."""

    method: str
    """Method used to calculate the confidence interval."""

    status: dict
    """Status of the calculation progress."""


class BootstrapResult(NamedTuple):
    """Parametric bootstrap result."""

    mle: dict
    """MLE of the model parameters."""

    data: dict
    """Simulation data based on MLE."""

    models: dict
    """Bootstrap models."""

    params: dict
    """Bootstrap parameters."""

    deviance: dict
    """Bootstrap deviance."""

    p_value: dict
    """Model fitness :math:`p`-value."""

    n: int
    """Numbers of bootstrap."""

    n_valid: int
    """Numbers of valid bootstrap."""

    seed: int
    """Seed of random number generator used in simulation."""


class PosteriorResult(FitResult):
    """Result obtained from Bayesian fit."""

    _idata: az.InferenceData
    _mle: dict | None = None
    _ppc: PPCResult | None = None
    _loo: az.stats.stats_utils.ELPDData | None = None
    _waic: az.stats.stats_utils.ELPDData | None = None
    _rhat: dict[str, float] | None = None
    _divergence: int | None = None

    def __init__(
        self, sampler: MCMC | NestedSampler, helper: Helper, fit: Fit
    ):
        if not isinstance(sampler, (MCMC, NestedSampler)):
            raise ValueError(f'unknown sampler type {type(sampler)}')

        super().__init__(helper)
        self._sampler = sampler
        self._fit = fit
        if isinstance(sampler, MCMC):
            self._init_from_numpyro(sampler)
        else:
            self._init_from_jaxns(sampler)

    def __repr__(self):
        super().__repr__()

    def _repr_html_(self):
        return self.__repr__()

    def ci(
        self,
        params: str | Iterable[str] | None = None,
        prob: float | int = 1,
        hdi: bool = False,
    ) -> CredibleInterval:
        """Calculate credible intervals.

        Parameters
        ----------
        params : str or sequence of str, optional
            Parameters to calculate confidence intervals. If not specified,
            calculate for parameters of interest.
        prob : float or int, optional
            The probability mass of samples within the credible interval. If
            0 < `prob` < 1, the value is interpreted as the probability mass.
            If `prob` >= 1, it is interpreted as the number of standard
            deviations. For example, ``prob=1`` produces a 1-sigma or 68.3%
            credible interval. The default is 1.
        hdi : bool, optional
            Whether to return the highest density interval. The default is
            False, which means an equal tailed interval is returned.

        Returns
        -------
        CredibleInterval
            The credible interval.
        """
        if prob <= 0.0:
            raise ValueError('prob must be non-negative')

        params = self._check_ci_params(params)

        prob_ = 1.0 - 2.0 * stats.norm.sf(prob) if prob >= 1.0 else prob

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
            median=format_result(median, params),
            interval=format_result(interval, params),
            error=format_result(error, params),
            prob=prob_,
            method='HDI' if hdi else 'ETI',
        )

    def ppc(
        self,
        n: int = 10000,
        seed: int | None = None,
        parallel: bool = True,
        progress: bool = True,
    ):
        """Perform posterior predictive check.

        Parameters
        ----------
        n : int, optional
            The number of posterior predictions. The default is 10000.
        seed : int, optional
            The seed of random number generator used in posterior predictions.
        parallel : bool, optional
            Whether to run simulation fit in parallel. The default is True.
        progress : bool, optional
            Whether to display progress bar. The default is True.
        """
        # reuse the previous result if all setup is the same
        if self._ppc and self._ppc.n == n and self._ppc.seed == seed:
            return

        if self._mle is None:
            self._mle = {}
            helper = self._helper

            # MLE information of the model
            free_params = helper.params_names['free']
            mle_idx = self._idata['log_likelihood']['total'].argmax(...)
            init = self._idata['posterior'][free_params].sel(**mle_idx)
            init = {k: v.values for k, v in init.data_vars.items()}
            init = helper.constr_dic_to_unconstr_arr(init)
            mle_unconstr = self._fit._optimize_lm(init)[0]

            # MLE of model params in constrained space
            mle, cov = helper.get_mle(mle_unconstr)
            err = jnp.sqrt(jnp.diagonal(cov))
            params_names = helper.params_names['all']
            self._mle['mle'] = dict(zip(params_names, zip(mle, err)))

            # model deviance at MLE
            self._mle['deviance'] = jax.jit(helper.deviance)(mle_unconstr)

        helper = self._helper
        free_params = helper.params_names['free']
        n = int(n)
        seed = helper.seed['pred'] if seed is None else int(seed)

        # randomly select n samples from posterior
        rng = np.random.default_rng(seed)
        idata = self._idata
        i = rng.integers(0, idata['posterior'].chain.size, n)
        j = rng.integers(0, idata['posterior'].draw.size, n)
        params = {
            k: v.values[i, j]
            for k, v in idata['posterior'][free_params].data_vars.items()
        }
        models = {
            k: v.values[i, j]
            for k, v in helper.get_models(idata['posterior']).items()
        }

        # perform ppc
        result = helper.simulate_and_fit(
            seed, params, models, 1, parallel, progress, 'PPC'
        )
        valid = result.pop('valid')
        result = jax.tree_map(lambda x: x[valid], result)

        self._ppc = PPCResult(
            params_rep=params,
            models_rep=models,
            data=result['data'],
            params_fit=result['params'],
            models_fit=result['models'],
            deviance=result['deviance'],
            p_value=jax.tree_map(
                lambda obs, sim: np.sum(sim >= obs, axis=0) / len(sim),
                self._mle['deviance'],
                result['deviance'],
            ),
            n=n,
            n_valid=np.sum(valid),
            seed=seed,
        )

    def _init_from_numpyro(self, sampler: MCMC):
        helper = self._helper
        params_names = helper.params_names
        samples = sampler.get_samples(group_by_chain=True)

        # stats of samples
        rename = {'num_steps': 'n_steps'}
        sample_stats = {}
        for k, v in sampler.get_extra_fields(group_by_chain=True).items():
            name = rename.get(k, k)
            value = jax.device_get(v).copy()
            sample_stats[name] = value
            if k == 'num_steps':
                sample_stats['tree_depth'] = np.log2(value).astype(int) + 1

        # attrs for each group of arviz.InferenceData
        attrs = {
            'elisa_version': __version__,
            'inference_library': 'numpyro',
            'inference_library_version': numpyro.__version__,
        }

        # create arviz.InferenceData
        self._generate_idata(samples, attrs, sample_stats)

        # effective sample size
        params = helper.get_params(samples)
        ess = az.ess(params)
        self._ess = {k: int(ess[k].values) for k in params.keys()}

        # relative mcmc efficiency
        # the calculation of reff is according to arviz loo:
        # https://github.com/arviz-devs/arviz/blob/main/arviz/stats/stats.py#L770
        if sampler.num_chains == 1:
            self._reff = 1.0
        else:
            # use only free parameters to calculate reff
            free = {k: params[k] for k in params_names['free']}
            reff_p = az.ess(free, method='mean', relative=True)
            self._reff = np.hstack(list(reff_p.data_vars.values())).mean()

        # model evidence
        self._lnZ = (None, None)

    def _init_from_jaxns(self, sampler: NestedSampler):
        helper = self._helper
        result = sampler._results

        # get posterior samples
        total = result.total_num_samples
        rng_key = jax.random.PRNGKey(helper.seed['mcmc'])
        samples = jax.tree_map(
            lambda x: x[None, ...],
            sampler.get_samples(rng_key, total),
        )

        # attrs for each group of arviz.InferenceData
        attrs = {
            'elisa_version': __version__,
            'inference_library': 'jaxns',
            'inference_library_version': metadata.version('jaxns'),
        }

        self._generate_idata(samples, attrs)

        # effective sample size
        self._ess = {'total': int(result.ESS)}
        # relative mcmc efficiency
        self._reff = float(result.ESS / result.total_num_samples)
        # model evidence
        self._lnZ = (float(result.log_Z_mean), float(result.log_Z_uncert))

    def _generate_idata(self, samples, attrs, sample_stats=None):
        samples = jax.tree_map(jax.device_get, samples)
        helper = self._helper

        params = helper.get_params(samples)
        models = helper.get_models(samples)
        posterior = params | models
        posterior_predictive = helper.simulate(helper.seed['pred'], models, 1)
        loglike = helper.get_loglike(samples)
        loglike = (
            loglike['data']
            | loglike['point']
            | loglike['group']
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

        # create InferenceData
        self._idata = az.from_dict(
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
        the early workflow, R-hat below 1.1 is often sufficient. See [1]_ for
        more information.

        References
        ----------
        .. [1] https://arxiv.org/abs/1903.08008
        """
        if self._rhat is None:
            params_names = self._helper.params_names['free']
            posterior = self._idata['posterior'][params_names]

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
                reff=self.reff,
                scale='deviance',
            )

        return self._loo

    @property
    def lnZ(self) -> tuple[float, float] | tuple[None, None]:
        """Log model evidence and uncertainty."""
        return self._lnZ


class CredibleInterval(NamedTuple):
    """Credible interval result."""

    median: dict[str, float]
    """Median of the model parameters."""

    interval: dict[str, tuple[float, float]]
    """The credible intervals."""

    error: dict[str, tuple[float, float]]
    """The credible intervals in error form."""

    prob: float
    """The probability mass."""

    method: str
    """Highest Density Interval (HDI), or equal tailed interval (ETI)."""


class PPCResult(NamedTuple):
    """Posterior predictive check result."""

    params_rep: dict
    """Posterior of free parameters used to perform ppc."""

    models_rep: dict
    """Models' values corresponding to `params_rep`."""

    data: dict
    """Posterior predictive data."""

    params_fit: dict
    """Best fit parameters of posterior predictive data."""

    deviance: dict
    """Deviance of posterior predictive data and best fit models."""

    models_fit: dict
    """Best fit models' values of posterior predictive data."""

    p_value: dict
    """Posterior predictive :math:`p`-value."""

    n: int
    """Numbers of posterior prediction."""

    n_valid: int
    """Numbers of valid ppc."""

    seed: int
    """Seed of random number generator used in simulation."""


def format_result(result: dict, order: Sequence[str]) -> dict:
    """Sort the result and use float type."""
    formatted = jax.tree_map(float, result)
    return {k: formatted[k] for k in order}
