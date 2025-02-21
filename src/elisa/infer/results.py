"""Subsequent analysis of maximum likelihood or Bayesian fit."""

from __future__ import annotations

import bz2
import gzip
import lzma
import warnings
from abc import ABC, abstractmethod
from importlib import metadata
from typing import TYPE_CHECKING, NamedTuple

import arviz as az
import astropy.units as u
import dill
import jax
import jax.numpy as jnp
import nautilus
import numpy as np
import numpyro
import scipy.stats as stats
import ultranest
from astropy.cosmology import Planck18
from iminuit import Minuit
from iminuit.util import Matrix as CovarMatrix
from jax.experimental.mesh_utils import create_device_mesh
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec
from numpyro.infer import MCMC

from elisa.__about__ import __version__
from elisa.infer.helper import check_params
from elisa.infer.nested_sampling import NestedSampler
from elisa.plot.plotter import MLEResultPlotter, PosteriorResultPlotter
from elisa.util.config import get_parallel_number
from elisa.util.misc import make_pretty_table

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from typing import Literal

    from arviz.stats.stats_utils import ELPDData
    from astropy.cosmology.flrw.lambdacdm import LambdaCDM
    from astropy.units import Quantity as Q
    from iminuit.util import FMin
    from xarray import DataArray

    from elisa.infer.fit import BayesFit
    from elisa.infer.helper import Helper
    from elisa.plot.plotter import Plotter
    from elisa.util.typing import JAXArray

ReactiveNestedSampler = ultranest.ReactiveNestedSampler
Sampler = nautilus.Sampler


class FitResult(ABC):
    """Fit result."""

    _helper: Helper
    _plotter: Plotter | None
    _flux_fn: Callable
    _lumin_fn: Callable
    _eiso_fn: Callable
    _n_parallel: int | None

    def __init__(self, helper: Helper):
        self._helper = helper
        self._n_parallel = None

        models = helper.model
        ne = {name: model.ne for name, model in models.items()}
        ene = {name: model.ene for name, model in models.items()}

        def _flux(
            egrid: JAXArray,
            params: dict[str, JAXArray],
            energy: bool,
            comps: bool,
        ) -> dict[str, JAXArray] | dict[str, dict[str, JAXArray]]:
            """Calculate flux."""
            if energy:
                fns = ene
            else:
                fns = ne
            de = jnp.diff(egrid)
            flux = {}
            for name, fn in fns.items():
                f = fn(egrid, params, comps)
                if comps:
                    flux[name] = jax.tree.map(
                        lambda v: jnp.sum(v * de, axis=-1), f
                    )
                else:
                    flux[name] = jnp.sum(f * de, axis=-1)
            return flux

        self._flux_fn = jax.jit(_flux, static_argnums=(2, 3))

        # some data has the same model, record the unique models to
        # avoid redundant computation in flux, lumin, and eiso
        mapping = {}
        for data_name, model in models.items():
            if model not in mapping:
                mapping[model] = [data_name]
            else:
                mapping[model].append(data_name)
        self._model_mapping = {d: v[0] for v in mapping.values() for d in v}

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def _repr_html_(self):
        pass

    @property
    @abstractmethod
    def plot(self) -> Plotter:
        """Result plotter."""
        pass

    def summary(self, file=None) -> None:
        """Print the summary of fit result.

        Parameters
        ----------
        file: file-like
            An object with a ``write(string)`` method. This is passed to
            :py:func:`print`.
        """
        print(repr(self), file=file)

    @abstractmethod
    def flux(
        self, *args, **kwargs
    ) -> dict[str, jax.Array] | dict[str, dict[str, jax.Array]]:
        pass

    @abstractmethod
    def lumin(
        self, *args, **kwargs
    ) -> dict[str, jax.Array] | dict[str, dict[str, jax.Array]]:
        pass

    @abstractmethod
    def eiso(
        self, *args, **kwargs
    ) -> dict[str, jax.Array] | dict[str, dict[str, jax.Array]]:
        pass

    @property
    def ndata(self) -> dict[str, int]:
        """Data points number."""
        return self._helper.ndata

    @property
    def dof(self) -> int:
        """Degree of freedom."""
        return self._helper.dof

    @property
    @abstractmethod
    def gof(self) -> dict[str, float]:
        """Goodness of fit p-value."""
        pass

    @property
    @abstractmethod
    def _params_dist(self) -> dict[str, JAXArray]:
        pass

    def save(
        self,
        path: str,
        compress: Literal['gzip', 'bz2', 'lzma'] = 'gzip',
    ) -> None:
        """Save the fit result to a file.

        Parameters
        ----------
        path : str
            The file path to save fit result.
        compress : {'gzip', 'bz2', 'lzma'}
            The compression algorithm to use.
        """
        if compress == 'gzip':
            open_ = gzip.open
        elif compress == 'bz2':
            open_ = bz2.open
        elif compress == 'lzma':
            open_ = lzma.open
        else:
            raise ValueError(f'unsupported compression algorithm {compress}')

        with open_(path, 'wb') as f:
            dill.dump(self, f)

    @staticmethod
    def load(
        path: str,
        decompress: Literal['gzip', 'bz2', 'lzma'] = 'gzip',
    ) -> FitResult:
        """Load a previously saved fit result.

        Parameters
        ----------
        path : str
            The file path of previously saved fit result.
        decompress : {'gzip', 'bz2', 'lzma'}
            The decompression algorithm used to load the fit result.

        Returns
        -------
        FitResult
            The loaded fit result.
        """
        if decompress == 'gzip':
            open_ = gzip.open
        elif decompress == 'bz2':
            open_ = bz2.open
        elif decompress == 'lzma':
            open_ = lzma.open
        else:
            raise ValueError(
                f'unsupported decompression algorithm {decompress}'
            )

        with open_(path, 'rb') as f:
            return dill.load(f)

    def _to_unit_cl(self, cl: int | float):
        """Convert cl into unit."""
        if cl <= 0:
            raise ValueError('cl must be positve')
        elif cl < 1:
            return cl
        else:
            return 1.0 - 2.0 * stats.norm.sf(cl)

    def _check_fn(self, fn: dict[str, Callable] | None):
        """Check user provided function."""
        if fn is None:
            return {}
        else:
            helper = self._helper
            fn = {str(k): v for k, v in fn.items()}

            if fn.keys() & helper.params_default.keys():
                raise ValueError(
                    'names of fn must not overlap with model parameters'
                )

            fn_checked = {}
            if not isinstance(fn, dict):
                raise TypeError('fn must be dict of functions')
            for k, v in fn.items():
                msg = (
                    f"fn['{k}'] must be a function of these parameters: "
                    f'{", ".join(helper.params_names["all"])}'
                )
                if not callable(v):
                    raise TypeError(msg)
                else:
                    try:
                        jitted = jax.jit(v)
                        fn_result = jitted(helper.params_default)
                        fn_checked[str(k)] = jitted
                    except Exception as e:
                        raise ValueError(msg) from e

                    if np.shape(fn_result) != ():
                        raise ValueError(
                            f"fn['{k}'] must return a scalar value"
                        )

            return fn_checked


class MLEResult(FitResult):
    """Result of maximum likelihood fit."""

    _plotter: MLEResultPlotter | None = None

    def __init__(self, minuit: Minuit, helper: Helper):
        super().__init__(helper)

        self._minuit = minuit

        self._mle_unconstr = jnp.array(minuit.values, float)
        mle, covar = helper.get_mle(self._mle_unconstr)

        if np.allclose(covar, covar.T):
            try:
                np.linalg.cholesky(covar)
                pos_def = True
            except np.linalg.LinAlgError:
                pos_def = False
        else:
            pos_def = False

        if not pos_def and minuit.covariance is not None:
            covar_unconstr = jnp.array(minuit.covariance, float)
            covar = helper.params_covar(self._mle_unconstr, covar_unconstr)

        var2pos = dict(zip(helper.params_names['all'], range(len(mle))))
        self._covar = CovarMatrix(var2pos)
        self._covar[:] = covar

        err = jnp.sqrt(jnp.diagonal(covar))

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
        tabs = self._tabs()
        return (
            f'Parameters\n{tabs["params"]}\n\n'
            f'Fit Statistics\n{tabs["stat"]}\n\n'
            f'Information Criterion\n{tabs["ic"]}\n\n'
            f'Fit Status\n{self.status}\n'
        )

    def _repr_html_(self):
        """The repr in Jupyter notebook environment."""
        tabs = self._tabs()
        params_tab = tabs['params'].get_html_string(format=True)
        stat_tab = tabs['stat'].get_html_string(format=True)
        ic_tab = tabs['ic'].get_html_string(format=True)
        status_tab = self.status._repr_html_()
        return (
            '<details open><summary><b>MLE Result</b></summary>'
            '<details open style="padding-left: 1em">'
            f'<summary><b>Parameters</b></summary>{params_tab}</details>'
            '<details open style="padding-left: 1em">'
            f'<summary><b>Fit Statistics</b></summary>{stat_tab}</details>'
            f'<details open style="padding-left: 1em">'
            '<summary><b>Information Criterion</b></summary>'
            f'{ic_tab}</details>'
            '<details style="padding-left: 1em">'
            f'<summary><b>Fit Status</b></summary>{status_tab}</details>'
            '</details>'
        )

    def _tabs(self):
        params_tab = make_pretty_table(
            ['Parameter', 'MLE', 'Error'],
            [(k, f'{v[0]:.4g}', f'{v[1]:.4g}') for k, v in self.mle.items()],
        )
        stat_type = self._helper.statistic
        deviance = self.deviance
        ndata = self.ndata
        rows = [
            [i, f'{stat_type[i]}', f'{deviance[i]:.2f}', ndata[i]]
            for i in self.ndata.keys()
            if i != 'total'
        ]
        rows.append(
            [
                'Total',
                'stat/dof',
                f'{deviance["total"]:.2f}/{self.dof}',
                ndata['total'],
            ]
        )
        names = ['Data', 'Statistic', 'Value', 'Channels']
        stat_tab = make_pretty_table(names, rows)

        rows = [['AIC', f'{self.aic:.2f}'], ['BIC', f'{self.bic:.2f}']]
        names = ['Method', 'Value']
        ic_tab = make_pretty_table(names, rows)
        return {'params': params_tab, 'stat': stat_tab, 'ic': ic_tab}

    @property
    def plot(self) -> MLEResultPlotter:
        if self._plotter is None:
            self._plotter = MLEResultPlotter(self)
        return self._plotter

    def boot(
        self,
        n: int = 10000,
        seed: int | None = None,
        parallel: bool = True,
        n_parallel: int | None = None,
        progress: bool = True,
        update_rate: int = 50,
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
        n_parallel : int, optional
            Number of parallel processes to use when `parallel` is ``True``.
            Defaults to ``jax.local_device_count()``.
        progress : bool, optional
            Whether to display progress bar. The default is True.
        update_rate : int, optional
            The update rate of progress bar. The default is 50.
        """
        n = int(n)
        n_parallel = get_parallel_number(n_parallel)
        if parallel and (n % n_parallel):
            n += n_parallel - n % n_parallel

        # reuse the previous result if all setup is the same
        if self._boot and self._boot.n == n and self._boot.seed == seed:
            return

        helper = self._helper
        seed = helper.seed['pred'] if seed is None else int(seed)
        params = {i: self._mle[i][0] for i in helper.params_names['free']}
        models = self._model_values

        # perform parametric bootstrap
        result = helper.simulate_and_fit(
            seed,
            params,
            models,
            n,
            parallel,
            n_parallel,
            progress,
            update_rate,
            'Bootstrap',
        )
        valid = result.pop('valid')
        result = jax.tree.map(lambda x: x[valid], result)

        self._boot = BootstrapResult(
            mle={k: v[0] for k, v in self.mle.items()},
            data=result['data'],
            models=result['models'],
            params=result['params'],
            deviance=result['deviance'],
            p_value=jax.tree.map(
                lambda obs, sim: np.sum(sim >= obs, axis=0) / len(sim),
                self._deviance,
                result['deviance'],
            ),
            n=n,
            n_valid=np.sum(valid),
            seed=seed,
        )

    @property
    def _params_dist(self) -> dict[str, jax.Array] | None:
        """Bootstrapped parameter distribution."""
        boot = self._boot
        if boot is None:
            return None
        n = boot.n_valid - boot.n_valid % jax.local_device_count()
        return {k: v[:n] for k, v in boot.params.items()}

    def covar(
        self,
        params: str | Sequence[str] | None = None,
        fn: dict[str, Callable] | None = None,
        method: Literal['hess', 'boot'] = 'hess',
        parallel: bool = True,
    ) -> ParamsCovar:
        """Calculate covariance matrix.

        Parameters
        ----------
        params : str or sequence of str, optional
            Parameters to calculate covariance matrix. If not specified,
            calculate for parameters of interest.
        fn : dict
            A dict containing functions to calculate the covariance matrix.
            The keys are the names of the function results, and the values are
            the functions whose input is a dict of model parameters.
        method : {'hess', 'boot'}, optional
            Method used to calculate covariance. Available options are:

                * ``'hess'``: inverse of Hessian matrix from Minuit
                * ``'boot'``: calculate covariance based on bootstrap samples,
                  :meth:`MLEResult.boot` must be called before using this
                  method.

            The default is ``'hess'``.
        parallel : bool, optional
            Whether to evaluate `fn` in parallel when `method` is ``'boot'``.
            The default is True.

        Returns
        -------
        ParamsCovar
            The covariance matrix.
        """
        params_mle = {k: v[0] for k, v in self.mle.items()}
        params = check_params(params, self._helper)
        fn = self._check_fn(fn)

        if method == 'hess':
            if not fn:
                covar = np.array(self._covar)
            else:

                @jax.jit
                @jax.jacobian
                @jax.jit
                def jacobian(params_arr):
                    params_dic = dict(zip(params_mle.keys(), params_arr))
                    fn_arr = jnp.array([f(params_dic) for f in fn.values()])
                    return jnp.hstack([params_arr, fn_arr])

                jac = np.array(jacobian(jnp.array(list(params_mle.values()))))
                old_covar = np.array(self._covar)
                covar = jac @ old_covar @ jac.T

        elif method == 'boot':
            self._raise_if_no_boot()

            @jax.jit
            @jax.vmap
            @jax.jit
            def eval_fn(params):
                params_arr = jnp.array([params[k] for k in params_mle])
                fn_arr = jnp.array([f(params) for f in fn.values()])
                return jnp.hstack([params_arr, fn_arr])

            if parallel:
                n_parallel = get_parallel_number(self._n_parallel)
                devices = create_device_mesh(
                    mesh_shape=(n_parallel,),
                    devices=jax.devices()[:n_parallel],
                )
                mesh = Mesh(devices, axis_names=('i',))
                pi = PartitionSpec('i')
                eval_fn = shard_map(
                    f=eval_fn,
                    mesh=mesh,
                    in_specs=(pi,),
                    out_specs=pi,
                    check_rep=False,
                )

            samples = eval_fn(self._params_dist)
            covar = np.cov(samples, rowvar=False)

        else:
            raise ValueError("method must be either 'hess' or 'boot'")

        names = tuple(params) + tuple(fn.keys())
        var2pos = dict(zip(names, range(len(names))))
        matrix = CovarMatrix(var2pos)
        mask = np.array(
            [p in params for p in params_mle] + [True] * len(fn),
            dtype=bool,
        )
        matrix[:] = np.array(covar)[:, mask][mask]
        return ParamsCovar(names=names, matrix=matrix)

    def ci(
        self,
        cl: float | int = 1,
        params: str | Iterable[str] | None = None,
        fn: dict[str, Callable] | None = None,
        method: Literal['profile', 'boot'] = 'profile',
        rtol: float | dict[str, float] = 1e-6,
        parallel: bool = True,
    ) -> ConfidenceInterval:
        """Calculate confidence intervals.

        Parameters
        ----------
        cl : float or int, optional
            Confidence level for the confidence interval. If 0 < `cl` < 1, the
            value is interpreted as the confidence level. If `cl` >= 1, it is
            interpreted as the number of standard deviations. For example,
            ``cl=1`` produces a 1-sigma or 68.3% confidence interval.
            The default is 1.
        params : str or sequence of str, optional
            Parameters to calculate confidence intervals. If not specified,
            calculate for parameters of interest.
        fn : dict, optional
            A dict containing functions to calculate the confidence intervals.
            The keys are the names of the function results, and the values are
            the functions whose input is a dict of model parameters.
        method : {'profile', 'boot'}, optional
            Method used to calculate confidence. Available options are:

                * ``'profile'``: use Minos algorithm of Minuit to find the
                  confidence intervals based on the profile likelihood
                * ``'boot'``: use parametric bootstrap method to calculate
                  the confidence intervals. :meth:`MLEResult.boot` must be
                  called before using this method.

            The default is ``'profile'``.
        rtol : float, or dict of float, optional
            The relative tolerance in determining the value of composite
            parameters and `fn` when `method` is ``'profile'``.
            The default is 1e-6.
        parallel : bool, optional
            Whether to evaluate `fn` in parallel when `method` is ``'boot'``.
            The default is True.

        Returns
        -------
        ConfidenceInterval
            The confidence intervals.
        """
        cl = self._to_unit_cl(cl)
        params = check_params(params, self._helper)
        params_set = set(params)
        free = params_set.intersection(self._helper.params_names['free'])
        composite = params_set.intersection(
            self._helper.params_names['deterministic']
        )
        assert free | composite == params_set

        fn = self._check_fn(fn)
        rtol_keys = tuple(fn.keys()) + tuple(composite)
        if isinstance(rtol, float):
            rtol = {k: rtol for k in rtol_keys}
        else:
            rtol = jax.tree.map(float, dict(rtol))
            for k in rtol_keys:
                rtol.setdefault(k, 1e-6)
        if np.any([i > 0.01 for i in rtol.values()]):
            raise ValueError('rtol must be less than 0.01')

        if method == 'profile':
            self._warn_invalid_fit()
            if not self._minuit.valid:
                intervals, status = self._ci_invalid(params_set | fn.keys())
            else:
                res1 = self._ci_free(free, cl) if free else ({}, {})

                if composite:

                    def factory(k):
                        def _(p):
                            return p[k]

                        return _

                    fn_composite = {k: factory(k) for k in composite}
                    res2 = self._ci_fn(fn_composite, cl, rtol)
                else:
                    res2 = ({}, {})

                res3 = self._ci_fn(fn, cl, rtol) if fn else ({}, {})

                intervals = res1[0] | res2[0] | res3[0]
                status = res1[1] | res2[1] | res3[1]

        elif method == 'boot':
            intervals, status = self._ci_boot(cl, params, fn, parallel)

        else:
            raise ValueError("method must be either 'profile' or 'boot'")

        params_mle = {k: v[0] for k, v in self._mle.items()}
        vars_names = params + list(fn.keys())
        vars_mle = {k: v for k, v in params_mle.items() if k in vars_names}
        vars_mle |= {k: v(params_mle) for k, v in fn.items()}
        errors = {
            k: (intervals[k][0] - vars_mle[k], intervals[k][1] - vars_mle[k])
            for k in vars_names
        }

        return ConfidenceInterval(
            mle=_format_result(vars_mle, vars_names),
            intervals=_format_result(intervals, vars_names),
            errors=_format_result(errors, vars_names),
            cl=1.0 - 2.0 * stats.norm.sf(cl) if cl >= 1.0 else cl,
            method=method,
            status=status,
        )

    def _ci_invalid(self, names: Iterable[str]):
        """Confidence interval of invalid fit."""
        interval = {k: (float('nan'), float('nan')) for k in names}
        status = {
            k: {
                'valid': (False, False),
                'at_limit': (False, False),
                'at_max_fcn': (False, False),
                'new_min': (False, False),
            }
            for k in names
        }
        return interval, status

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

    def _ci_fn(
        self,
        fn: dict[str, Callable],
        cl: float | int,
        rtol: dict[str, float],
    ):
        """Confidence intervals of function of free parameters."""
        params_mle = {k: v[0] for k, v in self._mle.items()}
        fn_mle = {k: v(params_mle) for k, v in fn.items()}

        def get_minuit(name, mle, r) -> Minuit:
            loss = self._loss_factory(fn[name], r)
            grad = jax.jit(jax.grad(loss))
            init = np.hstack([mle, self._minuit.values])
            minuit = Minuit(loss, init, grad=grad)
            minuit.strategy = 2
            minuit.migrad()
            return minuit

        def get_minuit_iter_rtol(name, mle) -> Minuit:
            rtol_desired = rtol[name]
            minuit0 = get_minuit(name, mle, rtol_desired)
            if not minuit0.accurate:
                # When profiling the likelihood, deviance difference for
                # 1-sigma confidence interval is 1, thus the deviance
                # varies slow within 1-sigma confidence interval.
                # The rtol should be less than 1% of the variance.
                fn_var = self.covar(params=(), fn={'_': fn[name]}).matrix[0, 0]
                rel_err = np.sqrt(fn_var) / np.abs(mle)
                rtol_max = np.min([0.01, 0.01 * rel_err, 100 * rtol_desired])
                if rtol_desired < rtol_max:
                    for r in np.geomspace(rtol_desired, rtol_max, num=15)[1:]:
                        minuit = get_minuit(name, mle, r)
                        if minuit.accurate:
                            return minuit
            return minuit0

        interval = {}
        status = {}
        for name, mle in fn_mle.items():
            minuit = get_minuit_iter_rtol(name, mle)
            minuit.minos(0, cl=cl)
            ci = minuit.merrors[0]
            interval[name] = (mle + ci.lower, mle + ci.upper)
            status[name] = {
                'valid': (ci.lower_valid, ci.upper_valid),
                'at_limit': (ci.at_lower_limit, ci.at_upper_limit),
                'at_max_fcn': (ci.at_lower_max_fcn, ci.at_upper_max_fcn),
                'new_min': (ci.lower_new_min, ci.upper_new_min),
            }

        return interval, status

    def _loss_factory(self, fn: Callable, rtol: float):
        """Factory method to create joint loss of params and func of params.

        Parameters
        ----------
        fn : Callable
            Function accepts model parameters and outputs a single value.
        rtol : float
            Relative tolerance of the function value.

        References
        ----------
        .. [1] Eq.24 of https://doi.org/10.1007/s11222-021-10012-y
        .. [2] https://github.com/vemomoto/vemomoto/blob/master/ci_rvm/ci_rvm/ci_rvm.py#L1455
        """
        helper = self._helper
        params_free = helper.params_names['free']

        @jax.jit
        def loss(x: np.ndarray):
            """Joint loss of params and func of params."""
            unconstr_dic = dict(zip(params_free, x[1:]))
            params = helper.unconstr_dic_to_params_dic(unconstr_dic)
            fn_value = fn(params)
            s1 = fn_value / x[0] - 1.0
            s2 = (s1 * s1) / rtol
            return helper.deviance_total(x[1:]) + s2

        return loss

    def _ci_boot(
        self,
        cl: float | int,
        params: Iterable[str],
        fn: dict[str, Callable] | None = None,
        parallel: bool = True,
        params_setting: dict[str, JAXArray] | None = None,
    ):
        """Bootstrap confidence interval."""
        self._raise_if_no_boot()

        if params_setting is not None:
            params_setting = dict(params_setting)
        else:
            params_setting = {}
        boot = self._boot
        boot_params = self._params_dist
        nboot = len(list(boot_params.values())[0])

        cl = self._to_unit_cl(cl)
        q = (0.5 - 0.5 * cl, 0.5 + 0.5 * cl)

        interval = {
            k: np.quantile(v, q=q).tolist()
            for k, v in boot_params.items()
            if k in params
        }
        status = {
            'nboot': nboot,
            'seed': int(boot.seed),
            'dist': jax.tree.map(  # get a copy of the distribution
                lambda x: x.copy(), jax.device_get(boot_params)
            ),
            'params_setting': params_setting,
        }

        if fn is not None and fn:
            eval_fn = jax.jit(
                jax.vmap(lambda p: jax.tree.map(lambda f: f(p), fn))
            )

            if parallel:
                n_parallel = get_parallel_number(self._n_parallel)
                devices = create_device_mesh(
                    mesh_shape=(n_parallel,),
                    devices=jax.devices()[:n_parallel],
                )
                mesh = Mesh(devices, axis_names=('i',))
                pi = PartitionSpec('i')
                eval_fn = shard_map(
                    f=eval_fn,
                    mesh=mesh,
                    in_specs=(pi,),
                    out_specs=pi,
                    check_rep=False,
                )

            if params_setting:
                params_setting = {
                    k: jnp.full(nboot, v) for k, v in params_setting.items()
                }

            fn_values = jax.device_get(eval_fn(boot_params | params_setting))
            status['dist'] |= fn_values

            interval |= {
                k: np.quantile(v, q).tolist() for k, v in fn_values.items()
            }

        return interval, status

    def _warn_invalid_fit(self):
        if not self._minuit.valid:
            warnings.warn('fit must be valid to calculate confidence interval')

    def _raise_if_no_boot(self):
        if self._boot is None:
            raise RuntimeError(
                'before using the bootstrap method, '
                'MLEResult.boot(...) must be called'
            )

    def _intensity_ci(
        self,
        egrid: JAXArray,
        energy: bool,
        cl: float | int,
        converter: Callable,
        method: Literal['profile', 'boot'],
        comps: bool,
        params: dict[str, JAXArray] | None,
    ) -> dict[str, Q | float]:
        """Calculate confidence interval of flux.

        Parameters
        ----------
        egrid : array-like
            Energy grid used in trapezoidal rule.
        energy : bool
            Whether the intensity is based on energy flux.
        cl : float or int
            Confidence level.
        converter : callable
            Function to convert the flux into desired intensity.
        method : {'profile', 'boot'}
            Method used to calculate confidence interval.
        comps : bool
            Whether to return the result of each component.
        params : dict, optional
            Parameters dict to overwrite the bootstrap parameters.

        Returns
        -------
        dict
            The confidence interval of intensity.
        """
        cl = self._to_unit_cl(cl)

        mle_params = {k: v[0] for k, v in self._mle.items()}
        if params is not None:
            mle_params |= params
        mle_flux = self._flux_fn(egrid, mle_params, energy, comps)

        mapping = self._model_mapping
        fn = jax.jit(lambda p: self._flux_fn(egrid, p, energy, comps))
        if comps:

            def factory(d, c):
                def _(p):
                    return fn(p)[d][c]

                return _

            fn_dic = {
                f'{d}_{c}': factory(d, c)
                for d in mapping.values()
                for c in mle_flux[d].keys()
            }
        else:

            def factory(d):
                def _(p):
                    return fn(p)[d]

                return _

            fn_dic = {d: factory(d) for d in mapping.values()}

        if method == 'profile':
            if params is not None:
                warnings.warn('params is ignored when using profile method')

            # use log transform to stabilize the profile likelihood
            def transform(f):
                def _(p):
                    return jnp.log(f(p))

                return _

            fn_dic = jax.tree.map(transform, fn_dic)
            rtol = {k: 1e-8 for k in fn_dic.keys()}
            intervals, status = self._ci_fn(fn_dic, cl, rtol=rtol)
            intervals = jax.tree.map(jnp.exp, intervals)
        elif method == 'boot':
            intervals, status = self._ci_boot(cl, [], fn_dic, True, params)
        else:
            raise ValueError("method must be either 'profile' or 'boot'")

        if comps:
            intervals = {
                k: {c: intervals[f'{k}_{c}'] for c in mle_flux[k].keys()}
                for k in mapping.values()
            }

            if method == 'profile':
                status = {
                    k: {c: status[f'{k}_{c}'] for c in mle_flux[k].keys()}
                    for k in mapping.values()
                }
            else:
                dist = status['dist']
                status['dist'] = {
                    k: {c: dist[f'{k}_{c}'] for c in mle_flux[k].keys()}
                    for k in mapping.values()
                }

        if energy:
            unit = u.Unit('erg cm^-2 s^-1')
        else:
            unit = u.Unit('ph cm^-2 s^-1')
        convert = lambda x: (x if x is None else converter(x * unit))

        intervals = {k: intervals[v] for k, v in mapping.items()}
        errors = jax.tree.map(
            lambda x, y: (y[0] - x, y[1] - x), mle_flux, intervals
        )

        if method == 'profile':
            status = {k: status[v] for k, v in mapping.items()}
        else:
            status['dist'] = jax.tree.map(convert, status['dist'])

        return {
            'mle': jax.tree.map(convert, mle_flux),
            'intervals': jax.tree.map(convert, intervals),
            'errors': jax.tree.map(convert, errors),
            'cl': cl,
            'status': status,
        }

    def flux(
        self,
        emin: float | int,
        emax: float | int,
        energy: bool = True,
        cl: float | int = 1,
        method: Literal['profile', 'boot'] = 'profile',
        ngrid: int = 1000,
        comps: bool = False,
        log: bool = True,
        params: dict[str, float | int] | None = None,
    ) -> MLEFlux:
        r"""Calculate the flux of model.

        .. warning::
            The flux is calculated by trapezoidal rule, and is accurate only
            if enough numbers of energy grids are used.

        Parameters
        ----------
        emin : float or int
            Minimum value of energy range, in units of keV.
        emax : float or int
            Maximum value of energy range, in units of keV.
        energy : bool, optional
            When True, calculate energy flux in units of erg cm⁻² s⁻¹;
            otherwise calculate photon flux in units of ph cm⁻² s⁻¹.
            The default is True.
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
                * ``'boot'``: use parametric bootstrap method to calculate the
                  confidence intervals. :meth:`MLEResult.boot` must be called
                  before using this method.

            The default is ``'profile'``.
        ngrid : int, optional
            The energy grid number to use in integration. The default is 1000.

        Other Parameters
        ----------------
        comps : bool, optional
            Whether to return the result of each component. The default is
            False.
        log : bool, optional
            Whether to use logarithmically regular energy grid. The default is
            True.
        params : dict, optional
            Parameters dict to overwrite the fitted parameters.

        Returns
        -------
        MLEFlux
            The flux of the model.
        """
        if log:
            egrid = jnp.geomspace(emin, emax, ngrid)
        else:
            egrid = jnp.linspace(emin, emax, ngrid)

        converter = lambda x: x
        flux = self._intensity_ci(
            egrid, energy, cl, converter, method, comps, params
        )

        return MLEFlux(emin, emax, bool(energy), **flux, method=method)

    def lumin(
        self,
        emin_rest: float | int,
        emax_rest: float | int,
        z: float | int,
        cl: float | int = 1,
        method: Literal['profile', 'boot'] = 'profile',
        ngrid: int = 1000,
        comps: bool = False,
        log: bool = True,
        params: dict[str, float | int] | None = None,
        cosmo: LambdaCDM = Planck18,
    ) -> MLELumin:
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
                * ``'boot'``: use parametric bootstrap method to calculate the
                  confidence intervals. :meth:`MLEResult.boot` must be called
                  before using this method.

            The default is ``'profile'``.
        ngrid : int, optional
            The energy grid number to use in integration. The default is 1000.

        Other Parameters
        ----------------
        comps : bool, optional
            Whether to return the result of each component. The default is
            False.
        log : bool, optional
            Whether to use logarithmically regular energy grid. The default is
            True.
        params : dict, optional
            Parameters dict to overwrite the fitted parameters.
        cosmo : LambdaCDM, optional
            Cosmology model used to calculate luminosity. The default is
            Planck18.

        Returns
        -------
        MLELumin
            The luminosity of the model.
        """
        if log:
            egrid = jnp.geomspace(emin_rest, emax_rest, ngrid) / (1.0 + z)
        else:
            egrid = jnp.linspace(emin_rest, emax_rest, ngrid) / (1.0 + z)

        factor = 4.0 * np.pi * cosmo.luminosity_distance(z) ** 2
        to_lumin = lambda x: (x * factor).to('erg s^-1')
        lumin = self._intensity_ci(
            egrid, True, cl, to_lumin, method, comps, params
        )

        return MLELumin(emin_rest, emax_rest, z, cosmo, **lumin, method=method)

    def eiso(
        self,
        emin_rest: float | int,
        emax_rest: float | int,
        z: float | int,
        duration: float | int,
        cl: float | int = 1,
        method: Literal['profile', 'boot'] = 'profile',
        ngrid: int = 1000,
        comps: bool = False,
        log: bool = True,
        params: dict[str, float | int] | None = None,
        cosmo: LambdaCDM = Planck18,
    ) -> MLEEiso:
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
            Observed duration of the source, in units of seconds.
        cl : float or int, optional
            Confidence level for the confidence interval. If 0 < `cl` < 1,
            the value is interpreted as the confidence level. If `cl` >= 1,
            it is interpreted as the number of standard deviations.
            For example, ``cl=1`` produces a 1-sigma or 68.3% confidence
            interval. The default is 1.
        method : {'profile', 'boot'}, optional
            Method used to calculate confidence. Available options are:

                * ``'profile'``: use Minos algorithm of Minuit to find the
                  confidence intervals based on the profile likelihood
                * ``'boot'``: use parametric bootstrap method to calculate the
                  confidence intervals. :meth:`MLEResult.boot` must be called
                  before using this method

            The default is ``'profile'``.
        ngrid : int, optional
            The energy grid number to use in integration. The default is
            1000.

        Other Parameters
        ----------------
        comps : bool, optional
            Whether to return the result of each component. The default is
            False.
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
        MLEEiso
            The isotropic emission energy of the model.
        """
        if log:
            egrid = jnp.geomspace(emin_rest, emax_rest, ngrid) / (1.0 + z)
        else:
            egrid = jnp.linspace(emin_rest, emax_rest, ngrid) / (1.0 + z)

        # This includes correction for energy redshift and time dilation.
        factor = 4.0 * np.pi * cosmo.luminosity_distance(z) ** 2
        factor *= duration / (1 + z) * u.s
        to_eiso = lambda x: (x * factor).to('erg')

        eiso = self._intensity_ci(
            egrid, True, cl, to_eiso, method, comps, params
        )

        return MLEEiso(
            emin_rest, emax_rest, z, duration, cosmo, **eiso, method=method
        )

    @property
    def gof(self) -> dict[str, float]:
        if self._boot is None:
            raise RuntimeError('MLEResult.boot() must be called to assess gof')
        p_value = self._boot.p_value
        p_value = p_value['group'] | {'total': p_value['total']}
        return {k: float(p_value[k]) for k in self.ndata.keys()}

    @property
    def mle(self) -> dict[str, tuple[float, float]]:
        """MLE and error of parameters."""
        return _format_result(self._mle, self._helper.params_names['all'])

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


class PosteriorResult(FitResult):
    """Result obtained from Bayesian fit."""

    _plotter: PosteriorResultPlotter | None = None
    _idata: az.InferenceData
    _deviance: dict | None = None
    _mle_result: dict | None = None
    _ppc: PPCResult | None = None
    _psislw_: DataArray | None = None
    _loo: az.stats.stats_utils.ELPDData | None = None
    _waic: az.stats.stats_utils.ELPDData | None = None
    _rhat: dict[str, float] | None = None
    _divergence: int | None = None
    _pit: dict[str, tuple] | None = None
    _params: dict[str, JAXArray] | None = None
    _info_tabs: dict | None = None

    def __init__(
        self,
        sampler: MCMC | NestedSampler | ReactiveNestedSampler | Sampler,
        helper: Helper,
        fit: BayesFit,
    ):
        if not isinstance(
            sampler, (MCMC, NestedSampler, ReactiveNestedSampler, Sampler)
        ):
            raise ValueError(f'unknown sampler type {type(sampler)}')

        super().__init__(helper)
        self._fit = fit
        if isinstance(sampler, MCMC):
            self._init_from_numpyro(sampler)
        elif isinstance(sampler, NestedSampler):
            self._init_from_jaxns(sampler)
        elif isinstance(sampler, ReactiveNestedSampler):
            self._init_from_ultranest(sampler)
        else:
            self._init_from_nautilus(sampler)

    def __repr__(self):
        tabs = self._tabs()
        return (
            f'Parameters\n{tabs["params"]}\n\n'
            f'Fit Statistics\n{tabs["stat"]}\n\n'
            f'Information Criterion\n{tabs["ic"]}\n\n'
            f'Pareto k diagnostic\n{tabs["k"]}\n'
        )

    def _repr_html_(self):
        """The repr in Jupyter notebook environment."""
        tabs = self._tabs()
        params_tab = tabs['params'].get_html_string(format=True)
        stat_tab = tabs['stat'].get_html_string(format=True)
        ic_tab = tabs['ic'].get_html_string(format=True)
        k_tab = tabs['k'].get_html_string(format=True)
        return (
            '<details open><summary><b>Posterior Result</b></summary>'
            '<details open style="padding-left: 1em">'
            f'<summary><b>Parameters</b></summary>{params_tab}</details>'
            '<details open style="padding-left: 1em">'
            f'<summary><b>Statistics</b></summary>{stat_tab}</details>'
            '<details open style="padding-left: 1em">'
            '<summary><b>Information Criterion</b></summary>'
            f'{ic_tab}</details>'
            '<details open style="padding-left: 1em">'
            f'<summary><b>Pareto k diagnostic</b></summary>{k_tab}</details>'
            '</details>'
        )

    def _tabs(self):
        if self._info_tabs is not None:
            return self._info_tabs
        params_name = self._helper.params_names['all']
        params = self.idata['posterior'][params_name]
        mean = params.mean()
        std = params.std(ddof=1)
        median = params.median()
        ci = params.quantile(0.5 + 0.683 * np.array([-0.5, 0.5])) - median
        ess = self.ess
        rhat = self.rhat
        rows = [
            [
                k,
                f'{mean[k]:.3g}',
                f'{std[k]:.3g}',
                f'{median[k]:.3g}',
                f'[{ci[k][0]:.3g}, {ci[k][1]:.3g}]',
                f'{ess[k]}',
                f'{rhat[k]:.2f}' if not np.isnan(rhat[k]) else 'N/A',
            ]
            for k in params_name
        ]
        names = [
            'Parameter',
            'Mean',
            'StdDev',
            'Median',
            '68.3% Quantile',
            'ESS',
            'Rhat',
        ]
        params_tab = make_pretty_table(names, rows)

        stat_type = self._helper.statistic
        deviance = self.deviance
        rows = [
            [
                i,
                stat_type[i],
                f'{deviance[i]["mean"]:.2f}',
                f'{deviance[i]["median"]:.2f}',
                j,
            ]
            for i, j in self.ndata.items()
            if i != 'total'
        ]
        rows.append(
            [
                'Total',
                'stat/dof',
                f'{deviance["total"]["mean"]:.2f}/{self.dof}',
                f'{deviance["total"]["median"]:.2f}/{self.dof}',
                self.ndata['total'],
            ]
        )
        names = [
            'Data',
            'Statistic',
            'Mean',
            'Median',
            'Channels',
        ]
        stat_tab = make_pretty_table(names, rows)

        loo = self.loo
        waic = self.waic
        rows = [
            [
                'LOOIC',
                f'{loo.elpd_loo:.2f} ± {loo.se:.2f}',
                f'{loo.p_loo:.2f}',
            ],
            [
                'WAIC',
                f'{waic.elpd_waic:.2f} ± {waic.se:.2f}',
                f'{waic.p_waic:.2f}',
            ],
        ]
        names = ['Method', 'Deviance', 'p']
        ic_tab = make_pretty_table(names, rows)

        ranges = ['(-Inf, 0.5]', '(0.5, 0.7]', '(0.7, 1]', '(1, Inf)']
        flags = ['good', 'ok', 'bad', 'very bad']
        bins = np.asarray([-np.inf, 0.5, 0.7, 1, np.inf])
        counts, *_ = np.histogram(loo.pareto_k.values, bins)
        pct = [f'{i:.1%}' for i in counts / np.sum(counts)]
        rows = list(zip(ranges, flags, counts, pct))
        names = ['Range', 'Flag', 'Count', 'Pct.']
        k_tab = make_pretty_table(names, rows)

        self._info_tabs = {
            'params': params_tab,
            'stat': stat_tab,
            'ic': ic_tab,
            'k': k_tab,
        }
        return self._info_tabs

    @property
    def plot(self) -> PosteriorResultPlotter:
        if self._plotter is None:
            self._plotter = PosteriorResultPlotter(self)
        return self._plotter

    def covar(
        self,
        params: str | Iterable[str] | None = None,
        fn: dict[str, Callable] | None = None,
        parallel: bool = True,
    ) -> ParamsCovar:
        """Calculate the covariance matrix.

        Parameters
        ----------
        params : str or sequence of str, optional
            Parameters to calculate covariance matrix. If not specified,
            calculate for all parameters.
        fn : dict, optional
            A dict containing functions to calculate the covariance matrix.
            The keys are the names of the function results, and the values are
            the functions whose input is a dict of model parameters.
        parallel : bool, optional
            Whether to use parallel computation for `fn`. The default is True.

        Returns
        -------
        ParamsCovar
            The covariance matrix.
        """
        params = check_params(params, self._helper)
        fn = self._check_fn(fn)

        params_dist = self._params_dist

        @jax.jit
        @jax.vmap
        @jax.jit
        def eval_fn(params):
            params_arr = jnp.array([params[k] for k in params_dist])
            fn_arr = jnp.array([f(params) for f in fn.values()])
            return jnp.hstack([params_arr, fn_arr])

        if parallel:
            n_parallel = get_parallel_number(self._n_parallel)
            devices = create_device_mesh(
                mesh_shape=(n_parallel,),
                devices=jax.devices()[:n_parallel],
            )
            mesh = Mesh(devices, axis_names=('i',))
            pi = PartitionSpec('i')
            eval_fn = shard_map(
                f=eval_fn,
                mesh=mesh,
                in_specs=(pi,),
                out_specs=pi,
                check_rep=False,
            )

        samples = eval_fn(params_dist)
        covar = np.cov(samples, rowvar=False)

        names = tuple(params) + tuple(fn.keys())
        var2pos = dict(zip(names, range(len(names))))
        matrix = CovarMatrix(var2pos)
        mask = np.array(
            [p in params for p in params_dist] + [True] * len(fn),
            dtype=bool,
        )
        matrix[:] = np.array(covar)[:, mask][mask]
        return ParamsCovar(names=names, matrix=matrix)

    def ci(
        self,
        cl: float | int = 1,
        params: str | Iterable[str] | None = None,
        fn: dict[str, Callable] | None = None,
        hdi: bool = False,
        parallel: bool = True,
    ) -> CredibleInterval:
        """Calculate credible intervals.

        Parameters
        ----------
        cl : float or int, optional
            The credible level of samples within the credible interval. If
            0 < `cl` < 1, the value is interpreted as the probability mass.
            If `cl` >= 1, it is interpreted as the number of standard
            deviations. For example, ``cl=1`` produces a 1-sigma or 68.3%
            credible interval. The default is 1.
        params : str or sequence of str, optional
            Parameters to calculate confidence intervals. If not specified,
            calculate for parameters of interest.
        fn : dict, optional
            A dict containing functions to calculate the confidence intervals.
            The keys are the names of the function results, and the values are
            the functions whose input is a dict of model parameters.
        hdi : bool, optional
            Whether to return the highest density interval. The default is
            False, which means an equal tailed interval is returned.
        parallel : bool, optional
            Whether to use parallel computation for `fn`. The default is True.

        Returns
        -------
        CredibleInterval
            The credible interval.
        """
        cl = self._to_unit_cl(cl)
        fn = self._check_fn(fn)
        params = check_params(params, self._helper)

        if hdi:
            median = self.idata['posterior'].median()
            median = {
                k: float(v) for k, v in median.data_vars.items() if k in params
            }
            interval = az.hdi(self.idata, cl, var_names=params)
            interval = {
                k: (float(v[0]), float(v[1]))
                for k, v in interval.data_vars.items()
            }
        else:
            q = [0.5, 0.5 - cl / 2.0, 0.5 + cl / 2.0]
            quantile = self.idata['posterior'].quantile(q)
            quantile = {
                k: v for k, v in quantile.data_vars.items() if k in params
            }
            median = {k: float(v[0]) for k, v in quantile.items()}
            interval = {
                k: (float(v[1]), float(v[2])) for k, v in quantile.items()
            }

        dist = {
            k: v.data
            for k, v in self.idata['posterior'][params].data_vars.items()
        }

        if fn:
            median_, interval_, dist_ = self._ci_fn(fn, cl, hdi, parallel)
            median |= median_
            interval |= interval_
            dist |= dist_

        vars_names = tuple(params) + tuple(fn.keys())

        error = {
            k: (interval[k][0] - median[k], interval[k][1] - median[k])
            for k in vars_names
        }

        return CredibleInterval(
            median=_format_result(median, vars_names),
            intervals=_format_result(interval, vars_names),
            errors=_format_result(error, vars_names),
            cl=cl,
            method='HDI' if hdi else 'ETI',
            dist=dist,
        )

    def _ci_fn(
        self,
        fn: dict[str, Callable],
        cl: float | int,
        hdi: bool,
        parallel: bool = True,
        params_setting: dict[str, JAXArray] | None = None,
    ):
        if params_setting is not None:
            params_setting = dict(params_setting)
        else:
            params_setting = {}

        params = self._params_dist

        cl = self._to_unit_cl(cl)

        eval_fn = jax.jit(jax.vmap(lambda p: jax.tree.map(lambda f: f(p), fn)))

        if parallel:
            n_parallel = get_parallel_number(self._n_parallel)
            devices = create_device_mesh(
                mesh_shape=(n_parallel,),
                devices=jax.devices()[:n_parallel],
            )
            mesh = Mesh(devices, axis_names=('i',))
            pi = PartitionSpec('i')
            eval_fn = shard_map(
                f=eval_fn,
                mesh=mesh,
                in_specs=(pi,),
                out_specs=pi,
                check_rep=False,
            )

        if params_setting:
            n = len(list(params.values())[0])
            params_setting = {
                k: jnp.full(n, v) for k, v in params_setting.items()
            }

        dist = jax.device_get(eval_fn(params | params_setting))

        if hdi:
            median = jax.tree.map(np.median, dist)
            interval = az.hdi(dist, cl)
            interval = jax.tree.map(
                lambda x: (float(x[0]), float(x[1])),
                interval.data_vars,
            )
        else:
            q = [0.5, 0.5 - cl / 2.0, 0.5 + cl / 2.0]
            quantile = jax.tree.map(lambda x: np.quantile(x, q), dist)
            median = jax.tree.map(lambda x: float(x[0]), quantile)
            interval = jax.tree.map(
                lambda x: (float(x[1]), float(x[2])), quantile
            )

        return median, interval, dist

    def _intensity_ci(
        self,
        egrid: JAXArray,
        energy: bool,
        cl: float | int,
        converter: Callable,
        hdi: bool,
        comps: bool,
        params: dict[str, JAXArray] | None,
    ) -> dict[str, Q | float]:
        """Calculate confidence interval of flux.

        Parameters
        ----------
        egrid : array-like
            Energy grid used in trapezoidal rule.
        energy : bool
            Whether the intensity is based on energy flux.
        cl : float or int
            Credible level.
        converter : callable
            Function to convert the flux into desired intensity.
        hdi : bool
            Whether to return the highest density interval.
        comps : bool
            Whether to return the result of each component.
        params : dict, optional
            Parameters dict to overwrite the posterior parameters.

        Returns
        -------
        dict
            The credible interval of intensity.
        """
        cl = self._to_unit_cl(cl)

        fn = jax.jit(lambda p: self._flux_fn(egrid, p, energy, comps))

        median, intervals, dist = self._ci_fn(
            {'intensity': fn}, cl, hdi, True, params
        )
        median = median['intensity']
        intervals = intervals['intensity']
        dist = dist['intensity']

        if energy:
            unit = u.Unit('erg cm^-2 s^-1')
        else:
            unit = u.Unit('ph cm^-2 s^-1')
        convert = lambda x: (x if x is None else converter(x * unit))

        errors = jax.tree.map(
            lambda x, y: (y[0] - x, y[1] - x), median, intervals
        )

        return {
            'median': jax.tree.map(convert, median),
            'intervals': jax.tree.map(convert, intervals),
            'errors': jax.tree.map(convert, errors),
            'cl': cl,
            'dist': jax.tree.map(convert, dist),
        }

    def flux(
        self,
        emin: float | int,
        emax: float | int,
        cl: float | int = 1,
        energy: bool = True,
        ngrid: int = 1000,
        hdi: bool = False,
        comps: bool = False,
        log: bool = True,
        params: dict[str, float | int] | None = None,
    ) -> PosteriorFlux:
        r"""Calculate the flux of model.

        .. warning::
            The flux is calculated by trapezoidal rule, and is accurate only
            if enough numbers of energy grids are used.

        Parameters
        ----------
        emin : float or int
            Minimum value of energy range, in units of keV.
        emax : float or int
            Maximum value of energy range, in units of keV.
        cl : float or int, optional
            The credible level of samples within the credible interval. If
            0 < `cl` < 1, the value is interpreted as the probability mass.
            If `cl` >= 1, it is interpreted as the number of standard
            deviations. For example, ``cl=1`` produces a 1-sigma or 68.3%
            credible interval. The default is 1.
        energy : bool, optional
            When True, calculate energy flux in units of erg cm⁻² s⁻¹;
            otherwise calculate photon flux in units of ph cm⁻² s⁻¹.
            The default is True.
        ngrid : int, optional
            The energy grid number to use in integration. The default is 1000.

        Other Parameters
        ----------------
        hdi : bool, optional
            Whether to return the highest density interval. The default is
            False, which means an equal tailed interval is returned.
        comps : bool, optional
            Whether to return the result of each component. The default is
            False.
        log : bool, optional
            Whether to use logarithmically regular energy grid. The default is
            True.
        params : dict, optional
            Parameters dict to overwrite the fitted parameters. Ignored when
            `method` is ``'profile'``.

        Returns
        -------
        PosteriorFlux
            The flux of the model.
        """
        if log:
            egrid = jnp.geomspace(emin, emax, ngrid)
        else:
            egrid = jnp.linspace(emin, emax, ngrid)

        flux = self._intensity_ci(
            egrid, energy, cl, lambda x: x, hdi, comps, params
        )

        return PosteriorFlux(
            emin, emax, bool(energy), **flux, method='HDI' if hdi else 'ETI'
        )

    def lumin(
        self,
        emin_rest: float | int,
        emax_rest: float | int,
        z: float | int,
        cl: float | int = 1,
        ngrid: int = 1000,
        hdi: bool = False,
        comps: bool = False,
        log: bool = True,
        params: dict[str, float | int] | None = None,
        cosmo: LambdaCDM = Planck18,
    ) -> PosteriorLumin:
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
        cl : float or int, optional
            The credible level of samples within the credible interval. If
            0 < `cl` < 1, the value is interpreted as the probability mass.
            If `cl` >= 1, it is interpreted as the number of standard
            deviations. For example, ``cl=1`` produces a 1-sigma or 68.3%
            credible interval. The default is 1.
        ngrid : int, optional
            The energy grid number to use in integration. The default is 1000.

        Other Parameters
        ----------------
        hdi : bool, optional
            Whether to return the highest density interval. The default is
            False, which means an equal tailed interval is returned.
        comps : bool, optional
            Whether to return the result of each component. The default is
            False.
        log : bool, optional
            Whether to use logarithmically regular energy grid. The default is
            True.
        params : dict, optional
            Parameters dict to overwrite the fitted parameters. Ignored when
            `method` is ``'profile'``.
        cosmo : LambdaCDM, optional
            Cosmology model used to calculate luminosity. The default is
            Planck18.

        Returns
        -------
        PosteriorLumin
            The luminosity of the model.
        """
        if log:
            egrid = jnp.geomspace(emin_rest, emax_rest, ngrid) / (1.0 + z)
        else:
            egrid = jnp.linspace(emin_rest, emax_rest, ngrid) / (1.0 + z)

        z = float(z)
        factor = 4.0 * np.pi * cosmo.luminosity_distance(z) ** 2
        to_lumin = lambda x: (x * factor).to('erg s^-1')

        lumin = self._intensity_ci(
            egrid, True, cl, to_lumin, hdi, comps, params
        )

        return PosteriorLumin(
            emin_rest,
            emax_rest,
            z,
            cosmo,
            **lumin,
            method='HDI' if hdi else 'ETI',
        )

    def eiso(
        self,
        emin_rest: float | int,
        emax_rest: float | int,
        z: float | int,
        duration: float | int,
        cl: float | int = 1,
        ngrid: int = 1000,
        hdi: bool = False,
        comps: bool = False,
        log: bool = True,
        params: dict[str, float | int] | None = None,
        cosmo: LambdaCDM = Planck18,
    ) -> PosteriorEiso:
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
            Observed duration of the source, in units of seconds.
        cl : float or int, optional
            The credible level of samples within the credible interval. If
            0 < `cl` < 1, the value is interpreted as the probability mass.
            If `cl` >= 1, it is interpreted as the number of standard
            deviations. For example, ``cl=1`` produces a 1-sigma or 68.3%
            credible interval. The default is 1.
        ngrid : int, optional
            The energy grid number to use in integration. The default is
            1000.

        Other Parameters
        ----------------
        hdi : bool, optional
            Whether to return the highest density interval. The default is
            False, which means an equal tailed interval is returned.
        comps : bool, optional
            Whether to return the result of each component. The default is
            False.
        log : bool, optional
            Whether to use logarithmically regular energy grid. The default
            is True.
        params : dict, optional
            Parameters dict to overwrite the fitted parameters. Ignored when
            `method` is ``'profile'``.
        cosmo : LambdaCDM, optional
            Cosmology model used to calculate luminosity. The default is
            Planck18.

        Returns
        -------
        PosteriorEiso
            The isotropic emission energy of the model.
        """
        if log:
            egrid = jnp.geomspace(emin_rest, emax_rest, ngrid) / (1.0 + z)
        else:
            egrid = jnp.linspace(emin_rest, emax_rest, ngrid) / (1.0 + z)

        # This includes correction for energy redshift and time dilation.
        z = float(z)
        factor = 4.0 * np.pi * cosmo.luminosity_distance(z) ** 2
        factor *= duration / (1 + z) * u.s
        to_eiso = lambda x: (x * factor).to('erg')

        eiso = self._intensity_ci(egrid, True, cl, to_eiso, hdi, comps, params)

        return PosteriorEiso(
            emin_rest,
            emax_rest,
            z,
            duration,
            cosmo,
            **eiso,
            method='HDI' if hdi else 'ETI',
        )

    def ppc(
        self,
        n: int = 10000,
        seed: int | None = None,
        parallel: bool = True,
        n_parallel: int | None = None,
        progress: bool = True,
        update_rate: int = 50,
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
        n_parallel : int, optional
            Number of parallel processes to use when `parallel` is ``True``.
            Defaults to ``jax.local_device_count()``.
        progress : bool, optional
            Whether to display progress bar. The default is True.
        update_rate : int, optional
            The update rate of progress bar. The default is 50.
        """
        n = int(n)
        n_parallel = get_parallel_number(n_parallel)
        if parallel and (n % n_parallel):
            n += n_parallel - n % n_parallel

        # reuse the previous result if all setup is the same
        if self._ppc and self._ppc.n == n and self._ppc.seed == seed:
            return

        helper = self._helper
        free_params = helper.params_names['free']
        seed = helper.seed['pred'] if seed is None else int(seed)

        # randomly select n samples from posterior
        rng = np.random.default_rng(seed)
        idata = self.idata
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
            seed,
            params,
            models,
            1,
            parallel,
            n_parallel,
            progress,
            update_rate,
            'PPC',
        )
        valid = result.pop('valid')
        result = jax.tree.map(lambda x: x[valid], result)

        self._ppc = PPCResult(
            params_rep=params,
            models_rep=models,
            data=result['data'],
            params_fit=result['params'],
            models_fit=result['models'],
            deviance=result['deviance'],
            p_value=jax.tree.map(
                lambda obs, sim: np.sum(sim >= obs, axis=0) / len(sim),
                self._mle['deviance'],
                result['deviance'],
            ),
            n=n,
            n_valid=np.sum(valid),
            seed=seed,
        )

    @property
    def gof(self) -> dict[str, float]:
        if self._ppc is None:
            raise RuntimeError(
                'PosteriorResult.ppc() must be called to assess gof'
            )
        p_value = self._ppc.p_value
        p_value = p_value['group'] | {'total': p_value['total']}
        return {k: float(p_value[k]) for k in self.ndata.keys()}

    @property
    def _mle(self):
        """MLE result."""
        if self._mle_result is None:
            mle_result = {}
            helper = self._helper

            # MLE information of the model
            free_params = helper.params_names['free']
            mle_idx = self.idata['log_likelihood']['total'].argmax(...)
            init = self.idata['posterior'][free_params].sel(**mle_idx)
            init = {k: v.values for k, v in init.data_vars.items()}
            init = helper.constr_dic_to_unconstr_arr(init)
            mle_unconstr = self._fit._optimize_lm(init, throw=False)[0]

            # MLE of model params in constrained space
            mle, cov = jax.device_get(helper.get_mle(mle_unconstr))
            err = np.sqrt(np.diagonal(cov))
            params_names = helper.params_names['all']
            mle_result['params'] = dict(zip(params_names, zip(mle, err)))

            sites = jax.device_get(jax.jit(helper.get_sites)(mle_unconstr))

            # model deviance at MLE
            loglike = sites['loglike']
            # drop unnecessary terms
            loglike.pop('data')
            loglike.pop('channels')
            mle_result['deviance'] = jax.tree.map(lambda x: -2.0 * x, loglike)

            # model values at MLE
            mle_result['models'] = sites['models']

            self._mle_result = mle_result
        return self._mle_result

    @property
    def mle(self) -> dict[str, tuple[float, float]]:
        """MLE parameters."""
        return dict(self._mle['params'])

    @property
    def idata(self) -> az.InferenceData:
        """ArviZ InferenceData."""
        return self._idata

    @property
    def _params_dist(self) -> dict[str, JAXArray]:
        """Posterior parameters, the size is truncated to <= nmax."""
        if self._params is not None:
            return self._params

        nmax = 10000
        post = self.idata['posterior'][self._helper.params_names['free']]
        n = post.chain.size * post.draw.size
        if n > nmax:
            n = nmax - nmax % jax.local_device_count()
            rng = np.random.default_rng(self._helper.seed['mcmc'])
            i = rng.integers(0, post.chain.size, n)
            j = rng.integers(0, post.draw.size, n)
            post = {k: v.values[i, j] for k, v in post.items()}
        else:
            post = {k: np.hstack(v.values) for k, v in post.data_vars.items()}
        self._params = post
        return self._params

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
        samples = jax.tree.map(
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
        ess = int(result.ESS)
        self._ess = {p: ess for p in self._helper.params_names['all']}
        # relative mcmc efficiency
        self._reff = float(ess / result.total_num_samples)
        # model evidence
        self._lnZ = (float(result.log_Z_mean), float(result.log_Z_uncert))

    def _init_from_ultranest(self, sampler: ReactiveNestedSampler):
        result = sampler._transform_back(sampler.results['samples'])
        nsamples = len(sampler.results['samples'])
        ncores = jax.local_device_count()
        ndrop = nsamples % ncores

        # get posterior samples
        samples = jax.tree.map(lambda x: x[None, : nsamples - ndrop], result)

        # attrs for each group of arviz.InferenceData
        attrs = {
            'elisa_version': __version__,
            'inference_library': 'ultranest',
            'inference_library_version': ultranest.__version__,
        }

        self._generate_idata(samples, attrs)

        # effective sample size
        ess = int(sampler.results['ess'])
        self._ess = {p: ess for p in self._helper.params_names['all']}
        # relative mcmc efficiency
        self._reff = float(ess / nsamples)
        # model evidence
        self._lnZ = (
            float(sampler.results['logz']),
            float(sampler.results['logzerr']),
        )

    def _init_from_nautilus(self, sampler: Sampler):
        result = sampler.posterior(equal_weight=True)[0]
        result = sampler._transform_back(result)
        ncores = jax.local_device_count()

        # get posterior samples
        samples = jax.tree.map(
            lambda x: x[None, : len(x) - len(x) % ncores],
            result,
        )

        # attrs for each group of arviz.InferenceData
        attrs = {
            'elisa_version': __version__,
            'inference_library': 'nautilus',
            'inference_library_version': nautilus.__version__,
        }

        self._generate_idata(samples, attrs)

        # effective sample size
        ess = int(sampler.n_eff)
        self._ess = {p: ess for p in self._helper.params_names['all']}
        # relative mcmc efficiency
        total_sample = len(sampler.posterior(equal_weight=False)[0])
        self._reff = float(ess / total_sample)
        # model evidence
        self._lnZ = (float(sampler.log_z), None)

    def _generate_idata(self, samples, attrs, sample_stats=None):
        samples = jax.tree.map(jax.device_get, samples)
        helper = self._helper

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
    def deviance(self) -> dict:
        """Mean and median of model deviance."""
        if self._deviance is None:
            stat_keys = {
                i: f'{i}_total' if i != 'total' else i
                for i in self.ndata.keys()
            }
            keys = list(stat_keys.values())
            deviance = -2.0 * self.idata['log_likelihood'][keys]
            deviance_mean = deviance.mean()
            deviance_median = deviance.median()
            self._deviance = {
                k: {
                    'mean': float(deviance_mean[v]),
                    'median': float(deviance_median[v]),
                }
                for k, v in stat_keys.items()
            }

        return self._deviance

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
            params_names = self._helper.params_names['all']
            posterior = self.idata['posterior'][params_names]

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
            if 'sample_stats' in self.idata:
                n = int(self.idata['sample_stats']['diverging'].sum())
            else:
                n = 0

            self._divergence = n

        return self._divergence

    @property
    def waic(self) -> ELPDData:
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
                self.idata, var_name='channels', scale='deviance'
            )

        return self._waic

    @property
    def loo(self) -> ELPDData:
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
                self.idata,
                var_name='channels',
                reff=self.reff,
                scale='deviance',
            )

        return self._loo

    @property
    def lnZ(self) -> tuple[float, float] | tuple[None, None]:
        """Log model evidence and uncertainty."""
        return self._lnZ

    @property
    def _psislw(self) -> DataArray:
        if self._psislw_ is None:
            idata = self.idata
            reff = self.reff
            stack_kwargs = {'__sample__': ('chain', 'draw')}
            log_weights, kss = az.psislw(
                -idata['log_likelihood']['channels'].stack(**stack_kwargs),
                reff,
            )
            self._psislw_ = log_weights
        return self._psislw_

    def _loo_expectation(self, values: DataArray, data: str) -> DataArray:
        """Computes weighted expectations using the PSIS weights.

        Notes
        -----
        The expectations estimated assume that the PSIS approximation is
        working well. A small Pareto k estimate is necessary, but not
        sufficient to give reliable estimates.

        Parameters
        ----------
        values : DataArray
            Values to compute the expectation.
        data : str
            The data name.

        Returns
        -------
        DataArray
            The expectation of the values.
        """
        assert data in self._helper.data_names
        channel = self._helper.channels[f'{data}_channel']
        log_weights = self._psislw.sel(channel=channel)
        log_weights = log_weights.rename({'channel': f'{data}_channel'})
        log_expectation = log_weights + np.log(np.abs(values))
        weighted = np.sign(values) * np.exp(log_expectation)
        return weighted.sum(dim='__sample__')

    @property
    def _loo_pit(self) -> dict[str, tuple]:
        """Leave-one-out probability integral transform."""
        if self._pit is not None:
            return self._pit

        idata = self.idata
        helper = self._helper
        stack_kwargs = {'__sample__': ('chain', 'draw')}
        y_hat = idata['posterior_predictive']['channels'].stack(**stack_kwargs)
        loo_pit = az.loo_pit(
            y=idata['observed_data']['channels'],
            y_hat=y_hat,
            log_weights=self._psislw,
        )

        loo_pit = {
            name: loo_pit.sel(channel=data.channel).values
            for name, data in helper.data.items()
        }

        discrete_stats = {'cstat', 'pstat', 'wstat'}
        data_stats = helper.statistic
        has_discrete = discrete_stats.intersection(data_stats.values())
        if has_discrete:
            data_minus = {}
            for k, d in helper.data.items():
                unit = 1.0 / (d.channel_width * d.spec_exposure)
                if data_stats[k] in {'cstat', 'pstat'}:
                    data_minus[k] = (d.spec_counts - 1.0) * unit
                elif data_stats[k] == 'wstat':
                    # Get the next small net spectrum values
                    data_minus[k] = (
                        np.maximum(
                            (d.spec_counts - 1.0)
                            - d.back_ratio * d.back_counts,
                            d.spec_counts
                            - d.back_ratio * (d.back_counts + 1.0),
                        )
                        * unit
                    )
                else:  # chi2, pgstat
                    data_minus[k] = d.ce

            y_miuns = idata['observed_data']['channels'].copy()
            y_miuns.data = np.hstack(list(data_minus.values()))

            loo_pit_minus = az.loo_pit(
                y=y_miuns,
                y_hat=y_hat,
                log_weights=self._psislw,
            )
            loo_pit_minus = {
                name: loo_pit_minus.sel(channel=data.channel).values
                for name, data in helper.data.items()
            }
        else:
            loo_pit_minus = loo_pit

        self._pit = {
            name: (loo_pit_minus[name], loo_pit[name])
            for name in loo_pit.keys()
        }
        return self._pit


class ParamsCovar(NamedTuple):
    """Covariance matrix of the model parameters."""

    names: tuple[str, ...]
    """Parameter names."""

    matrix: CovarMatrix
    """Covariance matrix."""


class ConfidenceInterval(NamedTuple):
    """Confidence interval result."""

    mle: dict[str, float]
    """MLE of the parameters."""

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


class MLEFlux(NamedTuple):
    """The flux of the MLE model."""

    emin: float
    """Minimum value of energy range."""

    emax: float
    """Maximum value of energy range."""

    energy: bool
    """Whether the flux is in energy flux. False for photon flux."""

    mle: dict[str, Q] | dict[str, dict[str, Q]]
    """The model flux at MLE."""

    intervals: dict[str, tuple[Q, Q]] | dict[str, dict[str, tuple[Q, Q]]]
    """The confidence intervals of the model flux."""

    errors: dict[str, tuple[Q, Q]] | dict[str, dict[str, tuple[Q, Q]]]
    """The confidence intervals of the model flux in error form."""

    cl: float
    """The confidence level."""

    method: str
    """Method used to calculate the confidence interval."""

    status: dict
    """Status of the calculation progress."""


class MLELumin(NamedTuple):
    """The luminosity of the MLE model."""

    emin_rest: float
    """Minimum value of rest-frame energy range."""

    emax_rest: float
    """Maximum value of rest-frame energy range."""

    z: float
    """Redshift of the source."""

    cosmo: LambdaCDM
    """Cosmology model used to calculate luminosity."""

    mle: dict[str, Q] | dict[str, dict[str, Q]]
    """The model luminosity at MLE."""

    intervals: dict[str, tuple[Q, Q]] | dict[str, dict[str, tuple[Q, Q]]]
    """The confidence intervals of the model luminosity."""

    errors: dict[str, tuple[Q, Q]] | dict[str, dict[str, tuple[Q, Q]]]
    """The confidence intervals of the model luminosity in error form."""

    cl: float
    """The confidence level."""

    method: str
    """Method used to calculate the confidence interval."""

    status: str
    """Status of the calculation progress."""


class MLEEiso(NamedTuple):
    """The isotropic emission energy of the MLE model."""

    emin_rest: float
    """Minimum value of rest-frame energy range."""

    emax_rest: float
    """Maximum value of rest-frame energy range."""

    z: float
    """Redshift of the source."""

    duration: float
    """Observed duration of the source."""

    cosmo: LambdaCDM
    """Cosmology model used to calculate Eiso."""

    mle: dict[str, Q] | dict[str, dict[str, Q]]
    r"""The model Eiso at MLE."""

    intervals: dict[str, tuple[Q, Q]] | dict[str, dict[str, tuple[Q, Q]]]
    """The confidence intervals of the model Eiso."""

    errors: dict[str, tuple[Q, Q]] | dict[str, dict[str, tuple[Q, Q]]]
    """The confidence intervals of the model Eiso in error form."""

    cl: float
    """The confidence level."""

    method: str
    """Method used to calculate the confidence interval."""

    status: str
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


class CredibleInterval(NamedTuple):
    """Credible interval result."""

    median: dict[str, float]
    """Median of the parameters."""

    intervals: dict[str, tuple[float, float]]
    """The credible intervals."""

    errors: dict[str, tuple[float, float]]
    """The credible intervals in error form."""

    cl: float
    """The credible level."""

    method: str
    """Highest Density Interval (HDI), or equal tailed interval (ETI)."""

    dist: dict[str, JAXArray]
    """Posterior distribution."""


class PosteriorFlux(NamedTuple):
    """Posterior flux."""

    emin: float
    """Minimum value of energy range."""

    emax: float
    """Maximum value of energy range."""

    energy: bool
    """Whether the flux is in energy flux. False for photon flux."""

    median: dict[str, Q] | dict[str, dict[str, Q]]
    """The median flux."""

    intervals: dict[str, tuple[Q, Q]] | dict[str, dict[str, tuple[Q, Q]]]
    """The credible intervals of the model flux."""

    errors: dict[str, tuple[Q, Q]] | dict[str, dict[str, tuple[Q, Q]]]
    """The credible intervals of the model flux in error form."""

    cl: float
    """The credible level."""

    method: str
    """Highest Density Interval (HDI), or equal tailed interval (ETI)."""

    dist: dict[str, Q] | dict[str, dict[str, Q]]
    """Posterior flux distribution."""


class PosteriorLumin(NamedTuple):
    """Posterior luminosity."""

    emin_rest: float
    """Minimum value of rest-frame energy range."""

    emax_rest: float
    """Maximum value of rest-frame energy range."""

    z: float
    """Redshift of the source."""

    cosmo: LambdaCDM
    """Cosmology model used to calculate luminosity."""

    median: dict[str, Q] | dict[str, dict[str, Q]]
    """The median luminosity."""

    intervals: dict[str, tuple[Q, Q]] | dict[str, dict[str, tuple[Q, Q]]]
    """The credible intervals of the model luminosity."""

    errors: dict[str, tuple[Q, Q]] | dict[str, dict[str, tuple[Q, Q]]]
    """The credible intervals of the model luminosity in error form."""

    cl: float
    """The credible level."""

    method: str
    """Highest Density Interval (HDI), or equal tailed interval (ETI)."""

    dist: dict[str, Q] | dict[str, dict[str, Q]]
    """Posterior distribution of luminosity."""


class PosteriorEiso(NamedTuple):
    """Posterior isotropic emission energy."""

    emin_rest: float
    """Minimum value of rest-frame energy range."""

    emax_rest: float
    """Maximum value of rest-frame energy range."""

    z: float
    """Redshift of the source."""

    duration: float
    """Observed duration of the source."""

    cosmo: LambdaCDM
    """Cosmology model used to calculate Eiso."""

    median: dict[str, Q] | dict[str, dict[str, Q]]
    r"""The median Eiso."""

    intervals: dict[str, tuple[Q, Q]] | dict[str, dict[str, tuple[Q, Q]]]
    """The credible intervals of the model Eiso."""

    errors: dict[str, tuple[Q, Q]] | dict[str, dict[str, tuple[Q, Q]]]
    """The credible intervals of the model Eiso in error form."""

    cl: float
    """The credible level."""

    method: str
    """Highest Density Interval (HDI), or equal tailed interval (ETI)."""

    dist: dict[str, Q] | dict[str, dict[str, Q]]
    """Posterior distribution of Eiso."""


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


def _format_result(result: dict, order: Sequence[str]) -> dict:
    """Sort the result and use float type."""
    formatted = jax.tree.map(float, result)
    return {k: formatted[k] for k in order}
