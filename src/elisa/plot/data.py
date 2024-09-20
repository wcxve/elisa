"""Data classes for plotting."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cache, wraps
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats as stats
from jax.experimental.mesh_utils import create_device_mesh
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec

from elisa.infer.likelihood import (
    _STATISTIC_BACK_NORMAL,
    _STATISTIC_SPEC_NORMAL,
    _STATISTIC_WITH_BACK,
)
from elisa.plot.residuals import (
    pearson_residuals,
    pit_poisson,
    pit_poisson_normal,
    pit_poisson_poisson,
    quantile_residuals_poisson,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Callable, Literal

    from xarray import DataArray

    from elisa.infer.results import (
        BootstrapResult,
        FitResult,
        MLEResult,
        PosteriorResult,
        PPCResult,
    )
    from elisa.util.typing import Array, NumPyArray


def _cache_method(bound_method: Callable) -> Callable:
    """Cache instance method."""
    return cache(bound_method)


def _cache_method_with_check(
    instance: Any, bound_method: Callable, check_fields: Sequence[str]
) -> Callable:
    """Cache instance method with computation dependency check."""

    def get_id():
        return {field: id(getattr(instance, field)) for field in check_fields}

    cached_method = cache(bound_method)
    old_id = get_id()

    @wraps(bound_method)
    def _(*args, **kwargs):
        if (new_id := get_id()) != old_id:
            cached_method.cache_clear()
            old_id.update(new_id)
        return cached_method(*args, **kwargs)

    return _


def _get_cached_method_decorator(storage: list):
    def decorator(method: Callable):
        storage.append(method.__name__)
        return method

    return decorator


def _get_cached_method_with_check_decorator(
    storage: list, check_fields: str | Sequence[str]
):
    if isinstance(check_fields, str):
        check_fields = [check_fields]
    else:
        check_fields = list(check_fields)

    def decorator(method: Callable):
        name = method.__name__
        storage.append((name, check_fields))
        return method

    return decorator


class PlotData(ABC):
    """Base class for data used in plotting."""

    _cached_method: list[str]
    _cached_method_with_check: list[tuple[str, list[str]]]
    _unfolded_model_fn: dict[str, Callable]
    _ph_egrid: NumPyArray | None = None

    def __init__(self, name: str, result: FitResult, seed: int):
        self.name = str(name)
        self.result = result
        self.seed = seed
        self.data = result._helper.data[self.name]
        self.statistic = result._helper.statistic[self.name]

        for f in self._cached_method:
            method = getattr(self, f)
            setattr(self, f, _cache_method(method))

        for f, fields in self._cached_method_with_check:
            method = getattr(self, f)
            setattr(self, f, _cache_method_with_check(self, method, fields))

        model = self.result._helper.model[self.name]
        self._unfolded_model_fn = {
            'ne': jax.jit(lambda e, p: model.ne(e, p, comps=False)),
            'ene': jax.jit(lambda e, p: model.ene(e, p, comps=False)),
            'eene': jax.jit(lambda e, p: model.eene(e, p, comps=False)),
            'ne_comps': jax.jit(lambda e, p: model.ne(e, p, comps=True)),
            'ene_comps': jax.jit(lambda e, p: model.ene(e, p, comps=True)),
            'eene_comps': jax.jit(lambda e, p: model.eene(e, p, comps=True)),
        }

    @property
    def channel(self) -> NumPyArray:
        return self.data.channel

    @property
    def channel_emin(self) -> NumPyArray:
        return self.data.channel_emin

    @property
    def channel_emax(self) -> NumPyArray:
        return self.data.channel_emax

    @property
    def channel_emid(self) -> NumPyArray:
        return self.data.channel_emid

    @property
    def channel_width(self) -> NumPyArray:
        return self.data.channel_width

    @property
    def channel_emean(self) -> NumPyArray:
        return self.data.channel_emean

    @property
    def channel_errors(self) -> NumPyArray:
        return self.data.channel_errors

    @property
    def photon_egrid(self) -> NumPyArray:
        if self._ph_egrid is not None:
            return self._ph_egrid

        ph_egrid = self.data.photon_egrid
        mask = np.bitwise_and(
            self.channel_emin[0] <= ph_egrid, ph_egrid <= self.channel_emax[-1]
        )
        self._ph_egrid = ph_egrid[mask]
        return self._ph_egrid

    @property
    def spec_counts(self) -> Array:
        return self.data.spec_counts

    @property
    def spec_errors(self) -> Array:
        return self.data.spec_errors

    @property
    def back_ratio(self) -> float | Array:
        return self.data.back_ratio

    @property
    def back_counts(self) -> Array | None:
        return self.data.back_counts

    @property
    def back_errors(self) -> Array | None:
        return self.data.back_errors

    @property
    def net_counts(self) -> Array:
        return self.data.net_counts

    @property
    def net_errors(self) -> Array:
        return self.data.net_errors

    @property
    def ndata(self) -> int:
        return len(self.data.channel)

    @property
    def ce_data(self) -> Array:
        return self.data.ce

    @property
    def ce_errors(self) -> Array:
        return self.data.ce_errors

    @property
    @abstractmethod
    def ce_model(self) -> Array:
        """Point estimate of the folded source model."""
        pass

    @abstractmethod
    def ce_model_ci(self, cl: float = 0.683) -> Array | None:
        """Confidence/Credible intervals of the folded source model."""
        pass

    @property
    def has_comps(self) -> bool:
        return self.result._helper.model[self.name].has_comps

    @property
    def params_dist(self) -> dict[str, Array] | None:
        return self.result._params_dist

    def _unfolded_model(
        self,
        mtype: Literal['ne', 'ene', 'eene'],
        egrid: Array,
        params: dict,
        comps: bool,
    ) -> Array | dict:
        assert mtype in {'ne', 'ene', 'eene'}
        fn = self._unfolded_model_fn[f'{mtype}_comps' if comps else mtype]
        if len(np.shape(list(params.values())[0])) != 0:
            devices = create_device_mesh((jax.local_device_count(),))
            mesh = Mesh(devices, axis_names=('i',))
            p = PartitionSpec()
            pi = PartitionSpec('i')
            fn = shard_map(
                f=fn,
                mesh=mesh,
                in_specs=(p, pi),
                out_specs=pi,
                check_rep=False,
            )
        return jax.device_get(fn(egrid, params))

    @abstractmethod
    def unfolded_model(
        self,
        mtype: Literal['ne', 'ene', 'eene'],
        egrid: Array,
        params: dict,
        comps: bool,
        cl: float | Array | None = None,
    ) -> Array | dict:
        pass

    @abstractmethod
    def pit(self) -> tuple:
        """Probability integral transform."""
        pass

    @abstractmethod
    def residuals(
        self,
        rtype: Literal['rd', 'rp', 'rq'],
        seed: int | None,
        random_quantile: bool,
        mle: bool,
    ) -> Array | tuple[Array, bool | Array, bool | Array]:
        """Residuals between the data and the fitted models."""
        pass

    @abstractmethod
    def residuals_sim(
        self,
        rtype: Literal['rd', 'rp', 'rq'],
        seed: int | None,
        random_quantile: bool,
    ) -> Array | None:
        """Residuals bootstrap/ppc samples."""
        pass

    @abstractmethod
    def residuals_ci(
        self,
        rtype: Literal['rd', 'rp', 'rq'],
        cl: float,
        seed: int | None,
        random_quantile: bool,
        with_sign: bool,
    ) -> Array | None:
        """Confidence/Credible intervals of the residuals."""
        pass


_cached_method = []
_cached_method_with_check = []
_to_cached_method = _get_cached_method_decorator(_cached_method)
_to_cached_method_with_check = _get_cached_method_with_check_decorator(
    _cached_method_with_check, 'boot'
)


class MLEPlotData(PlotData):
    result: MLEResult
    _cached_method = _cached_method
    _cached_method_with_check = _cached_method_with_check

    @property
    def boot(self) -> BootstrapResult:
        return self.result._boot

    @property
    def params_mle(self) -> dict[str, Array]:
        return {k: v[0] for k, v in self.result._mle.items()}

    def get_model_mle(self, name: str) -> Array:
        return self.result._model_values[name]

    def get_model_boot(self, name: str) -> Array | None:
        boot = self.boot
        if boot is None:
            return None
        else:
            return boot.models[name]

    def get_data_boot(self, name: str) -> Array | None:
        boot = self.boot
        if boot is None:
            return None
        else:
            return boot.data[name]

    @property
    def ce_model(self) -> Array:
        return self.get_model_mle(self.name)

    @_to_cached_method_with_check
    def ce_model_ci(self, cl: float = 0.683) -> Array | None:
        if self.boot is None:
            return None

        assert 0.0 < cl < 1.0
        ci = np.quantile(
            self.get_model_boot(self.name),
            q=0.5 + cl * np.array([-0.5, 0.5]),
            axis=0,
        )
        return ci

    def unfolded_model(
        self,
        mtype: Literal['ne', 'ene', 'eene'],
        egrid: Array | None,
        params: dict | None,
        comps: bool,
        cl: float | Array | None = None,
    ) -> tuple[Array | dict, Array | dict | None]:
        assert mtype in {'ne', 'ene', 'eene'}
        if cl is not None:
            cl = np.atleast_1d(cl).astype(float)
            assert np.all(0.0 < cl) and np.all(cl < 1.0)
        params = {} if params is None else dict(params)
        comps = comps and self.has_comps

        egrid = jnp.asarray(egrid, float)

        params_mle = self.params_mle | params
        model_mle = self._unfolded_model(mtype, egrid, params_mle, comps)

        params_boot = self.params_dist
        if cl is None or params_boot is None:
            return model_mle, None
        else:
            n = [i.size for i in params_boot.values()][0]
            if params:
                params = {k: jnp.full(n, v) for k, v in params.items()}
                params_boot = params_boot | params
            model_boot = self._unfolded_model(mtype, egrid, params_boot, comps)
            q = 0.5 + cl[:, None] * np.array([-0.5, 0.5])
            if comps:
                ci = {
                    k: np.quantile(v, q, axis=0) for k, v in model_boot.items()
                }
            else:
                ci = np.quantile(model_boot, q, axis=0)
            return model_mle, ci

    @property
    def sign(self) -> dict[str, Array | None]:
        """Sign of the difference between the data and the fitted models."""
        return {'mle': self._sign_mle(), 'boot': self._sign_boot()}

    @_to_cached_method
    def _sign_mle(self) -> Array:
        return np.where(self.ce_data >= self.ce_model, 1.0, -1.0)

    @_to_cached_method_with_check
    def _sign_boot(self) -> Array | None:
        boot = self.get_model_boot(self.name)
        if boot is not None:
            boot = np.where(self.get_data_boot(self.name) >= boot, 1.0, -1.0)
        return boot

    def model(
        self,
        on_off: Literal['on', 'off'],
        mtype: Literal['mle', 'boot'],
    ) -> Array | None:
        """Point estimate or bootstrap models of the on/off measurement."""
        assert on_off in {'on', 'off'}
        assert mtype in {'mle', 'boot'}

        if (on_off == 'off') and (self.statistic not in _STATISTIC_WITH_BACK):
            return None

        name = f'{self.name}_N{on_off}_model'
        return getattr(self, f'get_model_{mtype}')(name)

    def deviance(self, rtype: Literal['mle', 'boot']) -> Array | None:
        """MLE and bootstrap deviance."""
        if rtype == 'mle':
            return self.result._deviance['point'][self.name]
        elif rtype == 'boot':
            if self.boot is not None:
                return self.boot.deviance['point'][self.name]
            else:
                return None
        else:
            raise ValueError(f'unknown deviance type: {rtype}')

    @property
    def _nsim(self) -> int:
        return 10000

    @_to_cached_method
    def pit(self) -> tuple[Array, Array]:
        stat = self.statistic

        if stat in _STATISTIC_SPEC_NORMAL:
            on_data = self.net_counts
        else:
            on_data = self.spec_counts
        on_model = self.model('on', 'mle')

        if stat in _STATISTIC_SPEC_NORMAL:  # chi2
            pit = stats.norm.cdf((on_data - on_model) / self.net_errors)
            return pit, pit

        if stat in _STATISTIC_WITH_BACK:
            off_data = self.back_counts
            off_model = self.model('off', 'mle')

            if stat in _STATISTIC_BACK_NORMAL:  # pgstat
                pit = pit_poisson_normal(
                    k=on_data,
                    lam=on_model,
                    v=off_data,
                    mu=off_model,
                    sigma=self.back_errors,
                    ratio=self.back_ratio,
                    seed=self.seed + 1,
                    nsim=self._nsim,
                )
                return pit, pit

            else:  # wstat
                return pit_poisson_poisson(
                    k1=on_data,
                    k2=off_data,
                    lam1=on_model,
                    lam2=off_model,
                    ratio=self.data.back_ratio,
                    minus=True,
                    seed=self.seed + 1,
                    nsim=self._nsim,
                )

        else:  # cstat, or pstat
            return pit_poisson(k=on_data, lam=on_model, minus=True)

    def residuals(
        self,
        rtype: Literal['rd', 'rp', 'rq'],
        seed: int | None = None,
        random_quantile: bool = True,
        mle: bool = True,
    ) -> Array | tuple[Array, bool | Array, bool | Array]:
        if rtype == 'rd':
            return self.deviance_residuals_mle()
        elif rtype == 'rp':
            return self.pearson_residuals_mle()
        elif rtype == 'rq':
            seed = self.seed if seed is None else int(seed)
            return self.quantile_residuals_mle(seed, random_quantile)
        else:
            raise NotImplementedError(f'{rtype} residual')

    def residuals_sim(
        self,
        rtype: Literal['rd', 'rp', 'rq'],
        seed: int | None = None,
        random_quantile: bool = True,
    ) -> Array | None:
        if self.boot is None or rtype == 'rq':
            return None

        if rtype == 'rd':
            r = self.deviance_residuals_boot()
        elif rtype == 'rp':
            r = self.pearson_residuals_boot()
        else:
            raise NotImplementedError(f'{rtype} residual')

        return r

    def residuals_ci(
        self,
        rtype: Literal['rd', 'rp', 'rq'],
        cl: float = 0.683,
        seed: int | None = None,
        random_quantile: bool = True,
        with_sign: bool = False,
    ) -> Array | None:
        if self.boot is None or rtype == 'rq':
            return None

        assert 0 < cl < 1

        r = self.residuals_sim(rtype, seed, random_quantile)

        if with_sign:
            return np.quantile(r, q=0.5 + cl * np.array([-0.5, 0.5]), axis=0)
        else:
            q = np.quantile(np.abs(r), q=cl, axis=0)
            return np.row_stack([-q, q])

    @_to_cached_method
    def deviance_residuals_mle(self) -> Array:
        return self._deviance_residuals('mle')

    @_to_cached_method_with_check
    def deviance_residuals_boot(self) -> Array | None:
        return self._deviance_residuals('boot')

    def _deviance_residuals(
        self, rtype: Literal['mle', 'boot']
    ) -> Array | None:
        if rtype == 'boot' and self.boot is None:
            return None

        # NB: if background is present, then this assumes the background is
        #     being profiled out, so that each src & bkg data pair has ~1 dof
        return self.sign[rtype] * np.sqrt(self.deviance(rtype))

    @_to_cached_method
    def pearson_residuals_mle(self) -> Array:
        return self._pearson_residuals('mle')

    @_to_cached_method_with_check
    def pearson_residuals_boot(self) -> Array | None:
        return self._pearson_residuals('boot')

    def _pearson_residuals(
        self, rtype: Literal['mle', 'boot']
    ) -> Array | None:
        if rtype == 'boot' and self.boot is None:
            return None

        stat = self.statistic

        if rtype == 'mle':
            if stat in _STATISTIC_SPEC_NORMAL:
                on_data = self.net_counts
            else:
                on_data = self.spec_counts
        else:
            on_data = self.get_data_boot(f'{self.name}_Non')

        if stat in _STATISTIC_SPEC_NORMAL:
            std = self.net_errors
        else:
            std = None

        r = pearson_residuals(on_data, self.model('on', rtype), std)

        if stat in _STATISTIC_WITH_BACK:
            if rtype == 'mle':
                off_data = self.back_counts
            else:
                off_data = self.get_data_boot(f'{self.name}_Noff')

            if self.statistic in _STATISTIC_BACK_NORMAL:
                std = self.back_errors
            else:
                std = None

            r_b = pearson_residuals(off_data, self.model('off', rtype), std)

            # NB: this assumes the background is being profiled out,
            #     so that each src & bkg data pair has ~1 dof
            r = self.sign[rtype] * np.sqrt(r * r + r_b * r_b)

        return r

    def quantile_residuals_mle(
        self, seed: int, random: bool
    ) -> tuple[Array, Array | bool, Array | bool]:
        pit_minus, pit = self.pit()

        if random:
            pit = np.random.default_rng(seed).uniform(pit_minus, pit)
        r = stats.norm.ppf(pit)

        lower = upper = False

        stat = self.statistic
        if stat == 'chi2':
            mask = (pit == 0.0) | (pit == 1.0)
            if np.any(mask):
                on_data = self.net_counts[mask]
                on_model = self.model('on', 'mle')[mask]
                error = self.net_errors[mask]
                r[mask] = (on_data - on_model) / error

        elif stat in {'cstat', 'pstat'}:
            mask = (pit == 0.0) | (pit == 1.0)
            if np.any(mask):
                on_data = self.spec_counts[mask]
                on_model = self.model('on', 'mle')[mask]
                r[mask] = quantile_residuals_poisson(
                    on_data,
                    on_model,
                    keep_sign=not random,
                    random=random,
                    seed=seed,
                )

        elif stat in {'pgstat', 'wstat'}:
            upper_mask = pit == 0.0
            if np.any(upper_mask):
                r[upper_mask] = stats.norm.ppf(1.0 / self._nsim)
                upper = np.full(r.shape, False)
                upper[upper_mask] = True

            lower_mask = pit == 1.0
            if np.any(lower_mask):
                r[lower_mask] = stats.norm.ppf(1.0 - 1.0 / self._nsim)
                lower = np.full(r.shape, False)
                lower[lower_mask] = True

        return r, lower, upper


# clean up helpers
del (
    _cached_method,
    _cached_method_with_check,
    _to_cached_method,
    _to_cached_method_with_check,
)

_cached_method = []
_cached_method_with_check = []
_to_cached_method = _get_cached_method_decorator(_cached_method)
_to_cached_method_with_check = _get_cached_method_with_check_decorator(
    _cached_method_with_check, 'ppc'
)


class PosteriorPlotData(PlotData):
    result: PosteriorResult
    _cached_method = _cached_method
    _cached_method_with_check = _cached_method_with_check

    @property
    def params(self) -> dict[str, Array]:
        return self.result._params_dist

    @property
    def ppc(self) -> PPCResult | None:
        return self.result._ppc

    @_to_cached_method
    def get_model_median(self, name: str) -> Array:
        posterior = self.result.idata['posterior'][name]
        return posterior.median(dim=('chain', 'draw')).values

    @_to_cached_method
    def get_model_loo(self, name: str) -> Array:
        posterior = self.result.idata['posterior'][name]
        posterior = posterior.stack(__sample__=('chain', 'draw'))
        return self.result._loo_expectation(posterior, self.name).values

    @_to_cached_method
    def get_model_posterior(self, name: str) -> DataArray:
        posterior = self.result.idata['posterior'][name]
        # return shape (n_samples, n_channel)
        return posterior.stack(__sample__=('chain', 'draw')).T

    def get_model_ppc(self, name: str) -> Array | None:
        if self.ppc is None:
            return None
        else:
            return self.ppc.models_fit[name]

    def get_model_mle(self, name: str) -> Array | None:
        mle = self.result._mle
        if mle is None:
            return None
        else:
            return mle['models'][name]

    @property
    def ce_model(self) -> Array:
        return self.get_model_median(self.name)

    @_to_cached_method
    def ce_model_ci(self, cl: float = 0.683) -> Array:
        assert 0.0 < cl < 1.0
        return np.quantile(
            self.get_model_posterior(self.name).values,
            q=0.5 + cl * np.array([-0.5, 0.5]),
            axis=0,
        )

    def unfolded_model(
        self,
        mtype: Literal['ne', 'ene', 'eene'],
        egrid: Array | None,
        params: dict | None,
        comps: bool,
        cl: float | Array | None = None,
    ) -> tuple[Array | dict, Array | dict | None]:
        assert mtype in {'ne', 'ene', 'eene'}
        if cl is not None:
            cl = np.atleast_1d(cl).astype(float)
            assert np.all(0.0 < cl) and np.all(cl < 1.0)
        params = {} if params is None else dict(params)
        comps = comps and self.has_comps
        egrid = jnp.asarray(egrid, float)
        post_params = self.params
        n = [i.size for i in post_params.values()][0]
        params = {k: jnp.full(n, v) for k, v in params.items()}
        post_params = post_params | params
        models = self._unfolded_model(mtype, egrid, post_params, comps)
        if comps:
            model = {k: np.median(v, axis=0) for k, v in models.items()}
            if cl is None:
                return model, None
            q = 0.5 + cl[:, None] * np.array([-0.5, 0.5])
            ci = {k: np.quantile(v, q, axis=0) for k, v in models.items()}
            return model, ci
        else:
            model = np.median(models, axis=0)
            q = 0.5 + cl[:, None] * np.array([-0.5, 0.5])
            ci = np.quantile(models, q, axis=0)
        return model, ci

    @property
    def sign(self) -> dict[str, Array | None]:
        """Sign of the difference between the data and the fitted models."""
        return {
            'posterior': self._sign_posterior(),
            'loo': self._sign_loo(),
            'median': self._sign_median(),
            'mle': self._sign_mle(),
            'ppc': self._sign_ppc(),
        }

    @_to_cached_method
    def _sign_posterior(self) -> Array:
        ce_posterior = self.get_model_posterior(self.name)
        return np.where(self.ce_data >= ce_posterior, 1.0, -1.0)

    @_to_cached_method
    def _sign_loo(self) -> Array:
        ce_loo = self.get_model_loo(self.name)
        return np.where(self.ce_data >= ce_loo, 1.0, -1.0)

    @_to_cached_method
    def _sign_median(self) -> Array:
        ce_median = self.get_model_median(self.name)
        return np.where(self.ce_data >= ce_median, 1.0, -1.0)

    @_to_cached_method_with_check
    def _sign_mle(self) -> Array | None:
        if self.ppc is None:
            return None

        ce_mle = self.get_model_mle(self.name)
        return np.where(self.ce_data >= ce_mle, 1.0, -1.0)

    @_to_cached_method_with_check
    def _sign_ppc(self) -> Array | None:
        if self.ppc is None:
            return None

        ce_ppc = self.get_model_ppc(self.name)
        return np.where(self.ppc.data[self.name] >= ce_ppc, 1.0, -1.0)

    def model(
        self,
        on_off: Literal['on', 'off'],
        mtype: Literal['posterior', 'loo', 'median', 'mle', 'ppc'],
    ) -> Array | None:
        assert on_off in {'on', 'off'}
        assert mtype in {'posterior', 'loo', 'median', 'mle', 'ppc'}

        if (on_off == 'off') and (self.statistic not in _STATISTIC_WITH_BACK):
            return None

        name = f'{self.name}_N{on_off}_model'
        return getattr(self, f'get_model_{mtype}')(name)

    def deviance(
        self,
        rtype: Literal['posterior', 'loo', 'mle', 'ppc'],
    ) -> DataArray | None:
        """Median, MLE, and ppc deviance."""
        if rtype == 'posterior':
            loglike = self.result.idata['log_likelihood'][self.name]
            return -2.0 * loglike.stack(__sample__=('chain', 'draw')).T
        elif rtype == 'loo':
            loglike = self.result.idata['log_likelihood'][self.name]
            deviance = -2.0 * loglike.stack(__sample__=('chain', 'draw')).T
            return self.result._loo_expectation(deviance, self.name)
        elif rtype == 'mle':
            if self.result._mle is not None:
                return self.result._mle['deviance']['point'][self.name]
            else:
                return None
        elif rtype == 'ppc':
            if self.ppc is not None:
                return self.ppc.deviance['point'][self.name]
            else:
                return None
        else:
            raise ValueError(f'unknown deviance type: {rtype}')

    def pit(self) -> tuple:
        return self.result._loo_pit[self.name]

    def residuals(
        self,
        rtype: Literal['rd', 'rp', 'rq'],
        seed: int | None = None,
        random_quantile: bool = True,
        mle: bool = False,
    ) -> Array | tuple[Array, bool | Array, bool | Array]:
        assert rtype in {'rd', 'rp', 'rq'}

        if rtype == 'rq':
            seed = self.seed if seed is None else int(seed)
            return self.quantile_residuals(seed, random_quantile)
        else:
            point_type = 'mle' if mle else 'loo'
            rname = 'deviance' if rtype == 'rd' else 'pearson'
            return getattr(self, f'{rname}_residuals_{point_type}')()

    def residuals_sim(
        self,
        rtype: Literal['rd', 'rp', 'rq'],
        seed: int | None = None,
        random_quantile: bool = True,
    ) -> Array | None:
        if self.ppc is None or rtype == 'rq':
            return None

        if rtype == 'rd':
            r = self.deviance_residuals_ppc()
        elif rtype == 'rp':
            r = self.pearson_residuals_ppc()
        else:
            raise NotImplementedError(f'{rtype} residual')
        return r

    def residuals_ci(
        self,
        rtype: Literal['rd', 'rp', 'rq'],
        cl: float = 0.683,
        seed: int | None = None,
        random_quantile: bool = True,
        with_sign: bool = False,
    ) -> Array | None:
        if self.ppc is None or rtype == 'rq':
            return None

        assert 0 < cl < 1

        r = self.residuals_sim(rtype, seed, random_quantile)

        if with_sign:
            return np.quantile(r, q=0.5 + cl * np.array([-0.5, 0.5]), axis=0)
        else:
            q = np.quantile(np.abs(r), q=cl, axis=0)
            return np.row_stack([-q, q])

    @_to_cached_method
    def deviance_residuals_loo(self) -> Array:
        return self._deviance_residuals('loo')

    @_to_cached_method
    def deviance_residuals_median(self) -> Array:
        return np.median(self._deviance_residuals('posterior'), axis=0)

    @_to_cached_method_with_check
    def deviance_residuals_mle(self) -> Array:
        return self._deviance_residuals('mle')

    @_to_cached_method_with_check
    def deviance_residuals_ppc(self) -> Array | None:
        if self.ppc is None:
            return None
        return self._deviance_residuals('ppc')

    def _deviance_residuals(
        self, rtype: Literal['loo', 'posterior', 'mle', 'ppc']
    ) -> Array | None:
        if rtype in ['mle', 'ppc'] and self.ppc is None:
            return None

        # NB: if background is present, then this assumes the background is
        #     being profiled out, so that each src & bkg data pair has ~1 dof
        return self.sign[rtype] * np.sqrt(self.deviance(rtype))

    @_to_cached_method
    def pearson_residuals_loo(self) -> Array:
        return self._pearson_residuals('loo')

    @_to_cached_method
    def pearson_residuals_median(self) -> Array:
        return np.median(self._pearson_residuals('posterior'), axis=0)

    @_to_cached_method_with_check
    def pearson_residuals_mle(self) -> Array:
        return self._pearson_residuals('mle')

    @_to_cached_method_with_check
    def pearson_residuals_ppc(self) -> Array | None:
        if self.ppc is None:
            return None
        return self._pearson_residuals('ppc')

    def _pearson_residuals(
        self, rtype: Literal['posterior', 'loo', 'mle', 'ppc']
    ) -> Array | None:
        if rtype in ['mle', 'ppc'] and self.ppc is None:
            return None

        stat = self.statistic
        mtype = 'posterior' if rtype == 'loo' else rtype

        if rtype in {'posterior', 'loo', 'mle'}:
            if stat in _STATISTIC_SPEC_NORMAL:
                on_data = self.net_counts
            else:
                on_data = self.spec_counts
        else:
            on_data = self.ppc.data[f'{self.name}_Non']
        on_model = self.model('on', mtype)

        if stat in _STATISTIC_SPEC_NORMAL:
            std = self.net_errors
        else:
            std = None

        r = pearson_residuals(on_data, on_model, std)

        if stat in _STATISTIC_WITH_BACK:
            if rtype in {'posterior', 'loo', 'mle'}:
                off_data = self.back_counts
            else:
                off_data = self.ppc.data[f'{self.name}_Noff']
            off_model = self.model('off', mtype)

            if stat in _STATISTIC_BACK_NORMAL:
                std = self.back_errors
            else:
                std = None

            r_b = pearson_residuals(off_data, off_model, std)

            # NB: this assumes the background is being profiled out,
            #     so that each src & bkg data pair has ~1 dof
            r = self.sign[rtype] * np.sqrt(r * r + r_b * r_b)

        if rtype == 'loo':
            r = self.result._loo_expectation(np.abs(r), self.name)
            r *= self.sign[rtype]

        return r

    def quantile_residuals(
        self, seed: int, random: bool
    ) -> tuple[Array, Array | bool, Array | bool]:
        pit_minus, pit = self.pit()
        if random:
            pit = np.random.default_rng(seed).uniform(pit_minus, pit)
        r = stats.norm.ppf(pit)

        # Assume the posterior prediction is nchan * ndraw times
        nchain = len(self.result.idata['posterior']['chain'])
        ndraw = len(self.result.idata['posterior']['draw'])
        nsim = nchain * ndraw

        lower = upper = False

        upper_mask = pit == 0.0
        if np.any(upper_mask):
            r[upper_mask] = stats.norm.ppf(1.0 / nsim)
            upper = np.full(r.shape, False)
            upper[upper_mask] = True

        lower_mask = pit == 1.0
        if np.any(lower_mask):
            r[lower_mask] = stats.norm.ppf(1.0 - 1.0 / nsim)
            lower = np.full(r.shape, False)
            lower[lower_mask] = True

        return r, lower, upper


# clean up helpers
del (
    _cached_method,
    _cached_method_with_check,
    _to_cached_method,
    _to_cached_method_with_check,
)
