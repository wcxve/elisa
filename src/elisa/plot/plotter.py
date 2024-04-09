"""Visualize fit and analysis results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cache, wraps
from typing import TYPE_CHECKING, NamedTuple

import jax
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from elisa.infer.likelihood import (
    _STATISTIC_BACK_NORMAL,
    _STATISTIC_SPEC_NORMAL,
    _STATISTIC_WITH_BACK,
)
from elisa.infer.results import check_params
from elisa.plot.misc import plot_corner, plot_trace
from elisa.plot.residuals import (
    pearson_residuals,
    pit_poisson,
    pit_poisson_normal,
    pit_poisson_poisson,
)
from elisa.plot.scale import LinLogScale, get_scale
from elisa.plot.util import get_colors, get_markers

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Callable, Literal

    from matplotlib.pyplot import Axes, Figure

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
    _cached_method: list[str]
    _cached_method_with_check: list[tuple[str, list[str]]]

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

    @property
    def channel(self) -> NumPyArray:
        return self.data.channel

    @property
    def ch_emin(self) -> NumPyArray:
        return self.data.ch_emin

    @property
    def ch_emax(self) -> NumPyArray:
        return self.data.ch_emax

    @property
    def ch_emid(self) -> NumPyArray:
        return self.data.ch_emid

    @property
    def ch_width(self) -> NumPyArray:
        return self.data.ch_width

    @property
    def ch_mean(self) -> NumPyArray:
        return self.data.ch_mean

    @property
    def ch_error(self) -> NumPyArray:
        return self.data.ch_error

    @property
    def ce_data(self) -> Array:
        return self.data.ce

    @property
    def ce_error(self) -> Array:
        return self.data.ce_error

    @property
    def spec_counts(self) -> Array:
        return self.data.spec_counts

    @property
    def spec_error(self) -> Array:
        return self.data.spec_error

    @property
    def back_ratio(self) -> float | Array:
        return self.data.back_ratio

    @property
    def back_counts(self) -> Array | None:
        return self.data.back_counts

    @property
    def back_error(self) -> Array | None:
        return self.data.back_error

    @property
    def net_counts(self) -> Array:
        return self.data.net_counts

    @property
    def net_error(self) -> Array:
        return self.data.net_error

    @property
    def ndata(self) -> int:
        return len(self.data.channel)

    @property
    @abstractmethod
    def ce_model(self) -> Array:
        """Point estimate of the folded source model."""
        pass

    @abstractmethod
    def ce_model_ci(self, cl: float = 0.683) -> Array | None:
        """Confidence/Credible intervals of the folded source model."""
        pass

    @abstractmethod
    def pit(self) -> tuple:
        """Probability integral transform."""
        pass

    @abstractmethod
    def residuals(
        self,
        rtype: Literal['deviance', 'pearson', 'quantile'],
        seed: int | None,
        random_quantile: bool,
        mle: bool,
    ) -> Array | tuple[Array, bool | Array, bool | Array]:
        """Residuals between the data and the fitted models."""
        pass

    @abstractmethod
    def residuals_sim(
        self,
        rtype: Literal['deviance', 'pearson', 'quantile'],
        seed: int | None,
        random_quantile: bool,
    ) -> Array | None:
        """Residuals bootstrap/ppc samples."""
        pass

    @abstractmethod
    def residuals_ci(
        self,
        rtype: Literal['deviance', 'pearson', 'quantile'],
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

    @property
    def on_models(self) -> dict[str, Array | None]:
        """Point estimate and bootstrap sample of the on measurement model."""
        on_name = f'{self.name}_Non_model'
        return {
            'mle': self.get_model_mle(on_name),
            'boot': self.get_model_boot(on_name),
        }

    @property
    def off_models(self) -> dict[str, Array | None]:
        """Point estimate and bootstrap sample of the off measurement model."""
        if self.statistic not in _STATISTIC_WITH_BACK:
            return {'mle': None, 'boot': None}

        off_name = f'{self.name}_Noff_model'
        return {
            'mle': self.get_model_mle(off_name),
            'boot': self.get_model_boot(off_name),
        }

    @property
    def deviance(self) -> dict[str, Array | None]:
        """MLE and bootstrap deviance."""
        mle = self.result._deviance['point'][self.name]
        if self.boot is not None:
            boot = self.boot.deviance['point'][self.name]
        else:
            boot = None
        return {'mle': mle, 'boot': boot}

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
        on_model = self.on_models['mle']

        if stat in _STATISTIC_SPEC_NORMAL:  # chi2
            pit = stats.norm.cdf((on_data - on_model) / self.net_error)
            return pit, pit

        if stat in _STATISTIC_WITH_BACK:
            off_data = self.back_counts
            off_model = self.off_models['mle']

            if stat in _STATISTIC_BACK_NORMAL:  # pgstat
                pit = pit_poisson_normal(
                    k=on_data,
                    lam=on_model,
                    v=off_data,
                    mu=off_model,
                    sigma=self.back_error,
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
        rtype: Literal['deviance', 'pearson', 'quantile'],
        seed: int | None = None,
        random_quantile: bool = True,
        mle: bool = True,
    ) -> Array | tuple[Array, bool | Array, bool | Array]:
        if rtype == 'deviance':
            return self.deviance_residuals_mle()
        elif rtype == 'pearson':
            return self.pearson_residuals_mle()
        elif rtype == 'quantile':
            seed = self.seed if seed is None else int(seed)
            return self.quantile_residuals_mle(seed, random_quantile)
        else:
            raise NotImplementedError(f'{rtype} residual')

    def residuals_sim(
        self,
        rtype: Literal['deviance', 'pearson', 'quantile'],
        seed: int | None = None,
        random_quantile: bool = True,
    ) -> Array | None:
        if self.boot is None or rtype == 'quantile':
            return None

        if rtype == 'deviance':
            r = self.deviance_residuals_boot()
        elif rtype == 'pearson':
            r = self.pearson_residuals_boot()
        else:
            raise NotImplementedError(f'{rtype} residual')

        return r

    def residuals_ci(
        self,
        rtype: Literal['deviance', 'pearson', 'quantile'],
        cl: float = 0.683,
        seed: int | None = None,
        random_quantile: bool = True,
        with_sign: bool = False,
    ) -> Array | None:
        if self.boot is None or rtype == 'quantile':
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
        return self.sign[rtype] * np.sqrt(self.deviance[rtype])

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
            std = self.net_error
        else:
            std = None

        r = pearson_residuals(on_data, self.on_models[rtype], std)

        if stat in _STATISTIC_WITH_BACK:
            if rtype == 'mle':
                off_data = self.back_counts
            else:
                off_data = self.get_data_boot(f'{self.name}_Noff')

            if self.statistic in _STATISTIC_BACK_NORMAL:
                std = self.back_error
            else:
                std = None

            r_b = pearson_residuals(off_data, self.off_models[rtype], std)

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

        if self.statistic in {'pgstat', 'wstat'}:
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
    def ppc(self) -> PPCResult | None:
        return self.result._ppc

    @_to_cached_method
    def get_model_median(self, name: str) -> Array:
        posterior = self.result._idata['posterior'][name]
        return posterior.median(dim=('chain', 'draw')).values

    @_to_cached_method
    def get_model_posterior(self, name: str) -> Array:
        posterior = self.result._idata['posterior'][name].values
        return np.concatenate(posterior)

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
            self.get_model_posterior(self.name),
            q=0.5 + cl * np.array([-0.5, 0.5]),
            axis=0,
        )

    @property
    def sign(self) -> dict[str, Array | None]:
        """Sign of the difference between the data and the fitted models."""
        return {
            'posterior': self._sign_posterior(),
            'median': self._sign_median(),
            'mle': self._sign_mle(),
            'ppc': self._sign_ppc(),
        }

    @_to_cached_method
    def _sign_posterior(self) -> Array:
        ce_posterior = self.get_model_posterior(self.name)
        return np.where(self.ce_data >= ce_posterior, 1.0, -1.0)

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

    @property
    def on_models(self) -> dict[str, Array | None]:
        on_name = f'{self.name}_Non_model'
        return {
            'posterior': self.get_model_posterior(on_name),
            'median': self.get_model_median(on_name),
            'mle': self.get_model_mle(on_name),
            'ppc': self.get_model_ppc(on_name),
        }

    @property
    def off_models(self) -> dict[str, Array | None]:
        if self.statistic not in _STATISTIC_WITH_BACK:
            return {
                'posterior': None,
                'median': None,
                'mle': None,
                'ppc': None,
            }

        off_name = f'{self.name}_Noff_model'
        return {
            'posterior': self.get_model_posterior(off_name),
            'median': self.get_model_median(off_name),
            'mle': self.get_model_mle(off_name),
            'ppc': self.get_model_ppc(off_name),
        }

    @property
    def deviance(self) -> dict[str, Array | None]:
        """Median, MLE, and ppc deviance."""
        loglike = self.result._idata['log_likelihood'][self.name].values
        posterior = -2.0 * np.concatenate(loglike)

        mle = self.result._mle
        if mle is not None:
            mle = mle['deviance']['point'][self.name]

        ppc = self.result._ppc
        if ppc is not None:
            ppc = ppc.deviance['point'][self.name]

        return {'posterior': posterior, 'mle': mle, 'ppc': ppc}

    def pit(self) -> tuple:
        return self.result._loo_pit[self.name]

    def residuals(
        self,
        rtype: Literal['deviance', 'pearson', 'quantile'],
        seed: int | None = None,
        random_quantile: bool = True,
        mle: bool = False,
    ) -> Array | tuple[Array, bool | Array, bool | Array]:
        assert rtype in {'deviance', 'pearson', 'quantile'}

        if rtype == 'quantile':
            seed = self.seed if seed is None else int(seed)
            return self.quantile_residuals(seed, random_quantile)
        else:
            point_type = 'mle' if mle else 'median'
            return getattr(self, f'{rtype}_residuals_{point_type}')()

    def residuals_sim(
        self,
        rtype: Literal['deviance', 'pearson', 'quantile'],
        seed: int | None = None,
        random_quantile: bool = True,
    ) -> Array | None:
        if self.ppc is None or rtype == 'quantile':
            return None

        if rtype == 'deviance':
            r = self.deviance_residuals_ppc()
        elif rtype == 'pearson':
            r = self.pearson_residuals_ppc()
        else:
            raise NotImplementedError(f'{rtype} residual')
        return r

    def residuals_ci(
        self,
        rtype: Literal['deviance', 'pearson', 'quantile'],
        cl: float = 0.683,
        seed: int | None = None,
        random_quantile: bool = True,
        with_sign: bool = False,
    ) -> Array | None:
        if self.ppc is None or rtype == 'quantile':
            return None

        assert 0 < cl < 1

        r = self.residuals_sim(rtype, seed, random_quantile)

        if with_sign:
            return np.quantile(r, q=0.5 + cl * np.array([-0.5, 0.5]), axis=0)
        else:
            q = np.quantile(np.abs(r), q=cl, axis=0)
            return np.row_stack([-q, q])

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
        self, rtype: Literal['posterior', 'mle', 'ppc']
    ) -> Array | None:
        if rtype in ['mle', 'ppc'] and self.ppc is None:
            return None

        # NB: if background is present, then this assumes the background is
        #     being profiled out, so that each src & bkg data pair has ~1 dof
        return self.sign[rtype] * np.sqrt(self.deviance[rtype])

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
        self, rtype: Literal['posterior', 'mle', 'ppc']
    ) -> Array | None:
        if rtype in ['mle', 'ppc'] and self.ppc is None:
            return None

        stat = self.statistic

        if rtype in {'posterior', 'mle'}:
            if stat in _STATISTIC_SPEC_NORMAL:
                on_data = self.net_counts
            else:
                on_data = self.spec_counts
        else:
            on_data = self.ppc.data[f'{self.name}_Non']
        on_model = self.on_models[rtype]

        if stat in _STATISTIC_SPEC_NORMAL:
            std = self.net_error
        else:
            std = None

        r = pearson_residuals(on_data, on_model, std)

        if stat in _STATISTIC_WITH_BACK:
            if rtype in {'posterior', 'mle'}:
                off_data = self.back_counts
            else:
                off_data = self.ppc.data[f'{self.name}_Noff']
            off_model = self.off_models[rtype]

            if self.statistic in _STATISTIC_BACK_NORMAL:
                std = self.back_error
            else:
                std = None

            r_b = pearson_residuals(off_data, off_model, std)

            # NB: this assumes the background is being profiled out,
            #     so that each src & bkg data pair has ~1 dof
            r = self.sign[rtype] * np.sqrt(r * r + r_b * r_b)

        return r

    def quantile_residuals(
        self, seed: int, random: bool
    ) -> tuple[Array, Array | bool, Array | bool]:
        pit_minus, pit = self.pit()
        if random:
            pit = np.random.default_rng(seed).uniform(pit_minus, pit)
        r = stats.norm.ppf(pit)

        # Assume the posterior prediction is nchan * ndraw times
        nchain = len(self.result._idata['posterior']['chain'])
        ndraw = len(self.result._idata['posterior']['draw'])
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


class PlotConfig(NamedTuple):
    """Plotting configuration."""

    alpha: float = 0.8
    palette: Any = 'colorblind'
    xscale: Literal['linear', 'log'] = 'log'
    yscale: Literal['linear', 'log', 'linlog'] = 'linlog'
    lin_frac: float = 0.15
    cl: tuple[float, ...] = (0.683, 0.95)
    default_residuals: Literal['deviance', 'pearson', 'quantile'] = 'quantile'
    random_quantile: bool = False
    mark_outlier_residuals: bool = True
    residuals_ci_with_sign: bool = True


class Plotter(ABC):
    """Plotter to visualize analysis results."""

    data: dict[str, PlotData] | None = None

    def __init__(self, result: FitResult, config: PlotConfig = None):
        self._result = result
        self.data = self.get_plot_data(result)
        self.config = config
        markers = get_markers(len(self.data))
        self._markers = dict(zip(self.data.keys(), markers))

    @staticmethod
    @abstractmethod
    def get_plot_data(result: FitResult) -> dict[str, PlotData]:
        """Get PlotData from FitResult."""
        pass

    @property
    def config(self) -> PlotConfig:
        """Plotting configuration."""
        return self._config

    @config.setter
    def config(self, config: PlotConfig):
        if config is None:
            config = PlotConfig()
        elif not isinstance(config, PlotConfig):
            raise TypeError('config must be a PlotConfig instance')

        self._config = config
        self._set_colors(config.palette)

    def _set_colors(self, palette: Any):
        colors = get_colors(len(self.data), palette=palette)
        self._colors = dict(zip(self.data.keys(), colors))

    @property
    def ndata(self):
        ndata = {name: data.ndata for name, data in self.data.items()}
        ndata['total'] = sum(ndata.values())
        return ndata

    def plot(self, *args, r=None, **kwargs) -> tuple[Figure, np.ndarray[Axes]]:
        config = self.config
        fig, axs = plt.subplots(
            nrows=2,
            ncols=1,
            sharex='all',
            height_ratios=[1.618, 1.0],
            gridspec_kw={'hspace': 0.03},
            figsize=(8, 6),
        )

        fig.align_ylabels(axs)

        for ax in axs:
            ax.tick_params(
                axis='both',
                which='both',
                direction='in',
                bottom=True,
                top=True,
                left=True,
                right=True,
            )

        plt.rcParams['axes.formatter.min_exponent'] = 3

        axs[-1].set_xlabel(r'$\mathrm{Energy\ [keV]}$')

        ylabels = {
            'ce': r'$C_E\ \mathrm{[s^{-1}\ keV^{-1}]}$',
            'residuals': r'$r_D\ [\mathrm{\sigma}]$',
            'ne': r'$N_E\ \mathrm{[s^{-1}\ cm^{-2}\ keV^{-1}]}$',
            'ene': r'$E N_E\ \mathrm{[erg\ s^{-1}\ cm^{-2}\ keV^{-1}]}$',
            'eene': r'$E^2 N_E\ \mathrm{[erg\ s^{-1}\ cm^{-2}]}$',
            'Fv': r'$F_{\nu}\ \mathrm{[erg\ s^{-1}\ cm^{-2}\ keV^{-1}]}$',
            'vFv': r'$\nu F_{\nu}\ \mathrm{[erg\ s^{-1}\ cm^{-2}]}$',
        }
        axs[0].set_ylabel(ylabels['ce'])
        axs[1].set_ylabel(ylabels['residuals'])

        self.plot_ce_model(axs[0])
        self.plot_ce_data(axs[0])
        self.plot_residuals(axs[1], r)

        axs[0].set_xscale(config.xscale)
        ax = axs[0]
        xmin, xmax = ax.dataLim.intervalx
        ax.set_xlim(xmin * 0.97, xmax * 1.06)

        yscale = config.yscale
        assert yscale in {'linear', 'log', 'linlog'}
        if yscale in {'linear', 'log'}:
            ax.set_yscale(yscale)
        else:
            ax.set_yscale('log')
            lin_thresh = ax.get_ylim()[0]
            lin_frac = config.lin_frac
            dmin, dmax = ax.get_yaxis().get_data_interval()
            scale = LinLogScale(
                axis=None,
                base=10.0,
                lin_thresh=lin_thresh,
                lin_scale=get_scale(10.0, lin_thresh, dmin, dmax, lin_frac),
            )
            ax.set_yscale(scale)
            ax.axhline(lin_thresh, c='k', lw=0.15, ls=':', zorder=-1)

        axs[0].legend()

        for ax in axs:
            ax.relim()
            ax.autoscale_view()

        return fig, axs

    def plot_ce_model(self, ax: Axes):
        config = self.config
        cl = np.atleast_1d(config.cl)
        step_kwargs = {'lw': 1.618, 'alpha': config.alpha}
        ribbon_kwargs = {'lw': 0.618, 'alpha': 0.2 * config.alpha}

        for name, data in self.data.items():
            color = self._colors[name]

            _plot_step(
                ax,
                data.ch_emin,
                data.ch_emax,
                data.ce_model,
                color=color,
                **step_kwargs,
            )

            quantiles = []
            for i_cl in cl:
                if (q := data.ce_model_ci(i_cl)) is not None:
                    quantiles.append(q)

            if quantiles:
                _plot_ribbon(
                    ax,
                    data.ch_emin,
                    data.ch_emax,
                    quantiles,
                    color=color,
                    **ribbon_kwargs,
                )

    def plot_ce_data(self, ax: Axes):
        config = self.config
        alpha = config.alpha
        xlog = config.xscale == 'log'

        for name, data in self.data.items():
            color = self._colors[name]
            marker = self._markers[name]
            ax.errorbar(
                x=data.ch_mean if xlog else data.ch_emid,
                xerr=data.ch_error if xlog else 0.5 * data.ch_width,
                y=data.ce_data,
                yerr=data.ce_error,
                alpha=alpha,
                color=color,
                fmt=f'{marker} ',
                label=name,
                lw=0.75,
                ms=2.4,
                mec=color,
                mfc='#FFFFFFCC',
            )

    def plot_residuals(
        self,
        ax: Axes,
        rtype: Literal['deviance', 'pearson', 'quantile'] | None = None,
        seed: int | None = None,
    ):
        config = self.config
        cl = np.sort(np.atleast_1d(config.cl))
        random_quantile = config.random_quantile
        with_sign = config.residuals_ci_with_sign
        mark_outlier = config.mark_outlier_residuals
        ribbon_kwargs = {'lw': 0.618, 'alpha': 0.15 * config.alpha}

        if rtype is None:
            rtype = config.default_residuals

        alpha = config.alpha
        xlog = config.xscale == 'log'

        normal_q = stats.norm.isf(0.5 * (1.0 - cl))

        for name, data in self.data.items():
            color = self._colors[name]
            marker = self._markers[name]
            x = data.ch_mean if xlog else data.ch_emid
            xerr = data.ch_error if xlog else 0.5 * data.ch_width

            quantiles = []
            for i_cl in cl:
                q = data.residuals_ci(
                    rtype, i_cl, seed, random_quantile, with_sign
                )
                if q is not None:
                    quantiles.append(q)

            if quantiles:
                _plot_ribbon(
                    ax,
                    data.ch_emin,
                    data.ch_emax,
                    quantiles,
                    color=color,
                    **ribbon_kwargs,
                )
            else:
                for q in normal_q:
                    ax.fill_between(
                        [data.ch_emin[0], data.ch_emax[-1]],
                        -q,
                        q,
                        color=color,
                        **ribbon_kwargs,
                    )

            use_mle = True if quantiles else False
            r = data.residuals(rtype, seed, config.random_quantile, use_mle)
            if rtype == 'quantile':
                r, lower, upper = r
            else:
                lower = upper = False
            ax.errorbar(
                x=x,
                y=r,
                yerr=1.0,
                xerr=xerr,
                color=color,
                alpha=alpha,
                linewidth=0.75,
                linestyle='',
                marker=marker,
                markersize=2.4,
                markeredgecolor=color,
                markerfacecolor='#FFFFFFCC',
                lolims=lower,
                uplims=upper,
            )

            if mark_outlier:
                if quantiles:
                    q = quantiles[-1]
                else:
                    q = [-normal_q[-1], normal_q[-1]]
                mask = (r < q[0]) | (r > q[1])
                ax.scatter(x[mask], r[mask], marker='x', c='r')

        for q in normal_q:
            ax.axhline(q, ls=':', lw=1, c='gray', zorder=0)
            ax.axhline(-q, ls=':', lw=1, c='gray', zorder=0)

        ax.axhline(0, ls='--', lw=1, c='gray', zorder=0)
        yabs_max = abs(max(ax.get_ylim(), key=abs))
        ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)

    def plot_qq(
        self,
        rtype: Literal['deviance', 'pearson', 'quantile'] | None = None,
        seed: int | None = None,
        detrend: bool = True,
    ):
        """Quantile-Quantile plot."""
        config = self.config
        random_quantile = config.random_quantile
        if rtype is None:
            rtype = config.default_residuals

        rsim = {
            name: data.residuals_sim(rtype, seed, random_quantile)
            for name, data in self.data.items()
        }
        if any(i is None for i in rsim.values()):
            rsim['total'] = None
        else:
            rsim['total'] = np.hstack(list(rsim.values()))

        use_mle = True if rsim else False
        r = {
            name: data.residuals(rtype, seed, random_quantile, use_mle)
            for name, data in self.data.items()
        }
        if rtype == 'quantile':
            r = {k: v[0] for k, v in r.items()}
        r['total'] = np.hstack(list(r.values()))

        n_subplots = len(self.data)
        if n_subplots == 1:
            ncols = 1
        else:
            ncols = n_subplots // 2
            if n_subplots % 2:
                ncols += 1

        fig = plt.figure(figsize=(4 + ncols * 2.25, 4), tight_layout=True)
        gs1 = fig.add_gridspec(1, 2, width_ratios=[4, ncols * 2.25])
        gs2 = gs1[0, 1].subgridspec(2, ncols, wspace=0.35)
        ax1 = fig.add_subplot(gs1[0, 0])
        axs = gs2.subplots(squeeze=False)
        ax1.set_xlabel('Normal Theoretical Quantiles')
        ax1.set_ylabel('Residuals')

        alpha = config.alpha
        ha = 'center' if detrend else 'left'
        text_x = 0.5 if detrend else 0.03

        axs = [ax1] + axs.ravel().tolist()
        names = ['total'] + list(self.ndata.keys())
        colors = ['k'] + get_colors(n_subplots, config.palette)
        for ax, name, color in zip(axs, names, colors):
            theor, q, line, lo, up = get_qq(r[name], detrend, 0.95, rsim[name])
            ax.scatter(theor, q, s=5, color=color, alpha=alpha)
            ax.plot(theor, line, ls='--', color=color, alpha=alpha)
            ax.plot(theor, lo, ls=':', color=color, alpha=alpha)
            ax.plot(theor, up, ls=':', color=color, alpha=alpha)
            ax.fill_between(
                theor, lo, up, alpha=0.2 * alpha, color=color, lw=0.0
            )
            ax.annotate(
                name,
                xy=(text_x, 0.97),
                xycoords='axes fraction',
                ha=ha,
                va='top',
                color=color,
            )
        if n_subplots % 2:
            axs[-1].set_visible(False)

    def plot_pit(self, detrend=True):
        """Probability integral transformation empirical CDF plot."""
        config = self.config

        pit = {name: data.pit()[1] for name, data in self.data.items()}
        pit['total'] = np.hstack(list(pit.values()))

        n_subplots = len(self.data)
        if n_subplots == 1:
            ncols = 1
        else:
            ncols = n_subplots // 2
            if n_subplots % 2:
                ncols += 1

        fig = plt.figure(figsize=(4 + ncols * 2.25, 4), tight_layout=True)
        gs1 = fig.add_gridspec(1, 2, width_ratios=[4, ncols * 2.25])
        gs2 = gs1[0, 1].subgridspec(2, ncols, wspace=0.35)
        ax1 = fig.add_subplot(gs1[0, 0])
        axs = gs2.subplots(squeeze=False)
        ax1.set_xlabel('Scaled Rank')
        ax1.set_ylabel('PIT ECDF')

        alpha = config.alpha
        ha = 'right' if detrend else 'left'
        text_x = 0.97 if detrend else 0.03

        ax_list = [ax1] + axs.ravel().tolist()
        names = ['total'] + list(self.ndata.keys())
        colors = ['k'] + get_colors(n_subplots, config.palette)

        for ax, name, color in zip(ax_list, names, colors):
            x, y, line, lower, upper = get_pit_ecdf(pit[name], 0.95, detrend)
            ax.plot(x, line, ls='--', color=color, alpha=alpha)
            ax.fill_between(
                x, lower, upper, alpha=0.2 * alpha, color=color, step='mid'
            )
            ax.step(x, y, alpha=alpha, color=color, where='mid')
            ax.annotate(
                text=name,
                xy=(text_x, 0.97),
                xycoords='axes fraction',
                ha=ha,
                va='top',
                color=color,
            )
        if n_subplots % 2:
            ax_list[-1].set_visible(False)

    # def plot_ne(self, ax: Axes):
    #     pass
    #
    # def plot_ene(self, ax: Axes):
    #     pass
    #
    # def plot_eene(self, ax: Axes):
    #     pass
    #
    # def plot_ufspec(self):
    #     pass
    #
    # def plot_eufspec(self):
    #     pass
    #
    # def plot_eeufspec(self):
    #     pass


class MLEResultPlotter(Plotter):
    data: dict[str, MLEPlotData]
    _result: MLEResult

    @staticmethod
    def get_plot_data(result: MLEResult) -> dict[str, MLEPlotData]:
        helper = result._helper
        keys = jax.random.split(
            jax.random.PRNGKey(helper.seed['resd']), len(helper.data_names)
        )
        data = {
            name: MLEPlotData(name, result, int(key[0]))
            for name, key in zip(helper.data_names, keys)
        }
        return data

    # def plot_corner(self):
    #     # profile and contour
    #     ...


class PosteriorResultPlotter(Plotter):
    data: dict[str, PosteriorPlotData]
    _result: PosteriorResult

    @staticmethod
    def get_plot_data(result: PosteriorResult) -> dict[str, PosteriorPlotData]:
        helper = result._helper
        keys = jax.random.split(
            jax.random.PRNGKey(helper.seed['resd']), len(helper.data_names)
        )
        data = {
            name: PosteriorPlotData(name, result, int(key[0]))
            for name, key in zip(helper.data_names, keys)
        }
        return data

    def plot_trace(
        self,
        params: str | Sequence[str] | None = None,
        fig_path: str | None = None,
    ):
        helper = self._result._helper
        params = check_params(params, helper)
        axes_scale = [
            'log' if helper.params_log[p] else 'linear' for p in params
        ]
        labels = [
            f'${helper.params_comp_latex[p]}$  ${helper.params_latex[p]}$'
            + (f'\n[{u}]' if (u := helper.params_unit[p]) else '')
            for p in params
        ]
        fig = plot_trace(self._result._idata, params, axes_scale, labels)
        if fig_path:
            fig.savefig(fig_path, bbox_inches='tight')

    def plot_corner(
        self,
        params: str | Sequence[str] | None = None,
        color: str | None = None,
        divergences: bool = True,
        fig_path: str | None = None,
    ):
        helper = self._result._helper
        params = check_params(params, helper)
        axes_scale = [
            'log' if helper.params_log[p] else 'linear' for p in params
        ]
        titles = [
            f'${helper.params_comp_latex[p]}$  ${helper.params_latex[p]}$'
            for p in params
        ]
        labels = [
            f'${helper.params_comp_latex[p]}$  ${helper.params_latex[p]}$'
            + (f'\n[{u}]' if (u := helper.params_unit[p]) else '')
            for p in params
        ]
        fig = plot_corner(
            idata=self._result._idata,
            params=params,
            axes_scale=axes_scale,
            levels=self.config.cl,
            titles=titles,
            labels=labels,
            color=color,
            divergences=divergences,
        )
        if fig_path:
            fig.savefig(fig_path, bbox_inches='tight')

    def plot_khat(self):
        config = self.config
        alpha = config.alpha
        xlog = config.xscale == 'log'

        fig, ax = plt.subplots(1, 1, squeeze=True)

        khat = self._result.loo.pareto_k
        if np.any(khat.values > 0.7):
            ax.axhline(0.7, color='r', lw=0.5, ls=':')

        for name, data in self.data.items():
            color = self._colors[name]
            marker = self._markers[name]
            khat_data = khat.sel(channel=data.channel).values
            x = data.ch_mean if xlog else data.ch_emid
            ax.errorbar(
                x=x,
                xerr=data.ch_error if xlog else 0.5 * data.ch_width,
                y=khat_data,
                alpha=alpha,
                color=color,
                fmt=f'{marker} ',
                label=name,
                lw=0.75,
                ms=2.4,
                mec=color,
                mfc='#FFFFFFCC',
            )

            mask = khat_data > 0.7
            if np.any(mask):
                ax.scatter(x=x[mask], y=khat_data[mask], marker='x', c='r')

        ax.set_xscale('log')
        ax.set_xlabel('Energy [keV]')
        ax.set_ylabel(r'Shape Parameter $\hat{k}$')


def _plot_step(
    ax: Axes, x_left: Array, x_right: Array, y: Array, **step_kwargs
) -> None:
    assert len(y) == len(x_left) == len(x_right)

    step_kwargs['where'] = 'post'

    mask = x_left[1:] != x_right[:-1]
    idx = np.insert(np.flatnonzero(mask) + 1, 0, 0)
    idx = np.append(idx, len(y))
    for i in range(len(idx) - 1):
        i_slice = slice(idx[i], idx[i + 1])
        x_slice = np.append(x_left[i_slice], x_right[i_slice][-1])
        y_slice = y[i_slice]
        y_slice = np.append(y_slice, y_slice[-1])
        ax.step(x_slice, y_slice, **step_kwargs)


def _plot_ribbon(
    ax,
    x_left: Array,
    x_right: Array,
    y_ribbons: Sequence[Array],
    **ribbon_kwargs,
) -> None:
    y_ribbons = list(map(np.asarray, y_ribbons))
    shape = y_ribbons[0].shape
    assert len(shape) == 2 and shape[0] == 2
    assert shape[1] == len(x_left) == len(x_right)
    assert all(ribbon.shape == shape for ribbon in y_ribbons)

    ribbon_kwargs['step'] = 'post'

    mask = x_left[1:] != x_right[:-1]
    idx = np.insert(np.flatnonzero(mask) + 1, 0, 0)
    idx = np.append(idx, shape[1])
    for i in range(len(idx) - 1):
        i_slice = slice(idx[i], idx[i + 1])
        x_slice = np.append(x_left[i_slice], x_right[i_slice][-1])

        for ribbon in y_ribbons:
            lower = ribbon[0]
            lower_slice = lower[i_slice]
            lower_slice = np.append(lower_slice, lower_slice[-1])
            upper = ribbon[1]
            upper_slice = upper[i_slice]
            upper_slice = np.append(upper_slice, upper_slice[-1])
            ax.fill_between(x_slice, lower_slice, upper_slice, **ribbon_kwargs)


def get_qq(
    q: NumPyArray,
    detrend: bool,
    cl: float,
    qsim: NumPyArray | None = None,
) -> tuple[NumPyArray, ...]:
    """Get the Q-Q and pointwise confidence/credible interval.

    References
    ----------
    .. [1] doi:10.1080/00031305.2013.847865
    """
    # https://stats.stackexchange.com/a/9007
    # https://stats.stackexchange.com/a/152834
    alpha = np.pi / 8  # 3/8 is also ok
    n = len(q)
    theor = stats.norm.ppf((np.arange(1, n + 1) - alpha) / (n - 2 * alpha + 1))

    q = np.sort(q)
    if qsim is not None:
        line, lower, upper = np.quantile(
            np.sort(qsim, axis=1),
            q=[0.5, 0.5 - 0.5 * cl, 0.5 + 0.5 * cl],
            axis=0,
        )
    else:
        line = np.array(theor)
        grid = np.arange(1, n + 1)
        lower = stats.beta.ppf(0.5 - cl * 0.5, grid, n + 1 - grid)
        upper = stats.beta.ppf(0.5 + cl * 0.5, grid, n + 1 - grid)
        lower = stats.norm.ppf(lower)
        upper = stats.norm.ppf(upper)

    if detrend:
        q -= theor
        line -= theor
        lower -= theor
        upper -= theor

    return theor, q, line, lower, upper


def get_pit_ecdf(
    pit: NumPyArray,
    cl: float,
    detrend: bool,
) -> tuple[NumPyArray, ...]:
    """Get the empirical CDF of PIT and pointwise confidence/credible interval.

    References
    ----------
    .. [1] doi:10.1007/s11222-022-10090-6
    """
    n = len(pit)

    # See ref [1] for the following
    scaled_rank = np.linspace(0.0, 1.0, n + 1)
    # Since binomial is discrete, we need to have lower and upper bounds with
    # a confidence/credible level >= cl to ensure the nominal coverage,
    # that is, we require that (cdf <= 0.5 - 0.5 * cl) for lower bound
    # and (0.5 + 0.5 * cl <= cdf) for upper bound
    lower_q = 0.5 - cl * 0.5
    lower = stats.binom.ppf(lower_q, n, scaled_rank)
    mask = stats.binom.cdf(lower, n, scaled_rank) > lower_q
    lower[mask] -= 1.0
    lower = np.clip(lower / n, 0.0, 1.0)

    upper_q = 0.5 + cl * 0.5
    upper = stats.binom.ppf(upper_q, n, scaled_rank)
    mask = stats.binom.cdf(upper, n, scaled_rank) < upper_q
    upper[mask] += 1.0
    upper = np.clip(upper / n, 0.0, 1.0)

    line = scaled_rank
    pit_ecdf = np.count_nonzero(pit <= scaled_rank[:, None], axis=1) / n

    if detrend:
        lower -= line
        upper -= line
        pit_ecdf -= line
        line = np.zeros_like(line)

    return scaled_rank, pit_ecdf, line, lower, upper

    # x = np.hstack([0.0, np.sort(pit), 1.0])
    # pit_ecdf = np.hstack([0.0, np.arange(n) / n, 1.0])
    # line = scaled_rank
    #
    # if detrend:
    #     pit_ecdf -= x
    #     lower -= scaled_rank
    #     upper -= scaled_rank
    #     line = np.zeros_like(scaled_rank)
    #
    # return x, pit_ecdf, scaled_rank, line, lower, upper


# def get_pit_pdf(pit_intervals: NumPyArray) -> NumPyArray:
#     """Get the pdf of PIT.
#
#     References
#     ----------
#     .. [1] doi:10.1111/j.1541-0420.2009.01191.x
#     """
#     assert len(pit_intervals.shape) == 2 and pit_intervals.shape[1] == 2
#
#     grid = np.unique(pit_intervals)
#     if grid[0] > 0.0:
#         grid = np.insert(grid, 0, 0)
#     if grid[-1] < 1.0:
#         grid = np.append(grid, 1.0)
#
#     n = len(pit_intervals)
#     mask = pit_intervals[:, 0] != pit_intervals[:, 1]
#     cover_mask = np.bitwise_and(
#         pit_intervals[:, :1] <= grid[:-1],
#         grid[1:] <= pit_intervals[:, 1:],
#     )
#     pdf = np.zeros((n, len(grid) - 1))
#     pdf[cover_mask] = np.repeat(
#         1.0 / (pit_intervals[mask, 1] - pit_intervals[mask, 0]),
#         np.count_nonzero(cover_mask[mask], axis=1),
#     )
#     idx = np.clip(grid.searchsorted(pit_intervals[~mask, 0]) - 1, 0, None)
#     pdf[~mask, idx] = 1.0 / (grid[idx + 1] - grid[idx])
#     return pdf.mean(0)
