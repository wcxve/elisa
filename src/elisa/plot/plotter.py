"""Visualize fit and analysis results."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import arviz as az
import jax
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from elisa.infer.helper import check_params
from elisa.plot.data import MLEPlotData, PosteriorPlotData
from elisa.plot.misc import plot_corner, plot_trace
from elisa.plot.scale import LinLogScale, get_scale
from elisa.plot.util import get_colors, get_markers

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any, Literal

    from matplotlib.pyplot import Axes, Figure

    from elisa.infer.results import FitResult, MLEResult, PosteriorResult
    from elisa.plot.data import PlotData
    from elisa.util.typing import Array, NumPyArray


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


def _adjust_log_range(
    ax: Axes,
    axis: Literal['x', 'y'] = 'y',
    octave: int = 4,
) -> None:
    octave = round(octave)
    assert octave > 0
    if axis == 'y':
        vmin, vmax = ax.dataLim.intervaly
        set_lim = ax.set_ylim
    else:
        vmin, vmax = ax.dataLim.intervalx
        set_lim = ax.set_xlim

    if np.log10(vmax / max(1e-30, vmin)) > int(octave):
        vmin = vmax / 10**octave

    set_lim(
        np.power(10, np.log10(vmin) - 0.05), np.power(10, np.log10(vmax) + 0.1)
    )


def _get_qq(
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


def _get_pit_ecdf(
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


# def _get_pit_pdf(pit_intervals: NumPyArray) -> NumPyArray:
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


class PlotConfig:
    """Plotting configuration."""

    _YLABLES = {
        'ce': r'$C_E\ \mathrm{[count\ s^{-1}\ keV^{-1}]}$',
        'ne': r'$N_E\ \mathrm{[ph\ cm^{-2}\ s^{-1}\ keV^{-1}]}$',
        'ene': r'$E N_E\ \mathrm{[erg\ cm^{-2}\ s^{-1}\ keV^{-1}]}$',
        'Fv': r'$F_{\nu}\ \mathrm{[erg\ cm^{-2}\ s^{-1}\ keV^{-1}]}$',
        'eene': r'$E^2 N_E\ \mathrm{[erg\ cm^{-2}\ s^{-1}]}$',
        'vFv': r'$\nu F_{\nu}\ \mathrm{[erg\ cm^{-2}\ s^{-1}]}$',
        'rd': r'$r_D\ [\mathrm{\sigma}]$',
        'rp': r'$r_\mathrm{P}\ [\mathrm{\sigma}]$',
        'rq': r'$r_Q\ [\mathrm{\sigma}]$',
    }

    def __init__(
        self,
        alpha: float = 0.8,
        palette: Any = 'colorblind',
        xscale: Literal['linear', 'log'] = 'log',
        yscale: Literal['linear', 'log', 'linlog'] = 'linlog',
        lin_frac: float = 0.15,
        cl: tuple[float, ...] = (0.683, 0.95),
        residuals: Literal['rd', 'rp', 'rq'] = 'rq',
        random_quantile: bool = False,
        mark_outlier_residuals: bool = False,
        residuals_ci_with_sign: bool = True,
        plot_comps: bool = False,
        seed: int | None = None,
    ):
        self.alpha = alpha
        self.palette = palette
        self.xscale = xscale
        self.yscale = yscale
        self.lin_frac = lin_frac
        self.cl = cl
        self.residuals = residuals
        self.random_quantile = random_quantile
        self.mark_outlier_residuals = mark_outlier_residuals
        self.residuals_ci_with_sign = residuals_ci_with_sign
        self.plot_comps = plot_comps
        self.seed = seed

    @property
    def alpha(self) -> float:
        """Transparency of colors."""
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float):
        alpha = float(alpha)
        if not 0.0 < alpha <= 1.0:
            raise ValueError('alpha must be in (0, 1]')
        self._alpha = alpha

    @property
    def palette(self) -> Any:
        """Color palettes, see [1]_ for details.

        References
        ----------
        .. [1] `seaborn tutorial: Choosing color palettes
                <https://seaborn.pydata.org/tutorial/color_palettes.html>`__
        """
        return self._palette

    @palette.setter
    def palette(self, palette: Any):
        self._palette = palette

    @property
    def xscale(self) -> Literal['linear', 'log']:
        """X-axis scale of spectral plot.

        Should be ``'linear'``, or ``'log'``.
        """
        return self._xscale

    @xscale.setter
    def xscale(self, xscale: Literal['linear', 'log']):
        if xscale not in {'linear', 'log'}:
            raise ValueError('xscale must be "linear" or "log"')
        self._xscale = xscale

    @property
    def yscale(self) -> Literal['linear', 'log', 'linlog']:
        """X-axis scale of spectral plot.

        Should be ``'linear'``, ``'log'``, or ``'linlog'``.
        """
        return self._yscale

    @yscale.setter
    def yscale(self, yscale: Literal['linear', 'log', 'linlog']):
        if yscale not in {'linear', 'log', 'linlog'}:
            raise ValueError('yscale must be "linear", "log", or "linlog"')
        self._yscale = yscale

    @property
    def lin_frac(self) -> float:
        """Linear fraction of the ``linlog`` plot."""
        return self._lin_frac

    @lin_frac.setter
    def lin_frac(self, lin_frac: float):
        lin_frac = float(lin_frac)
        if not 0.0 < lin_frac <= 0.5:
            raise ValueError('lin_frac must be in (0, 0.5]')
        self._lin_frac = lin_frac

    @property
    def cl(self) -> NumPyArray:
        """Confidence/Credible level."""
        return self._cl

    @cl.setter
    def cl(self, cl: float | Sequence[float]):
        cl = np.sort(np.atleast_1d(cl)).astype(float)
        for c in cl:
            if not 0.0 < c < 1.0:
                raise ValueError('cl must be in (0, 1)')
        self._cl = cl

    @property
    def residuals(self) -> Literal['rd', 'rp', 'rq']:
        """Default type of residual plot."""
        return self._residuals

    @residuals.setter
    def residuals(self, residuals: Literal['rd', 'rp', 'rq']):
        if residuals not in {'rd', 'rp', 'rq'}:
            raise ValueError(
                'residuals type must be "rd" (deviance), "rp" (pearson), or '
                '"rq" (quantile)'
            )
        self._residuals = residuals

    @property
    def random_quantile(self) -> bool:
        """Whether to randomize the quantile residual."""
        return self._random_quantile

    @random_quantile.setter
    def random_quantile(self, random_quantile: bool):
        self._random_quantile = bool(random_quantile)

    @property
    def mark_outlier_residuals(self) -> bool:
        """Whether to mark outlier residuals with red crosses."""
        return self._mark_outlier_residuals

    @mark_outlier_residuals.setter
    def mark_outlier_residuals(self, mark_outlier_residuals: bool):
        self._mark_outlier_residuals = bool(mark_outlier_residuals)

    @property
    def residuals_ci_with_sign(self) -> bool:
        """Whether to take account residuals' sign when calculate CI bands."""
        return self._residuals_ci_with_sign

    @residuals_ci_with_sign.setter
    def residuals_ci_with_sign(self, residuals_ci_with_sign: bool):
        self._residuals_ci_with_sign = bool(residuals_ci_with_sign)

    @property
    def plot_comps(self) -> bool:
        """Whether to plot additive components in spectral plot."""
        return self._plot_comps

    @plot_comps.setter
    def plot_comps(self, plot_comps: bool):
        self._plot_comps = bool(plot_comps)

    @property
    def seed(self) -> int | None:
        """Random seed used in calculation."""
        return self._seed

    @seed.setter
    def seed(self, seed: int | None):
        if seed is not None:
            seed = int(seed)
        self._seed = seed


class Plotter(ABC):
    """Plotter to visualize fit results."""

    _palette: Any | None = None
    _comps_latex: dict[str, str] | None = None
    _params_latex: dict[str, str] | None = None
    _supported: tuple[str, ...]
    data: dict[str, PlotData] | None = None

    def __init__(self, result: FitResult, config: PlotConfig = None):
        self._result = result
        self.data = self.get_plot_data(result)
        self.config = config
        markers = get_markers(len(self.data))
        self._markers = dict(zip(self.data.keys(), markers))

    @abstractmethod
    def __call__(self, plots: str = 'data ne r') -> dict[str, Figure]:
        pass

    @abstractmethod
    def plot_corner(self, *args, **kwargs) -> Figure:
        """Corner plot of bootstrap/posterior parameters."""
        pass

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

    @property
    def colors(self):
        """Plotting color for each data."""
        if self._palette != self.config.palette:
            colors = get_colors(len(self.data), palette=self.config.palette)
            self._colors = dict(zip(self.data.keys(), colors))
            self._palette = self.config.palette
        return self._colors

    @property
    def ndata(self):
        """Data points number."""
        ndata = {name: data.ndata for name, data in self.data.items()}
        ndata['total'] = sum(ndata.values())
        return ndata

    @property
    def comps_latex(self) -> dict[str, str]:
        """LaTeX representation of components."""
        if self._comps_latex is None:
            self._comps_latex = {
                k: f'${v}$ ' if v else ''
                for k, v in self._result._helper.params_comp_latex.items()
            }
        return self._comps_latex

    @property
    def params_latex(self) -> dict[str, str]:
        """LaTeX representation of parameters."""
        if self._params_latex is None:
            self._params_latex = {
                k: f'${v}$'
                for k, v in self._result._helper.params_latex.items()
            }
        return self._params_latex

    @property
    def params_unit(self) -> dict[str, str]:
        """Unit of parameters."""
        return self._result._helper.params_unit

    @property
    def params_titles(self) -> dict[str, str]:
        """Title of parameters."""
        comps_latex = self.comps_latex
        params_latex = self.params_latex
        params = self._result._helper.params_names['all']
        return {p: comps_latex[p] + params_latex[p] for p in params}

    @property
    def params_labels(self) -> dict[str, str]:
        """Label of parameters."""
        comps_latex = self.comps_latex
        params_latex = self.params_latex
        params_unit = {
            k: f'\n[{v}]' if v else v for k, v in self.params_unit.items()
        }
        params = self._result._helper.params_names['all']
        return {
            p: comps_latex[p] + params_latex[p] + params_unit[p]
            for p in params
        }

    def set_xlabel(self, ax: Axes):
        ax.set_xlabel(r'$\mathrm{Energy\ [keV]}$')

    def plot_spec(
        self,
        data: bool = True,
        ne: bool = True,
        ene: bool = False,
        eene: bool = False,
        residuals: bool | Literal['rd', 'rp', 'rq'] = True,
        *,
        egrid: Mapping[str, NumPyArray] | None = None,
        params: Mapping[str, float | int | Array] | None = None,
        label_Fv: bool = False,
        label_vFv: bool = False,
    ) -> Figure:
        r"""Spectral plot.

        Parameters
        ----------
        data : bool, optional
            Whether to plot folded model and data. The default is ``True``.
        ne : bool, optional
            Whether to plot :math:`N(E)`. The default is ``True``.
        ene : bool, optional
            Whether to plot :math:`E N(E)`. The default is ``False``.
        eene : bool, optional
            Whether to plot :math:`E^2 N(E)`. The default is ``False``.
        residuals : bool or {'rd', 'rp', 'rq'}, optional
            Whether to plot residuals. Available options are:

                * ``True``: plot default residuals
                * ``False``: do not plot residuals
                * ``'rd'``: plot deviance residuals
                * ``'rp'``: plot Pearson residuals
                * ``'rq'``: plot quantile residuals

            The default is ``True``.
        egrid : dict, optional
            Overwrite the photon energy grid when plotting unfolded model.
        params : dict, optional
            Overwrite the photon energy grid when plotting unfolded model.
        label_Fv : bool, optional
            Whether to label the y-axis of :math:`E N(E)` plot as
            :math:`F_{\nu}`. The default is ``False``.
        label_vFv : bool, optional
            Whether to label the y-axis of :math:`E^2 N(E)` plot as
            :math:`\nu F_{\nu}`. The default is ``False``.

        Returns
        -------
        Figure
            The Figure object containing spectral plot.
        """
        nrows = data + ne + ene + eene + bool(residuals)
        height_ratios = [1.618] * nrows
        if residuals:
            height_ratios[-1] = 1.0

        config = self.config
        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=1,
            sharex='all',
            height_ratios=height_ratios,
            gridspec_kw={'bottom': 0.07, 'top': 0.97, 'hspace': 0.03},
            figsize=(8, 4 + nrows),
            squeeze=False,
        )
        axs = axs.ravel()
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

        self.set_xlabel(axs[-1])

        plots = []
        if data:
            plots.append('ce')
        if ne:
            plots.append('ne')
        if ene:
            plots.append('ene')
        if eene:
            plots.append('eene')

        residuals: Literal['rd', 'rp', 'rq'] | None
        if residuals:
            plots.append('residuals')
            if residuals is True:
                residuals = config.residuals
        else:
            residuals = None

        axs_dict = dict(zip(plots, axs))

        yscale = config.yscale

        if data:
            ax = axs_dict['ce']
            self.plot_ce(ax)
            self.plot_folded(ax)
            if yscale == 'linear':
                ax.set_yscale('linear')
            else:
                ax.set_yscale('log')
                dmin, dmax = ax.get_yaxis().get_data_interval()
                vmin = ax.get_ylim()[0]
                if yscale == 'linlog' and dmin <= 0.0:
                    lin_frac = config.lin_frac
                    if np.log10(dmax / vmin) > 7:
                        vmin = 1e-7 * dmax
                    scale = LinLogScale(
                        axis=None,
                        base=10.0,
                        lin_thresh=vmin,
                        lin_scale=get_scale(10.0, vmin, dmin, dmax, lin_frac),
                    )
                    ax.set_yscale(scale)
                    ax.axhline(vmin, c='k', lw=0.15, ls=':', zorder=-1)
                else:
                    _adjust_log_range(ax, 'y', 7)
        if ne:
            self.plot_unfolded(axs_dict['ne'], 'ne', params, egrid)
            if yscale != 'linear':
                axs_dict['ne'].set_yscale('log')
                _adjust_log_range(axs_dict['ne'], 'y')
        if ene:
            self.plot_unfolded(
                axs_dict['ene'], 'ene', params, egrid, label_Fv=label_Fv
            )
            if yscale != 'linear':
                axs_dict['ene'].set_yscale('log')
                _adjust_log_range(axs_dict['ene'], 'y')
        if eene:
            self.plot_unfolded(
                axs_dict['eene'], 'eene', params, egrid, label_vFv=label_vFv
            )
            if yscale != 'linear':
                axs_dict['eene'].set_yscale('log')
                _adjust_log_range(axs_dict['eene'], 'y')
        if residuals:
            self.plot_residuals(axs_dict['residuals'], residuals)

        axs[0].set_xscale(config.xscale)
        intervalx = np.array([ax.dataLim.intervalx for ax in axs])
        xmin = intervalx[:, 0].min()
        xmax = intervalx[:, 1].max()
        axs[0].set_xlim(xmin * 0.97, xmax * 1.06)

        for ax in axs:
            ax.relim()
            ax.autoscale_view()

        return fig

    def plot_unfolded(
        self,
        ax: Axes,
        mtype: Literal['ne', 'ene', 'eene'],
        params: Mapping[str, float | int | Array] | None = None,
        egrid: Mapping[str, NumPyArray] | None = None,
        label_Fv: bool = False,
        label_vFv: bool = False,
    ):
        r"""Plot unfolded model.

        Parameters
        ----------
        ax : Axes
            The Axes object to plot.
        mtype : {'ne', 'ene', 'eene'}
            The type of unfolded model, available options are:

                * ``'ne'``: plot :math:`N(E)`
                * ``'ene'``: plot :math:`E N(E)`
                * ``'eene'``: plot :math:`E^2 N(E)`

        params : dict, optional
            Overwrite the parameters when plotting unfolded model.
        egrid : dict, optional
            Overwrite the photon energy grid when plotting unfolded model.
        label_Fv : bool, optional
            Whether to label the y-axis of :math:`E N(E)` plot as
            :math:`F_{\nu}`. The default is ``False``.
        label_vFv : bool, optional
            Whether to label the y-axis of :math:`E^2 N(E)` plot as
            :math:`\nu F_{\nu}`. The default is ``False``.
        """
        params = dict(params) if params is not None else {}
        if params:
            if any(np.shape(v) != () for v in params.values()):
                raise ValueError('params must be scalars')
        egrid = dict(egrid) if egrid is not None else {}
        config = self.config
        colors = self.colors
        cl = config.cl
        comps = config.plot_comps
        step_kwargs = {'lw': 1.618, 'alpha': config.alpha}
        ribbon_kwargs = {'lw': 0.618, 'alpha': 0.2 * config.alpha}

        if mtype == 'ne':
            label_type = 'ne'
        elif mtype == 'ene':
            label_type = 'Fv' if label_Fv else 'ene'
        elif mtype == 'eene':
            label_type = 'vFv' if label_vFv else 'eene'
        else:
            raise ValueError("mtype must be 'ne', 'ene', or 'eene'")

        ax.set_ylabel(config._YLABLES[label_type])

        for name, data in self.data.items():
            color = colors[name]
            egrid_ = egrid.get(name, data.photon_egrid)
            ne, ci = data.unfolded_model(mtype, egrid_, params, False, cl)
            _plot_step(
                ax, egrid_[:-1], egrid_[1:], ne, color=color, **step_kwargs
            )
            if ci is not None:
                _plot_ribbon(
                    ax,
                    egrid_[:-1],
                    egrid_[1:],
                    ci,
                    color=color,
                    **ribbon_kwargs,
                )

            if comps:
                if not data.has_comps:
                    continue

                ne, ci = data.unfolded_model(mtype, egrid_, params, True)
                for ne_ in ne.values():
                    _plot_step(
                        ax,
                        egrid_[:-1],
                        egrid_[1:],
                        ne_,
                        color=color,
                        **(step_kwargs | {'ls': ':'}),
                    )
                if ci is not None:
                    for ci_ in ci.values():
                        _plot_ribbon(
                            ax,
                            egrid_[:-1],
                            egrid_[1:],
                            ci_,
                            color=color,
                            **ribbon_kwargs,
                        )

    def plot_folded(self, ax: Axes):
        """Plot folded model.

        Parameters
        ----------
        ax : Axes
            The Axes object to plot.
        """
        config = self.config
        colors = self.colors
        cl = config.cl
        step_kwargs = {'lw': 1.618, 'alpha': config.alpha}
        ribbon_kwargs = {'lw': 0.618, 'alpha': 0.2 * config.alpha}

        ax.set_ylabel(config._YLABLES['ce'])

        for name, data in self.data.items():
            color = colors[name]

            _plot_step(
                ax,
                data.channel_emin,
                data.channel_emax,
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
                    data.channel_emin,
                    data.channel_emax,
                    quantiles,
                    color=color,
                    **ribbon_kwargs,
                )

    def plot_ce(self, ax: Axes):
        """Plot data.

        Parameters
        ----------
        ax : Axes
            The Axes object to plot.
        """
        config = self.config
        colors = self.colors
        alpha = config.alpha
        xlog = config.xscale == 'log'

        ax.set_ylabel(config._YLABLES['ce'])

        for name, data in self.data.items():
            color = colors[name]
            marker = self._markers[name]
            ax.errorbar(
                x=data.channel_emean if xlog else data.channel_emid,
                xerr=data.channel_errors if xlog else 0.5 * data.channel_width,
                y=data.ce_data,
                yerr=data.ce_errors,
                alpha=alpha,
                color=color,
                fmt=f'{marker} ',
                label=name,
                lw=0.75,
                ms=2.4,
                mec=color,
                mfc='#FFFFFFCC',
            )

        if len(self.data) > 5:
            ncols = int(np.ceil(len(self.data) / 4))
        else:
            ncols = 1
        ax.legend(ncols=ncols)

    def plot_residuals(
        self,
        ax: Axes,
        rtype: Literal['rd', 'rp', 'rq'] | None = None,
    ):
        """Plot residuals.

        Parameters
        ----------
        ax : Axes
            The Axes object to plot.
        rtype : {'rd', 'rp', 'rq'}, optional
            The type of residuals, available options are:

                * ``'rd'``: deviance residuals
                * ``'rp'``: Pearson residuals
                * ``'rq'``: quantile residuals

        """
        if rtype not in {'rd', 'rp', 'rq', None}:
            raise ValueError(
                'residuals type must be "rd" (deviance), "rp" (pearson), '
                '"rq" (quantile), or None (use default residuals)'
            )

        config = self.config
        colors = self.colors
        cl = config.cl
        random_quantile = config.random_quantile
        with_sign = config.residuals_ci_with_sign
        mark_outlier = config.mark_outlier_residuals
        seed = config.seed
        ribbon_kwargs = {'lw': 0.618, 'alpha': 0.15 * config.alpha}

        if rtype is None:
            rtype = config.residuals

        alpha = config.alpha
        xlog = config.xscale == 'log'

        normal_q = stats.norm.isf(0.5 * (1.0 - cl))

        ax.set_ylabel(config._YLABLES[rtype])

        for name, data in self.data.items():
            color = colors[name]
            marker = self._markers[name]
            x = data.channel_emean if xlog else data.channel_emid
            xerr = data.channel_errors if xlog else 0.5 * data.channel_width

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
                    data.channel_emin,
                    data.channel_emax,
                    quantiles,
                    color=color,
                    **ribbon_kwargs,
                )
            else:
                for q in normal_q:
                    ax.fill_between(
                        [data.channel_emin[0], data.channel_emax[-1]],
                        -q,
                        q,
                        color=color,
                        **ribbon_kwargs,
                    )

            use_mle = True if quantiles else False
            r = data.residuals(rtype, seed, config.random_quantile, use_mle)
            if rtype == 'rq':
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
        rtype: Literal['rd', 'rp', 'rq'] | None = None,
        seed: int | None = None,
        detrend: bool = True,
    ) -> Figure:
        """Quantile-Quantile plot.

        Parameters
        ----------
        rtype : {'rd', 'rp', 'rq'}, optional
            The type of residuals, available options are:

                * ``'rd'``: deviance residuals
                * ``'rp'``: Pearson residuals
                * ``'rq'``: quantile residuals
                * ``None``: use the default residuals type

            The default is ``None``.
        seed : int, optional
            Random seed used in calculation. The default is ``None``.
        detrend : bool, optional
            Whether to detrend the Q-Q plot. The default is ``True``.

        Returns
        -------
        Figure
            The Figure object containing Q-Q plot.
        """
        config = self.config
        random_quantile = config.random_quantile
        if rtype is None:
            rtype = config.residuals

        rsim = {
            name: data.residuals_sim(rtype, seed, random_quantile)
            for name, data in self.data.items()
        }
        if any(i is None for i in rsim.values()):
            rsim['total'] = None
        else:
            rsim['total'] = np.hstack(list(rsim.values()))

        use_mle = True if rsim['total'] is not None else False
        r = {
            name: data.residuals(rtype, seed, random_quantile, use_mle)
            for name, data in self.data.items()
        }
        if rtype == 'rq':
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
            theor, q, line, lo, up = _get_qq(
                r[name], detrend, 0.95, rsim[name]
            )
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

        return fig

    def plot_pit(self, detrend=True) -> Figure:
        """Probability integral transformation empirical CDF plot.

        Parameters
        ----------
        detrend : bool, optional
            Whether to detrend the PIT ECDF plot. The default is ``True``.

        Returns
        -------
        Figure
            The Figure object containing PIT ECDF plot.
        """
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

        axs = [ax1] + axs.ravel().tolist()
        names = ['total'] + list(self.ndata.keys())
        colors = ['k'] + get_colors(n_subplots, config.palette)

        for ax, name, color in zip(axs, names, colors):
            x, y, line, lower, upper = _get_pit_ecdf(pit[name], 0.95, detrend)
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
            axs[-1].set_visible(False)

        return fig

    def plot_gof(self) -> Figure:
        """Plot distribution of GOF statistics and p-value.

        Returns
        -------
        Figure
            The Figure object containing GOF statistics plot.
        """
        if isinstance(self, MLEResultPlotter):
            if self._result._boot is None:
                raise RuntimeError(
                    'MLEResult.boot() must be called to assess gof'
                )
            n = int(self._result._boot.n_valid)
            dev_obs = self._result.deviance
            dev_sim = self._result._boot.deviance
            dev_sim = dev_sim['group'] | {'total': dev_sim['total']}
            p_value = self._result._boot.p_value
        elif isinstance(self, PosteriorResultPlotter):
            if self._result._ppc is None:
                raise RuntimeError(
                    'PosteriorResult.ppc() must be called to assess gof'
                )
            n = int(self._result._ppc.n_valid)
            dev_obs = self._result._mle['deviance']
            dev_sim = self._result._ppc.deviance
            dev_obs = dev_obs['group'] | {'total': dev_obs['total']}
            dev_sim = dev_sim['group'] | {'total': dev_sim['total']}
            p_value = self._result._ppc.p_value
        else:
            raise NotImplementedError
        p_value = p_value['group'] | {'total': p_value['total']}

        config = self.config
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
        ax1.set_xlabel('$D$')
        ax1.set_ylabel(r'$P(\mathcal{D} \geq D)$')

        axs = [ax1] + axs.ravel().tolist()
        names = ['total'] + list(self.ndata.keys())
        colors = ['k'] + get_colors(n_subplots, config.palette)

        for ax, name, color in zip(axs, names, colors):
            d_obs = dev_obs[name]
            d_sim = np.sort(dev_sim[name])
            sf = 1.0 - np.arange(1.0, n + 1.0) / n
            ax.plot(d_sim, sf, color=color)
            ax.axvline(d_obs, color=color, ls=':')
            p = p_value[name]
            if p > 0.0:
                pstr = f'{name} $p = {p:.2g}$'
            else:
                pstr = f'{name} $p < {1}/{n}$'
            ax.annotate(
                text=pstr,
                xy=(0.97, 0.97),
                xycoords='axes fraction',
                ha='right',
                va='top',
                color=color,
            )
            ax.set_yscale('log')
        if n_subplots % 2:
            axs[-1].set_visible(False)

        return fig


class MLEResultPlotter(Plotter):
    data: dict[str, MLEPlotData]
    _result: MLEResult
    _supported = (
        'data',
        'r',
        'rd',
        'rp',
        'rq',
        'ne',
        'ene',
        'eene',
        'Fv',
        'vFv',
        'corner',
        'gof',
        'qq',
        'pit',
    )

    def __call__(self, plots: str = 'data ne r') -> dict[str, Figure]:
        r"""Plot MLE fit results.

        Parameters
        ----------
        plots : str, optional
            Plots to show, available plots are:

                * ``'data'``: data and folded model plot
                * ``'ne'``: :math:`N(E)` model plot
                * ``'ene'``: :math:`E N(E)` model plot
                * ``'eene'``: :math:`E^2 N(E)` model plot
                * ``'Fv'``: :math:`F_{\nu}` model plot
                * ``'vFv'``: :math:`\nu F_{\nu}` model plot
                * ``'r'``: default residuals plot
                * ``'rd'``: deviance residuals plot
                * ``'rp'``: Pearson residuals plot
                * ``'rq'``: quantile residuals plot
                * ``'corner'``: corner plot
                * ``'gof'``: goodness-of-fit statistics plot
                * ``'qq'``: quantiles-quantiles plot of residuals
                * ``'pit'``: probability integral transform plot of spectral
                  data

            Multiple plots can be combined by separating them with whitespace.
            THe default is ``'data ne r'``.

        Returns
        -------
        dict
            Dictionary containing Figure object for each plot.
        """
        plots = re.split(r'\s+', str(plots))
        if any(p not in self._supported for p in plots):
            supported = ', '.join(self._supported)
            err = ', '.join(p for p in plots if p not in self._supported)
            raise ValueError(f'supported plots are: {supported}; got {err}')

        plots_set = set(plots)
        dic = {}

        spec = {
            'data',
            'ne',
            'ene',
            'eene',
            'Fv',
            'vFv',
            'r',
            'rd',
            'rp',
            'rq',
        }
        if spec & plots_set:
            residuals = [i for i in plots if i in ('r', 'rd', 'rp', 'rq')]
            if residuals:
                residuals = residuals[-1]
                if residuals == 'r':
                    residuals = True
            else:
                residuals = False
            dic['spec'] = self.plot_spec(
                data='data' in plots,
                ne='ne' in plots,
                ene=bool({'ene', 'Fv'} & plots_set),
                eene=bool({'eene', 'vFv'} & plots_set),
                residuals=residuals,
                label_Fv='Fv' in plots,
                label_vFv='vFv' in plots,
            )

        if 'corner' in plots_set:
            dic['corner'] = self.plot_corner()

        if 'gof' in plots_set:
            dic['gof'] = self.plot_gof()

        if 'qq' in plots_set:
            dic['qq'] = self.plot_qq()

        if 'pit' in plots_set:
            dic['pit'] = self.plot_pit()

        return dic

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

    def plot_corner(
        self,
        params: str | Sequence[str] | None = None,
        color: str | None = None,
        bins: int | Sequence[int] = 40,
        hist_bin_factor: float | Sequence[float] = 1.5,
        fig_path: str | None = None,
    ) -> Figure:
        """Corner plot of bootstrap parameters.

        Parameters
        ----------
        params : str or sequence of str, optional
            Parameters to plot. The default is all spectral parameters.
        color : str, optional
            Color of the plot. The default is ``None``.
        bins : int or list of int, optional
            The number of bins to use in histograms, either as a fixed value
            for all dimensions, or as a list of integers for each dimension.
            The default is 40.
        hist_bin_factor : float or list of float, optional
            This is a factor (or list of factors, one for each dimension)
            that will multiply the bin specifications when making the 1-D
            histograms. This is generally used to increase the number of
            bins in the 1-D plots to provide more resolution.
            The default is 1.5.
        fig_path : str, optional
            Path to save the figure. The default is ``None``.

        Returns
        -------
        Figure
            The Figure object containing corner plot.
        """
        if self._result._boot is None:
            raise ValueError('MLEResult.boot() must be called first')

        helper = self._result._helper
        params = check_params(params, helper)
        axes_scale = [
            'log' if helper.params_log[p] else 'linear' for p in params
        ]
        params_titles = self.params_titles
        params_labels = self.params_labels
        fig = plot_corner(
            idata=az.from_dict(self._result._boot.params),
            bins=bins,
            hist_bin_factor=hist_bin_factor,
            params=params,
            axes_scale=axes_scale,
            levels=self.config.cl,
            titles=[params_titles[p] for p in params],
            labels=[params_labels[p] for p in params],
            color=color,
        )
        if fig_path:
            fig.savefig(fig_path, bbox_inches='tight')
        return fig


class PosteriorResultPlotter(Plotter):
    data: dict[str, PosteriorPlotData]
    _result: PosteriorResult
    _supported = (
        'data',
        'r',
        'rd',
        'rp',
        'rq',
        'ne',
        'ene',
        'eene',
        'Fv',
        'vFv',
        'corner',
        'gof',
        'qq',
        'pit',
        'trace',
        'khat',
    )

    def __call__(self, plots: str = 'data ne r') -> dict[str, Figure]:
        r"""Plot Bayesian fit results.

        Parameters
        ----------
        plots : str, optional
            Plots to show, available plots are:

                * ``'data'``: data and folded model plot
                * ``'ne'``: :math:`N(E)` model plot
                * ``'ene'``: :math:`E N(E)` model plot
                * ``'eene'``: :math:`E^2 N(E)` model plot
                * ``'Fv'``: :math:`F_{\nu}` model plot
                * ``'vFv'``: :math:`\nu F_{\nu}` model plot
                * ``'r'``: default residuals plot
                * ``'rd'``: deviance residuals plot
                * ``'rp'``: Pearson residuals plot
                * ``'rq'``: PSIS-LOO quantile residuals plot
                * ``'corner'``: corner plot
                * ``'gof'``: goodness-of-fit statistics plot
                * ``'qq'``: quantiles-quantiles plot of residuals
                * ``'pit'``: PSIS-LOO probability integral transform plot of
                  spectral data
                * ``'trace'``: trace plot of posterior samples
                * ``'khat'``: k-hat plot for Bayesian PSIS-LOO diagnostics

            Multiple plots can be combined by separating them with whitespace.
            THe default is ``'data ne r'``.

        Returns
        -------
        dict
            Dictionary containing Figure object for each plot.
        """
        plots = re.split(r'\s+', str(plots))
        if any(p not in self._supported for p in plots):
            supported = ', '.join(self._supported)
            err = ', '.join(p for p in plots if p not in self._supported)
            raise ValueError(f'supported plots are: {supported}; got {err}')

        plots_set = set(plots)
        dic = {}

        spec = {
            'data',
            'ne',
            'ene',
            'eene',
            'Fv',
            'vFv',
            'r',
            'rd',
            'rp',
            'rq',
        }
        if spec & plots_set:
            residuals = [i for i in plots if i in ('r', 'rd', 'rp', 'rq')]
            if residuals:
                residuals = residuals[-1]
                if residuals == 'r':
                    residuals = True
            else:
                residuals = False
            dic['spec'] = self.plot_spec(
                data='data' in plots,
                ne='ne' in plots,
                ene=bool({'ene', 'Fv'} & plots_set),
                eene=bool({'eene', 'vFv'} & plots_set),
                residuals=residuals,
                label_Fv='Fv' in plots,
                label_vFv='vFv' in plots,
            )

        if 'corner' in plots_set:
            dic['corner'] = self.plot_corner()

        if 'gof' in plots_set:
            dic['gof'] = self.plot_gof()

        if 'qq' in plots_set:
            dic['qq'] = self.plot_qq()

        if 'pit' in plots_set:
            dic['pit'] = self.plot_pit()

        if 'trace' in plots_set:
            dic['trace'] = self.plot_trace()

        if 'khat' in plots_set:
            dic['khat'] = self.plot_khat()

        return dic

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
    ) -> Figure:
        """Plot trace plot of posterior samples.

        Parameters
        ----------
        params : str or sequence of str, optional
            Parameters to plot. The default is all spectral parameters.
        fig_path : str, optional
            Path to save the figure. The default is ``None``.
        """
        helper = self._result._helper
        params = check_params(params, helper)
        axes_scale = [
            'log' if helper.params_log[p] else 'linear' for p in params
        ]
        params_labels = self.params_labels
        labels = [params_labels[p] for p in params]
        fig = plot_trace(self._result._idata, params, axes_scale, labels)
        if fig_path:
            fig.savefig(fig_path, bbox_inches='tight')
        return fig

    def plot_corner(
        self,
        params: str | Sequence[str] | None = None,
        color: str | None = None,
        divergences: bool = True,
        bins: int | Sequence[int] = 40,
        hist_bin_factor: float | Sequence[float] = 1.5,
        fig_path: str | None = None,
    ) -> Figure:
        """Corner plot of posterior parameters.

        Parameters
        ----------
        params : str or sequence of str, optional
            Parameters to plot. The default is all spectral parameters.
        color : str, optional
            Color of the plot. The default is ``None``.
        divergences : bool, optional
            Whether to show divergent samples. The default is ``True``.
        bins : int or list of int, optional
            The number of bins to use in histograms, either as a fixed value
            for all dimensions, or as a list of integers for each dimension.
            The default is 40.
        hist_bin_factor : float or list of float, optional
            This is a factor (or list of factors, one for each dimension)
            that will multiply the bin specifications when making the 1-D
            histograms. This is generally used to increase the number of
            bins in the 1-D plots to provide more resolution.
            The default is 1.5.
        fig_path : str, optional
            Path to save the figure. The default is ``None``.

        Returns
        -------
        Figure
            The Figure object containing corner plot.
        """
        helper = self._result._helper
        params = check_params(params, helper)
        axes_scale = [
            'log' if helper.params_log[p] else 'linear' for p in params
        ]
        params_titles = self.params_titles
        params_labels = self.params_labels
        fig = plot_corner(
            idata=self._result._idata,
            bins=bins,
            hist_bin_factor=hist_bin_factor,
            params=params,
            axes_scale=axes_scale,
            levels=self.config.cl,
            titles=[params_titles[p] for p in params],
            labels=[params_labels[p] for p in params],
            color=color,
            divergences=divergences,
        )
        if fig_path:
            fig.savefig(fig_path, bbox_inches='tight')
        return fig

    def plot_khat(self) -> Figure:
        """Plot k-hat diagnostic of PSIS-LOO."""
        config = self.config
        colors = self.colors
        alpha = config.alpha
        xlog = config.xscale == 'log'

        fig, ax = plt.subplots(1, 1, squeeze=True, tight_layout=True)

        khat = self._result.loo.pareto_k
        if np.any(khat.values > 0.7):
            ax.axhline(0.7, color='r', lw=0.5, ls=':')

        for name, data in self.data.items():
            color = colors[name]
            marker = self._markers[name]
            khat_data = khat.sel(channel=data.channel).values
            x = data.channel_emean if xlog else data.channel_emid
            ax.errorbar(
                x=x,
                xerr=data.channel_errors if xlog else 0.5 * data.channel_width,
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

        ax.set_xscale(config.xscale)
        ax.set_xlabel('Energy [keV]')
        ax.set_ylabel(r'Shape Parameter $\hat{k}$')

        return fig
