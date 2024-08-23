"""Miscellaneous plotting."""

from __future__ import annotations

from collections.abc import Sequence

import arviz as az
import corner
import matplotlib.pyplot as plt
import numpy as np

from elisa.plot.util import (
    gaussian_kernel_smooth,
    get_colors,
    get_contour_colors,
)
from elisa.util.misc import report_interval


def plot_corner(
    idata: az.InferenceData,
    bins: int | Sequence[int] = 40,
    hist_bin_factor: float | Sequence[float] = 1.5,
    params: str | Sequence[str] | None = None,
    axes_scale: str | Sequence[str] = 'linear',
    levels: float | Sequence[float] | None = None,
    titles: str | Sequence[str] | None = None,
    labels: str | Sequence[str] | None = None,
    color: str = None,
    divergences: bool = True,
):
    """Plot posterior corner plot.

    Parameters
    ----------
    idata : az.InferenceData
        arviz.InferenceData object.
    bins : int or list of int, optional
        The number of bins to use in histograms, either as a fixed value for
        all dimensions, or as a list of integers for each dimension.
        The default is 40.
    hist_bin_factor : float or list of float, optional
        This is a factor (or list of factors, one for each dimension) that
        will multiply the bin specifications when making the 1-D histograms.
        This is generally used to increase the number of bins in the 1-D
        plots to provide more resolution. The default is 1.5.
    params : str or list of str, optional
        One or more parameters to be plotted.
    axes_scale : str, or list of str, optional
        Scale to use for each parameter dimension. If only one scale is given,
        use that for all dimensions. Scale must be ``'linear'`` or ``'log'``.
    levels : float, or list of float, optional
        Credible levels to plot. If None, use 0.683 and 0.95.
    titles : str, or list of str, optional
        Titles to be displayed in the diagonal.
    labels : str, or list of str, optional
        Labels to be displayed in axis label.
    color : str, optional
        Color to use for the plot.
    divergences : bool, optional
        Whether to mark diverging samples.

    Returns
    -------
    fig : plt.Figure
        The figure containing the corner plot.

    """
    posterior = idata['posterior']

    if params is None:
        params = list(posterior.data_vars.keys())
    elif isinstance(params, str):
        params = [params]
    else:
        params = list(params)

    posterior = posterior[params]

    all_params = posterior.data_vars.keys()
    not_found = set(params) - set(all_params)
    if not_found:
        raise ValueError(f'parameter {not_found} not found in posterior')

    if titles is None:
        titles = params
    elif isinstance(titles, str):
        titles = [titles]
    else:
        titles = list(titles)
    assert len(titles) == len(params)

    if labels is None:
        labels = params
    elif isinstance(labels, str):
        labels = [labels]
    else:
        labels = list(labels)
    assert len(labels) == len(params)

    median = posterior.median()
    quantile = posterior.quantile([0.15865, 0.84135])
    median = {p: float(median[p].values) for p in params}
    quantile = {p: quantile[p].values.tolist() for p in params}
    titles = [
        f'{t} = {report_interval(median[p], *quantile[p])}'
        for t, p in zip(titles, params)
    ]

    if levels is None:
        levels = [
            [0.683, 0.954, 0.997],  # 1/2/3-sigma of 1d
            [0.393, 0.865, 0.989],  # 1/2/3-sigma of 2d
            [0.683, 0.9],  # 1-sigma of 1d, and 90% of 2d
            [0.393, 0.683, 0.9],  # 1-sigma, 68.3% and 90% of 2d
            [0.683, 0.95],  # 68.3% and 95% of 2d
        ][-1]
    else:
        levels = list(levels)

    # def to_hex(c):
    #     rgb_hex = ''.join(f'{round(i * 255):02x}' for i in c[:3])
    #     return f'#{rgb_hex}'
    # cmap = plt.get_cmap('Blues')
    # colors2 = [cmap(i*0.8 + 0.1) for i in levels]
    # colors1 = [scale_color(to_hex(c), 0.95) for c in colors2]

    if color is None:
        color = '#205295'
    else:
        color = str(color)

    plt.rcParams['axes.formatter.min_exponent'] = 3
    c1, c2 = get_contour_colors(color, len(levels), 0.8, 2.0)

    vmin = {p: posterior[p].values.min() for p in params}
    vmax = {p: posterior[p].values.max() for p in params}
    if any(vmin[p] == vmax[p] for p in params):
        plot_range = [
            (vmin[p], vmax[p]) if vmin[p] != vmax[p] else 0.99 for p in params
        ]
    else:
        plot_range = None

    fig = corner.corner(
        idata,
        bins=bins,
        range=plot_range,
        axes_scale=axes_scale,
        color=color,
        hist_bin_factor=hist_bin_factor,
        titles=titles,
        labels=labels,
        show_titles=True,
        title_fmt=None,
        quantiles=[0.15865, 0.5, 0.84135],
        use_math_text=True,
        labelpad=-0.08,
        divergences=divergences,
        divergences_kwargs={'color': 'red', 'alpha': 0.3, 'ms': 1},
        var_names=params,
        # kwargs for corner.hist2d
        levels=levels,
        plot_datapoints=True,
        plot_density=False,
        plot_contours=True,
        fill_contours=True,
        no_fill_contours=True,
        contour_kwargs={'colors': c1},
        contourf_kwargs={'colors': ['white'] + c2, 'alpha': 0.75},
        data_kwargs={'color': c2[0], 'alpha': 0.75, 'ms': 1.5},
    )

    return fig


def plot_trace(
    idata: az.InferenceData,
    params: str | Sequence[str] | None = None,
    axes_scale: str | Sequence[str] | None = None,
    labels: str | Sequence[str] | None = None,
) -> plt.Figure:
    """Plot posterior sampling trace.

    Parameters
    ----------
    idata : az.InferenceData
        arviz.InferenceData object.
    params : str, or list of str, optional
        One or more parameters to be plotted.
    axes_scale : str, or list of str, optional
        Scale to use for each parameter dimension. If only one scale is given,
        use that for all dimensions. Scale must be ``'linear'`` or ``'log'``.
    labels : str, or list of str, optional
        Labels to be displayed in y-axis label.

    Returns
    -------
    fig : plt.Figure
        The figure containing the trace plot.
    """
    colors = get_colors(len(idata['posterior']['chain']), palette='bright')
    posterior = idata['posterior']
    deviance = -2.0 * idata['log_likelihood']['total'].values

    if params is None:
        params = list(posterior.data_vars.keys())
    elif isinstance(params, str):
        params = [params]
    else:
        params = list(params)

    all_params = posterior.data_vars.keys()
    not_found = set(params) - set(all_params)
    if not_found:
        raise ValueError(f'parameter {not_found} not found in posterior')

    nparam = len(params)

    if 'sample_stats' in idata:
        idx = idata['sample_stats']['diverging'].values.nonzero()
        diverging_index = idx[1]
        diverging_sample = [
            posterior[p].values[idx[0], idx[1]] for p in params
        ]
    else:
        diverging_index = None
        diverging_sample = None

    if axes_scale is None:
        axes_scale = ['linear'] * nparam
    else:
        if isinstance(axes_scale, str):
            axes_scale = [axes_scale] * nparam
        elif len(axes_scale) != nparam:
            raise ValueError('`axes_scale` must match `params`')

    if labels is None:
        labels = params
    else:
        if isinstance(labels, str):
            labels = [labels]
        else:
            labels = list(labels)

    if len(labels) != nparam:
        raise ValueError('`tex` must match `params`')

    plt.rcParams['axes.formatter.min_exponent'] = 3
    fig, axes = plt.subplots(
        nrows=len(params),
        ncols=3,
        sharey='row',
        squeeze=False,
        gridspec_kw={'width_ratios': [3, 1, 1]},
        figsize=(9, nparam * 2),
        tight_layout=True,
    )
    fig.subplots_adjust(wspace=0.05, hspace=0)
    fig.align_ylabels(axes)

    axes[0, 0].set_title('trace')
    axes[0, 1].set_title('posterior')
    axes[0, 2].set_title('deviance')

    chain = posterior.chain.values
    draw = posterior.draw.values
    ndraw = posterior.draw.size
    bw = max(draw.size // 100, 10)

    for i in range(nparam):
        posterior_i = posterior[params[i]]
        if axes_scale[i] == 'log' and np.all(posterior_i.values > 0):
            scale = 'log'
            log_scale = True
        else:
            scale = 'linear'
            log_scale = False
        axes[i, 0].set_ylabel(labels[i])
        axes[i, 0].set_yscale(scale)
        for c in chain:
            sample = posterior_i[c].values
            loglike = deviance[c]
            draw_slice = draw[:: bw // 2]
            y = np.log(sample) if log_scale else sample
            smoothed = gaussian_kernel_smooth(draw, y, bw, draw_slice)
            x, kde = az.kde(y)
            if log_scale:
                smoothed = np.exp(smoothed)
                x = np.exp(x)

            color = colors[c]
            zorder = 10 - c

            axes[i, 0].step(
                draw + c / chain.size,
                sample,
                c=color,
                alpha=0.4,
                lw=0.15,
                zorder=zorder,
            )
            axes[i, 0].plot(
                draw_slice, smoothed, c=color, alpha=0.6, lw=1.5, zorder=zorder
            )
            axes[i, 1].plot(kde, x, c=color, alpha=0.6, lw=1.5, zorder=zorder)
            axes[i, 2].scatter(
                loglike, sample, color=color, s=0.05, alpha=0.05, zorder=zorder
            )

        axes[i, 0].set_xlim(0.5, ndraw - 0.5)
        # axes[i, 0].set_xticks([])
        axes[i, 1].set_xticks([])

    if diverging_index is not None:
        for i in range(nparam):
            ylim = axes[i, 0].get_ylim()
            y_lower = ylim[0]

            span = 0.1

            if axes_scale[i] == 'linear':
                y_upper = ylim[0] + span * np.diff(ylim)
            else:
                y_upper = ylim[0] * np.exp(span * np.diff(np.log(ylim)))

            axes[i, 0].vlines(diverging_index, y_lower, y_upper, color='k')
            axes[i, 0].set_ylim(ylim)

            xlim = axes[i, 1].get_xlim()
            x_lower = xlim[0]
            x_upper = xlim[0] + span * np.diff(xlim)

            axes[i, 1].hlines(diverging_sample[i], x_lower, x_upper, color='k')
            axes[i, 1].set_xlim(xlim)

    return fig
