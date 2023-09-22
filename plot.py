import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import xarray as xr
from corner import corner
from scipy.interpolate import splev, splrep
from scipy.spatial import ConvexHull


def get_profile(pars_lnL):
    pars, lnL = pars_lnL.T
    argsort = pars.argsort()
    pars = pars[argsort]
    lnL = lnL[argsort]
    p_best = pars[lnL.argmax()]
    lnL_max = lnL.max()
    lnL_pmin = lnL[pars.argmin()]
    lnL_pmax = lnL[pars.argmax()]
    x = []
    y = []
    for i, j in zip(pars, lnL):
        if i <= p_best:
            if lnL_pmin <= j <= lnL_max:
                x.append(i)
                y.append(j)
        else:
            if lnL_pmax <= j <= lnL_max:
                x.append(i)
                y.append(j)
    spl = splrep(x[1:-1], y[1:-1], k=2)
    newx = np.linspace(pars.min(), pars.max(), 100)
    newy = splev(newx, spl)
    return newx, newy


def plot_corner(
    idata, var_names=None, profile=True, level_idx=3, smooth=0.0,
    fig_path=None, **kwargs
):
    CL = [
        [0.683, 0.954, 0.997],  # 1/2/3-sigma for 1d normal
        [0.393, 0.865, 0.989],  # 1/2/3-sigma for 2d normal
        [0.683, 0.9],  # 68.3% and 90%
        [0.393, 0.683, 0.9]  # 1-sigma for 2d, 68.3% and 90%
    ]

    fig = corner(
        data=idata['posterior'],
        var_names=var_names,
        bins=kwargs.pop('bins', 40),
        quantiles=kwargs.pop('quantiles', [0.15865, 0.5, 0.84135]),
        levels=CL[level_idx],
        show_titles=True,
        title_fmt='.2f',
        color='#0343DF',
        max_n_ticks=5,
        smooth1d=0.0,
        smooth=smooth,
        fill_contours=False,
        no_fill_contours=True,
        plot_datapoints=True,
        plot_density=True,
        data_kwargs={'alpha': 1.0},
        pcolor_kwargs={'alpha': 0.95,
                       'edgecolors': 'face',
                       'linewidth': 0,
                       'rasterized': True},
        labelpad=-0.08,
        use_math_text=True,
        # label_kwargs={'fontsize': 14},
        # title_kwargs={'fontsize': 14}
        **kwargs
    )

    if profile and hasattr(idata, 'log_likelihood'):
        lnL = idata['log_likelihood']['total'].values.reshape(-1)
        lnL_max = lnL.max()
        pars = [idata['posterior'][p].values.reshape(-1) for p in var_names]
        pars = np.array(pars)
        for i in range(len(var_names)):
            x_best = pars[i, lnL.argmax()]
            y_best = lnL.max()
            pars_lnL = np.column_stack((pars[i], lnL))
            idx = ConvexHull(pars_lnL).vertices
            twinx = fig.axes[i * (len(var_names) + 1)].twinx()
            x, y = get_profile(pars_lnL[idx])
            twinx.plot(x, -y, c='tab:red')
            twinx.scatter(x_best, -y_best, c='tab:red', marker='o', s=15)
            # mask1 = x < x_best
            # idx1 = np.abs(lnL_max - y[mask1] - 0.5).argmin()
            # twinx.scatter(x[mask1][idx1], -y[mask1][idx1], c='tab:red', marker='s', s=15)
            # mask2 = x > x_best
            # idx2 = np.abs(lnL_max - y[mask2] - 0.5).argmin()
            # twinx.scatter(x[mask2][idx2], -y[mask2][idx2], c='tab:red', marker='s', s=15)
            twinx.yaxis.set_visible(False)
            twinx.axhline(-lnL_max + 0.5, c='tab:red', ls=':')

    if fig_path is not None:
        fig.savefig(fig_path)


def plot_ppc(
    x, y, y_hat, stat_best, dof, y_rep, stat, stat_rep, stat_min, stat_name,
    cl=0.9, xlabel='', colors=None
):
    """

    Parameters
    ----------
    x : list of (n, 2)
    y : list of (n)
    y_hat : list of (m, n)
    stat_best : float
    dof : int
    y_rep : list of (m, n)
    stat : (m,)
    stat_rep : (m,)
    stat_min : (m,)
    stat_name : str
    cl : float
    xlabel : str
    colors : list

    Returns
    -------

    """
    plt.style.use(['nature', 'science', 'no-latex'])

    fig, axes = plt.subplots(1, 2, figsize=(4, 2.2), dpi=150)
    fig.subplots_adjust(
        left=0.125, bottom=0.11, right=0.875, top=0.912, wspace=0.05
    )
    axes[0].set_box_aspect(1)
    axes[1].set_box_aspect(1)

    _min = min(stat.min(), stat_rep.min()) * 0.9
    _max = max(stat.max(), stat_rep.max()) * 1.1
    axes[0].plot([_min, _max], [_min, _max], ls=':', color='gray')
    axes[0].set_xlim(_min, _max)
    axes[0].set_ylim(_min, _max)
    ppp1 = np.sum(stat_rep > stat) / stat.size
    axes[0].set_title(f'$p$-value$=${ppp1:.3f}')
    axes[0].scatter(stat, stat_rep, s=1, marker='.', alpha=0.5)
    axes[0].set_aspect('equal')
    axes[0].set_xlabel(stat_name)
    axes[0].set_ylabel(stat_name + r'$^{\rm rep}$')

    ppp2 = (stat_min > stat_best).sum() / stat_min.size
    axes[1].set_title(f'$p$-value$=${ppp2:.3f}')
    # axes[1].hist(stat_min, bins='auto', density=True, alpha=0.4)
    grid, pdf = az.kde(stat_min)
    axes[1].fill_between(grid, pdf, alpha=0.4)
    axes[1].plot(grid, pdf)
    axes[1].set_xlim(grid.min(), grid.max())
    axes[1].set_ylim(bottom=0.0)
    axes[1].axvline(dof, ls=':', color='gray')
    axes[1].axvline(stat_best, c='r', ls='--')
    axes[1].set_xlabel(stat_name + r'$_{\rm min}$')
    # axes[1].set_ylabel('$N$ simulation')
    axes[1].set_ylabel('PDF')
    axes[1].yaxis.set_label_position('right')
    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_ticks_position('both')

    fig, axes = plt.subplots(2, 1, sharex=True, dpi=150)
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.align_ylabels(axes)

    if cl >= 1.0:
        cl = 1.0 - stats.norm.sf(cl) * 2.0

    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for xi, yi, yhi, yri, color in zip(x, y, y_hat, y_rep, colors):
        q_data = yi.cumsum()
        q_rep = yri.cumsum(axis=1)
        yhi_cumsum = yhi.cumsum(axis=1)

        total = yhi_cumsum[:, -1:]

        q_rep /= total#q_data[-1]#q_rep[:, -1:]
        q_data = q_data[None,:]/total#q_data[-1]
        yhi_cumsum /= total

        q_rep_hdi = az.hdi(np.expand_dims(q_rep, axis=0), hdi_prob=cl).T
        q_yhi_hdi = az.hdi(np.expand_dims(yhi_cumsum, axis=0), hdi_prob=cl).T

        data_resd = az.hdi(np.expand_dims(q_data - yhi_cumsum, axis=0), hdi_prob=cl).T
        rep_resd = az.hdi(np.expand_dims(q_rep - yhi_cumsum, axis=0), hdi_prob=cl).T

        left = xi[:, 0]
        right = xi[:, 1]
        mask = left[1:] != right[:-1]
        idx = [0, *(np.flatnonzero(mask) + 1), len(xi)]
        for j in range(len(idx) - 1):
            slice_j = slice(idx[j], idx[j + 1])
            ebins = np.append(left[slice_j], right[slice_j][-1])
            axes[0].step(
                ebins, np.append(q_data[slice_j], q_data[slice_j][-1]),
                lw=0.6, where='post', color=color
            )
            axes[0].fill_between(
                ebins,
                np.append(q_rep_hdi[0][slice_j], q_rep_hdi[0][slice_j][-1]),
                np.append(q_rep_hdi[1][slice_j], q_rep_hdi[1][slice_j][-1]),
                lw=0, step='post', alpha=0.4, color='gray'
            )
            axes[0].fill_between(
                ebins,
                np.append(q_yhi_hdi[0][slice_j], q_yhi_hdi[0][slice_j][-1]),
                np.append(q_yhi_hdi[1][slice_j], q_yhi_hdi[1][slice_j][-1]),
                lw=0, step='post', alpha=0.4, color=color
            )

            axes[1].fill_between(
                ebins,
                np.append(rep_resd[0][slice_j], rep_resd[0][slice_j][-1]),
                np.append(rep_resd[1][slice_j], rep_resd[1][slice_j][-1]),
                lw=0, step='post', alpha=0.4, color='gray'
            )
            axes[1].fill_between(
                ebins,
                np.append(data_resd[0][slice_j], data_resd[0][slice_j][-1]),
                np.append(data_resd[1][slice_j], data_resd[1][slice_j][-1]),
                step='post', alpha=0.4, color=color
            )

    axes[1].axhline(0, ls=':', c='gray', zorder=0)
    axes[0].set_xscale('log')
    axes[0].set_ylabel('Integrated Counts')

    plt.style.use('default')

    return ppp1, ppp2


def _split_step_plot(x_left, x_right, y):
    mask = x_left[1:] != x_right[:-1]
    idx = [0, *(np.flatnonzero(mask) + 1), len(y)]
    step_pairs = []
    for i in range(len(idx) - 1):
        slice_i = slice(idx[i], idx[i + 1])
        x_slice_i = np.append(x_left[slice_i], x_right[slice_i][-1])
        y_slice_i = y[slice_i]
        y_slice_i = np.append(y_slice_i, y_slice_i[-1])
        step_pairs.append((x_slice_i, y_slice_i))

    return step_pairs

def _quantile(data, cl, hdi=True, axis=None):
    if hdi:
        ndim = len(np.shape(data))
        if ndim == 1:
            data = data[None, None, :]
        elif ndim == 2:
            data = data[None, ...]
        elif ndim == 3:
            pass
        else:
            raise ValueError(f'`data` has shape={np.shape(data)}')
        return az.hdi(data, cl).T
    else:
        return np.quantile(data, [0.5 - cl / 2.0, 0.5 + cl / 2.0], axis=axis)


def _plot_data(
    ax, data, model, pars, marker, color, alpha, comps,
    pars_dist=None, pars_fit=None, data_sim=None,
    deviance_obs=None, deviance_dist=None,
    hdi=True, plot_uplims=False
):
    color = color + hex(round(255 * alpha))[2:]

    emid = data.ch_emid_geom
    eerr = np.abs(data.ch_error)
    net_spec = data.net_spec
    net_err = data.net_error#_gehrels

    if plot_uplims:
        uplims = net_spec < 0.0
    else:
        uplims = np.full(len(net_spec), False)
    upper = np.zeros_like(net_spec)
    upper[uplims] = net_err[uplims] * 1.645

    ax.errorbar(
        emid, net_spec + upper, net_err, eerr,
        fmt=f'{marker} ', c=color, label=data.name, ms=2.5, mfc='#FFFFFFCC',
        uplims=uplims, mec=color, capsize=1
    )

    other = [data.ph_ebins, data.ch_emin, data.ch_emax, data.resp_matrix]
    CE = model.CE(pars, *other)
    for i in _split_step_plot(data.ch_emin, data.ch_emax, CE):
        ax.step(*i, where='post', c=color, lw=1.3)

    if comps:
        CE_comps = model.CE(pars, *other, comps=True).values()
        if len(CE_comps) > 1:
            for comp in CE_comps:
                for i in _split_step_plot(data.ch_emin, data.ch_emax, comp):
                    ax.step(*i, where='post', c=color, ls=':')

    if pars_dist is not None:
        sim = model.CE(pars_dist, *other)
        # sim_comps = model.CE(sim_pars, *other, comps=True).values()
        for cl in [0.683, 0.95]:
            ci = _quantile(sim, cl, hdi, axis=0)
            ci_lower = _split_step_plot(data.ch_emin, data.ch_emax, ci[0])
            ci_upper = _split_step_plot(data.ch_emin, data.ch_emax, ci[1])
            for i, j in zip(ci_lower, ci_upper):
                ax.fill_between(*i, j[1], lw=0, step='post', color=color, alpha=0.15)

def _plot_ldata(
    ax, data, model, pars, marker, color, alpha, comps,
    pars_dist=None, pars_fit=None, data_sim=None,
    deviance_obs=None, deviance_dist=None,
    hdi=True
):
    _plot_data(ax, data, model, pars, marker, color, alpha, comps, pars_dist, hdi=True, plot_uplims=True)

    if ax.get_yscale() != 'log':
        ax.set_yscale('log')

def _plot_ufspec(
    ax, data, model, pars, marker, color, alpha, comps,
    pars_dist=None, pars_fit=None, data_sim=None,
    deviance_obs=None, deviance_dist=None,
    hdi=True
):
    color = color + hex(round(255 * alpha))[2:]

    ebins = np.geomspace(data.ch_emin.min(), data.ch_emax.max(), 2001)

    NE = model.NE(pars, ebins)
    ax.step(
        ebins, np.append(NE, NE[-1]), where='post', c=color, lw=1.3
    )

    if comps:
        NE_comps = model.NE(pars, ebins, comps=True).values()
        if len(NE_comps) > 1:
            for i in NE_comps:
                ax.step(ebins, np.append(i, i[-1]), where='post', c=color, ls=':')

    if pars_dist is not None:
        sim = model.NE(pars_dist, ebins)
        # sim_comps = model.NE(sim_pars, ebins, comps=True).values()
        for cl in [0.683, 0.95]:
            ci = _quantile(sim, cl, hdi, axis=0)
            ci = np.column_stack([ci, ci[:, -1]])
            ax.fill_between(ebins, *ci, lw=0, step='post', color=color, alpha=0.25)

            # if len(sim_comps) > 1 and cl == 0.683:
            #     for i in sim_comps:
            #         ci = np.quantile(i,
            #                          q=[0.5 - cl / 2.0, 0.5 + cl / 2.0],
            #                          axis=0)
            #         ci = np.column_stack([ci, ci[:, -1]])
            #         ax.step(ebins, ci[0], where='post', c=color, ls='-')
            #         ax.step(ebins, ci[1], where='post', c=color, ls='-')

    other = [data.ph_ebins, data.ch_emin, data.ch_emax, data.resp_matrix]
    CE = model.CE(pars, *other)

    ch_emin = data.ch_emin
    ch_emax = data.ch_emax
    flux = np.empty(len(data.ch_emid_geom))
    for i in range(len(flux)):
        flux[i] = model.flux(pars, (ch_emin[i], ch_emax[i]), 100, energy=False)
    factor = flux / data.ch_width / CE
    uf = data.net_spec * factor
    uf_err = factor * data.net_error#_gehrels

    uplims = uf < 0.0
    upper = np.zeros_like(uf)
    upper[uplims] = uf_err[uplims] * 1.645

    xlim = [0.95 * data.ch_emin.min(), 1.05 * data.ch_emax.max()]

    ylower = uf[~uplims] - uf_err[~uplims]
    ylim = [
        0.8 * np.min(ylower[ylower > 0.0]),
        1.2 * np.max(uf[~uplims] + uf[~uplims]),
    ]

    emid = data.ch_emid_geom
    eerr = np.abs(data.ch_error)
    ax.errorbar(
        emid, uf + upper, uf_err, eerr,
        fmt=f'{marker} ', c=color, label=data.name, ms=2.5, mfc='#FFFFFFCC',
        uplims=uplims, mec=color, capsize=1
    )

    if ax.get_yscale() != 'log':
        ax.set_yscale('log')

    return xlim, ylim


def _plot_eufspec(
    ax, data, model, pars, marker, color, alpha, comps,
    pars_dist=None, pars_fit=None, data_sim=None,
    deviance_obs=None, deviance_dist=None,
    hdi=True
):
    color = color + hex(round(255 * alpha))[2:]

    ebins = np.geomspace(data.ch_emin.min(), data.ch_emax.max(), 2001)

    ENE = model.ENE(pars, ebins)
    ax.step(ebins, np.append(ENE, ENE[-1]), where='post', color=color, lw=1.3)

    if comps:
        ENE_comps = model.ENE(pars, ebins, comps=True).values()
        if len(ENE_comps) > 1:
            for i in ENE_comps:
                ax.step(ebins, np.append(i, i[-1]), where='post', c=color, ls=':')

    if pars_dist is not None:
        sim = model.ENE(pars_dist, ebins)
        # sim_comps = model.ENE(sim_pars, ebins, comps=True).values()
        for cl in [0.683, 0.95]:
            ci = _quantile(sim, cl, hdi, axis=0)
            ci = np.column_stack([ci, ci[:, -1]])
            ax.fill_between(ebins, *ci, lw=0, step='post', color=color, alpha=0.25)

            # if len(sim_comps) > 1 and cl == 0.683:
            #     for i in sim_comps:
            #         ci = np.quantile(i,
            #                          q=[0.5 - cl / 2.0, 0.5 + cl / 2.0],
            #                          axis=0)
            #         ci = np.column_stack([ci, ci[:, -1]])
            #         ax.step(ebins, ci[0], where='post', c=color, ls='-')
            #         ax.step(ebins, ci[1], where='post', c=color, ls='-')

    other = [data.ph_ebins, data.ch_emin, data.ch_emax, data.resp_matrix]
    CE = model.CE(pars, *other)

    ch_emin = data.ch_emin
    ch_emax = data.ch_emax
    eflux = np.empty(len(data.ch_emid_geom))
    for i in range(len(eflux)):
        eflux[i] = model.flux(pars, (ch_emin[i], ch_emax[i]), 100)
    factor = eflux / data.ch_width / CE
    euf = data.net_spec * factor
    euf_err = factor * data.net_error#_gehrels

    uplims = euf < 0.0
    upper = np.zeros_like(euf)
    upper[uplims] = euf_err[uplims] * 1.645

    xlim = [0.95 * data.ch_emin.min(), 1.05 * data.ch_emax.max()]

    ylower = euf[~uplims] - euf_err[~uplims]
    ylim = [
        0.8 * np.min(ylower[ylower > 0.0]),
        1.2 * np.max(euf[~uplims] + euf[~uplims]),
    ]

    emid = data.ch_emid_geom
    eerr = np.abs(data.ch_error)
    ax.errorbar(
        emid, euf + upper, euf_err, eerr,
        fmt=f'{marker} ', c=color, label=data.name, ms=2.5, mfc='#FFFFFFCC',
        uplims=uplims, mec=color, capsize=1
    )

    if ax.get_yscale() != 'log':
        ax.set_yscale('log')

    return xlim, ylim


def _plot_eeufspec(
    ax, data, model, pars, marker, color, alpha, comps,
    pars_dist=None, pars_fit=None, data_sim=None,
    deviance_obs=None, deviance_dist=None,
    hdi=True
):
    color = color + hex(round(255 * alpha))[2:]

    ebins = np.geomspace(data.ch_emin.min(), data.ch_emax.max(), 2001)

    EENE = model.EENE(pars, ebins)
    ax.step(ebins, np.append(EENE, EENE[-1]), where='post')

    if comps:
        EENE_comps = model.EENE(pars, ebins, comps=True).values()
        if len(EENE_comps) > 1:
            for i in EENE_comps:
                ax.step(ebins, np.append(i, i[-1]), where='post', c=color, ls='--', lw=0.5)

    if pars_dist is not None:
        sim = model.EENE(pars_dist, ebins)
        # sim_comps = model.EENE(sim_pars, ebins, comps=True).values()
        for cl in [0.683, 0.95]:
            ci = _quantile(sim, cl, hdi, axis=0)
            ci = np.column_stack([ci, ci[:, -1]])
            ax.fill_between(ebins, *ci, lw=0, step='post', color=color, alpha=0.25)

            # if len(sim_comps) > 1 and cl == 0.683:
            #     for i in sim_comps:
            #         ci = np.quantile(i,
            #                          q=[0.5 - cl / 2.0, 0.5 + cl / 2.0],
            #                          axis=0)
            #         ci = np.column_stack([ci, ci[:, -1]])
            #         ax.step(ebins, ci[0], where='post', c=color, ls='-')
            #         ax.step(ebins, ci[1], where='post', c=color, ls='-')

    other = [data.ph_ebins, data.ch_emin, data.ch_emax, data.resp_matrix]
    CE = model.CE(pars, *other)

    ch_emin = data.ch_emin
    ch_emax = data.ch_emax
    eflux = np.empty(len(data.ch_emid_geom))

    for i in range(len(eflux)):
        eflux[i] = model.flux(pars, (ch_emin[i], ch_emax[i]), 100)
    factor = eflux / data.ch_width * data.ch_emid_geom / CE
    eeuf = data.net_spec * factor
    eeuf_err = factor * data.net_error#_gehrels

    uplims = eeuf < 0.0
    upper = np.zeros_like(eeuf)
    upper[uplims] = eeuf_err[uplims] * 1.645

    xlim = [0.95 * data.ch_emin.min(), 1.05 * data.ch_emax.max()]

    ylower = eeuf[~uplims] - eeuf_err[~uplims]
    ylim = [
        0.8 * np.min(ylower[ylower > 0.0]),
        1.2 * np.max(eeuf[~uplims] + eeuf[~uplims]),
    ]

    emid = data.ch_emid_geom
    eerr = np.abs(data.ch_error)

    ax.errorbar(
        emid, eeuf + upper, eeuf_err, eerr,
        fmt=f'{marker} ', c=color, label=data.name, ms=2.5, mfc='#FFFFFFCC',
        uplims=uplims, mec=color, capsize=1
    )

    if ax.get_yscale() != 'log':
        ax.set_yscale('log')

    return xlim, ylim


def _plot_icounts(
    ax, data, model, pars, marker, color, alpha, comps,
    pars_dist=None, pars_fit=None, data_sim=None,
    deviance_obs=None, deviance_dist=None,
    hdi=True
):
    color = color + hex(round(255 * alpha))[2:]

    counts = data.net_counts
    icounts = counts.cumsum()

    other = [data.ph_ebins, data.ch_emin, data.ch_emax, data.resp_matrix]
    model_counts = model.counts(pars, *other, data.spec_exposure)
    model_icounts = model_counts.cumsum()

    total_counts = model_icounts[-1]
    icounts /= total_counts
    model_icounts /= total_counts

    emid = data.ch_emid_geom
    eerr = np.abs(data.ch_error)
    ax.errorbar(
        emid, icounts, xerr=eerr,
        fmt=f'{marker} ', c=color, mec=color, ms=2.5, mfc='#FFFFFFCC',
        capsize=1
    )

    for i in _split_step_plot(data.ch_emin, data.ch_emax, model_icounts):
        ax.step(*i, where='post', c=color, lw=1.3)

    if data_sim is not None:
        # sim_model = model.counts(sim_pars, *other, data.spec_exposure)
        # total_counts = sim_model.sum(axis=1)[:, None]
        sim_icounts = data_sim.cumsum(axis=1) / total_counts
        for cl in [0.683, 0.95]:
            ci = _quantile(sim_icounts, cl, hdi, axis=0)
            ci_lower = _split_step_plot(data.ch_emin, data.ch_emax, ci[0])
            ci_upper = _split_step_plot(data.ch_emin, data.ch_emax, ci[1])
            for i, j in zip(ci_lower, ci_upper):
                ax.fill_between(*i, j[1], lw=0, step='post', color=color, alpha=0.1)

            if cl == 0.95:
                mask = (icounts < ci[0]) | (icounts > ci[1])
                ax.scatter(emid[mask], icounts[mask], marker='x', c='r', zorder=10)


def _plot_icounts_residual(
    ax, data, model, pars, marker, color, alpha, comps,
    pars_dist=None, pars_fit=None, data_sim=None,
    deviance_obs=None, deviance_dist=None,
    hdi=True
):
    color = color + hex(round(255 * alpha))[2:]

    counts = data.net_counts
    other = [data.ph_ebins, data.ch_emin, data.ch_emax, data.resp_matrix]
    model_counts = model.counts(pars, *other, data.spec_exposure)

    total_counts = model_counts.sum()
    icounts = counts.cumsum() / total_counts
    model_icounts = model_counts.cumsum() / total_counts

    emid = data.ch_emid_geom
    eerr = np.abs(data.ch_error)
    idiff = icounts - model_icounts
    ax.errorbar(
        emid, idiff, xerr=eerr,
        fmt=f'{marker} ', c=color, mec=color, ms=2.5, mfc='#FFFFFFCC',
        capsize=1
    )

    if data_sim is not None and pars_fit is not None:
        sim_icounts = data_sim.cumsum(1)
        sim_imodel = model.counts(pars_fit, *other, data.spec_exposure).cumsum(1)
        sim_total = sim_imodel[:, -1:]
        sim_icounts /= sim_total
        sim_imodel /= sim_total
        for cl in [0.683, 0.95]:
            ci = _quantile(sim_icounts - sim_imodel, cl, hdi, axis=0)
            ci_lower = _split_step_plot(data.ch_emin, data.ch_emax, ci[0])
            ci_upper = _split_step_plot(data.ch_emin, data.ch_emax, ci[1])
            for i, j in zip(ci_lower, ci_upper):
                ax.fill_between(*i, j[1], lw=0, step='post', color=color, alpha=0.15)

            # if cl == 0.683:
            #     mask = (idiff < ci[0]) | (idiff > ci[1])
            #     ax.scatter(emid[mask], idiff[mask], marker=marker, c='orange', s=2.5)

            if cl == 0.95:
                mask = (idiff < ci[0]) | (idiff > ci[1])
                ax.scatter(emid[mask], idiff[mask], marker='x', c='r', zorder=10)


# def _plot_edf(ax, data, model, pars, marker, color, alpha, sim_data=None, hdi=True):
#     color = color + hex(round(255 * alpha))[2:]
#
#     counts = data.net_counts
#     icounts = counts.cumsum()
#     total_counts = icounts[-1]
#     icounts /= np.abs(total_counts)
#
#     other = [data.ph_ebins, data.ch_emin, data.ch_emax, data.resp_matrix]
#     model_counts = model.counts(pars, *other, data.spec_exposure)
#     model_icounts = model_counts.cumsum()
#     model_icounts /= model_icounts[-1]
#
#     emid = data.ch_emid_geom
#     eerr = np.abs(data.ch_error)
#     ax.errorbar(
#         emid, icounts - model_icounts, xerr=eerr,
#         fmt=f'{marker} ', c=color, mec=color, ms=2.5, mfc='#FFFFFFCC',
#         capsize=1
#     )
#
#     if sim_data is not None:
#         sim_icounts = sim_data.cumsum(axis=1)
#         sim_icounts /= sim_icounts[:, -1:]  # np.abs(total_counts)
#         for cl in [0.683, 0.95]:
#             ci = _quantile(sim_icounts - model_icounts, cl, hdi, axis=0)
#             ci_lower = _split_step_plot(data.ch_emin, data.ch_emax, ci[0])
#             ci_upper = _split_step_plot(data.ch_emin, data.ch_emax, ci[1])
#             for i, j in zip(ci_lower, ci_upper):
#                 ax.fill_between(*i, j[1], step='post', color=color, alpha=0.15)


def _plot_delchi(
    ax, data, model, pars, marker, color, alpha, comps,
    pars_dist=None, pars_fit=None, data_sim=None,
    deviance_obs=None, deviance_dist=None,
    hdi=True
):
    emid = data.ch_emid_geom
    eerr = np.abs(data.ch_error)
    net_spec = data.net_spec
    net_err = data.net_error#_gehrels

    other = [data.ph_ebins, data.ch_emin, data.ch_emax, data.resp_matrix]
    CE = model.CE(pars, *other)

    color = color + hex(round(255 * alpha))[2:]
    mfc = '#FFFFFFCC'
    delchi = (net_spec - CE) / net_err
    ax.errorbar(
        emid, delchi, 1, eerr,
        fmt=f'{marker} ', c=color, label=data.name, ms=2.5, mfc=mfc,
        mec=color, capsize=1
    )

    if data_sim is not None and pars_fit is not None:
        sim_CE = data_sim / data.ch_width / data.spec_exposure
        model_CE = model.CE(pars_fit, *other)
        for cl in [0.683, 0.95]:
            ci = _quantile((sim_CE - model_CE)/sim_CE.std(axis=0, ddof=1), cl, hdi, axis=0)
            ci_lower = _split_step_plot(data.ch_emin, data.ch_emax, ci[0])
            ci_upper = _split_step_plot(data.ch_emin, data.ch_emax, ci[1])
            for i, j in zip(ci_lower, ci_upper):
                ax.fill_between(*i, j[1], lw=0, step='post', color=color, alpha=0.15)

            if cl == 0.95:
                mask = (delchi < ci[0]) | (delchi > ci[1])
                ax.scatter(emid[mask], delchi[mask], marker='x', c='r', zorder=10)


def _plot_ratio(
    ax, data, model, pars, marker, color, alpha, comps,
    pars_dist=None, pars_fit=None, data_sim=None,
    deviance_obs=None, deviance_dist=None,
    hdi=True
):
    emid = data.ch_emid_geom
    eerr = np.abs(data.ch_error)
    net_spec = data.net_spec
    net_err = data.net_error#_gehrels

    other = [data.ph_ebins, data.ch_emin, data.ch_emax, data.resp_matrix]
    CE = model.CE(pars, *other)

    color = color + hex(round(255 * alpha))[2:]
    mfc = '#FFFFFFCC'
    ratio = net_spec / CE
    ax.errorbar(
        emid, ratio, net_err/CE, eerr,
        fmt=f'{marker} ', c=color, label=data.name, ms=2.5, mfc=mfc,
        mec=color, capsize=1
    )

    if data_sim is not None and pars_fit is not None:
        sim_CE = data_sim / data.ch_width / data.spec_exposure
        model_CE = model.CE(pars_fit, *other)
        for cl in [0.683, 0.95]:
            ci = _quantile(sim_CE/model_CE, cl, hdi, axis=0)
            ci_lower = _split_step_plot(data.ch_emin, data.ch_emax, ci[0])
            ci_upper = _split_step_plot(data.ch_emin, data.ch_emax, ci[1])
            for i, j in zip(ci_lower, ci_upper):
                ax.fill_between(*i, j[1], lw=0, step='post', color=color, alpha=0.15)

            if cl == 0.95:
                mask = (ratio < ci[0]) | (ratio > ci[1])
                ax.scatter(emid[mask], ratio[mask], marker='x', c='r', zorder=10)


def _plot_residuals(
    ax, data, model, pars, marker, color, alpha, comps,
    pars_dist=None, pars_fit=None, data_sim=None,
    deviance_obs=None, deviance_dist=None,
    hdi=True
):
    emid = data.ch_emid_geom
    eerr = np.abs(data.ch_error)
    net_spec = data.net_spec
    net_err = data.net_error#_gehrels

    other = [data.ph_ebins, data.ch_emin, data.ch_emax, data.resp_matrix]
    CE = model.CE(pars, *other)

    color = color + hex(round(255 * alpha))[2:]
    diff = net_spec - CE
    ax.errorbar(
        emid, diff, net_err, eerr,
        fmt=f'{marker} ', c=color, label=data.name, ms=2.5, mfc='#FFFFFFCC',
        mec=color, capsize=1
    )

    if data_sim is not None and pars_fit is not None:
        sim_CE = data_sim / data.ch_width / data.spec_exposure
        model_CE = model.CE(pars_fit, *other)
        for cl in [0.683, 0.95]:
            ci = _quantile(sim_CE - model_CE, cl, hdi, axis=0)
            # ci = np.quantile(sim_CE - CE, q=[0.5 - cl / 2.0, 0.5 + cl / 2.0], axis=0)
            ci_lower = _split_step_plot(data.ch_emin, data.ch_emax, ci[0])
            ci_upper = _split_step_plot(data.ch_emin, data.ch_emax, ci[1])
            for i, j in zip(ci_lower, ci_upper):
                ax.fill_between(*i, j[1], lw=0, step='post', color=color, alpha=0.15)

            if cl == 0.95:
                mask = (diff < ci[0]) | (diff > ci[1])
                ax.scatter(emid[mask], diff[mask], marker='x', c='r', zorder=10)


def _plot_deviance(
    ax, data, model, pars, marker, color, alpha, comps,
    pars_dist=None, pars_fit=None, data_sim=None,
    deviance_obs=None, deviance_dist=None,
    hdi=True
):
    emid = data.ch_emid_geom
    eerr = np.abs(data.ch_error)
    color = color + hex(round(255 * alpha))[2:]
    ax.errorbar(
        emid, deviance_obs, xerr=eerr,
        fmt=f'{marker} ', c=color, mec=color, ms=2.5, mfc='#FFFFFFCC',
        capsize=1
    )

    if deviance_dist is not None:
        for cl in [0.683, 0.95]:
            upper = np.quantile(deviance_dist, q=cl, axis=0)
            for i in _split_step_plot(data.ch_emin, data.ch_emax, upper):
                ax.fill_between(*i, lw=0, step='post', color=color, alpha=0.15)

            if cl == 0.95:
                mask = deviance_obs > upper
                ax.scatter(emid[mask], deviance_obs[mask], marker='x', c='r', zorder=10)


def _plot_sign_deviance(
    ax, data, model, pars, marker, color, alpha, comps,
    pars_dist=None, pars_fit=None, data_sim=None,
    deviance_obs=None, deviance_dist=None,
    hdi=True
):
    emid = data.ch_emid_geom
    eerr = np.abs(data.ch_error)
    net_spec = data.net_spec

    other = [data.ph_ebins, data.ch_emin, data.ch_emax, data.resp_matrix]
    sign = np.sign(net_spec - model.CE(pars, *other))

    color = color + hex(round(255 * alpha))[2:]

    sign_deviance = sign * np.sqrt(deviance_obs)
    ax.errorbar(
        emid, sign_deviance, xerr=eerr,
        fmt=f'{marker} ', c=color, mec=color, ms=2.5, mfc='#FFFFFFCC',
        capsize=1
    )

    if deviance_dist is not None:
        for cl in [0.683, 0.95]:
            upper = np.quantile(deviance_dist, q=cl, axis=0)
            for i in _split_step_plot(data.ch_emin, data.ch_emax, upper):
                ui = np.sqrt(i[1])
                ax.fill_between(i[0], -ui, ui,  lw=0, step='post', color=color, alpha=0.15)

            if cl == 0.95:
                mask = deviance_obs > upper
                ax.scatter(emid[mask], sign_deviance[mask], marker='x', c='r', zorder=10)

def _plot_sign_deviance2(
    ax, data, model, pars, marker, color, alpha, comps,
    pars_dist=None, pars_fit=None, data_sim=None,
    deviance_obs=None, deviance_dist=None,
    hdi=True
):
    emid = data.ch_emid_geom
    eerr = np.abs(data.ch_error)
    net_spec = data.net_spec

    other = [data.ph_ebins, data.ch_emin, data.ch_emax, data.resp_matrix]
    sign = np.sign(net_spec - model.CE(pars, *other))

    color = color + hex(round(255 * alpha))[2:]

    sign_deviance = sign * np.sqrt(deviance_obs)
    ax.errorbar(
        emid, sign_deviance, xerr=eerr,
        fmt=f'{marker} ', c=color, mec=color, ms=2.5, mfc='#FFFFFFCC',
        capsize=1
    )

    if deviance_dist is not None:
        for cl in [0.683, 0.95]:
            sim_sign = np.sign(data_sim - model.CE(pars_fit, *other))
            ci = _quantile(sim_sign * deviance_dist**0.5, cl, hdi, axis=0)
            ci_lower = _split_step_plot(data.ch_emin, data.ch_emax, ci[0])
            ci_upper = _split_step_plot(data.ch_emin, data.ch_emax, ci[1])
            for i, j in zip(ci_lower, ci_upper):
                ax.fill_between(*i, j[1], lw=0, step='post', color=color, alpha=0.15)

            if cl == 0.95:
                mask = (sign_deviance < ci[0]) | (sign_deviance > ci[1])
                ax.scatter(emid[mask], sign_deviance[mask], marker='x', c='r', zorder=10)