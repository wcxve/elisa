"""Methods for grouping spectrum."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from scipy.special import xlogy

if TYPE_CHECKING:
    NDArray = np.ndarray


def group_const(n: int, c: int) -> tuple[NDArray, bool]:
    """Group data by containing `c` channels in each group.

    Parameters
    ----------
    n : int
        Number of channels.
    c : int
        Number of channels in each group.

    Returns
    -------
    flag : ndarray
        Grouping flag.
    success: bool
        Whether the scale is met for all grouped channels.
    """
    n = int(n)
    c = int(c)
    assert c > 0
    flag = np.full(n, -1, dtype=int)
    flag[::c] = 1
    r = n % c
    if r:  # Ensure the last group has at least `c` channels.
        flag[-r] = -1
    return flag, True


def group_min(data: NDArray, n: int) -> tuple[NDArray, bool]:
    """Group data by containing at least `n` counts in each channel.

    Parameters
    ----------
    data: array_like
        Counts array.
    n: int
        Minimum number of counts in each channel after grouping.

    Returns
    -------
    flag : ndarray
        Grouping flag.
    success: bool
        Whether the scale is met for all grouped channels.
    """
    n = int(n)
    assert n > 0
    nd = len(data)
    idx = np.empty(nd, np.int64)
    idx[0] = 0

    group_counts = 0
    ng = 1
    imax = nd - 1

    for i, di in enumerate(data):
        group_counts += di

        if i == imax:
            if group_counts < n and ng > 1:
                # if the last group does not have enough counts,
                # then combine the last two groups to ensure all
                # groups meet the count requirement
                ng -= 1

            break

        if group_counts >= n:
            idx[ng] = i + 1
            ng += 1
            group_counts = 0

    idx = idx[:ng]
    if np.all(np.add.reduceat(data, idx) >= n):
        success = True
    else:
        success = False

    flag = np.full(nd, -1, dtype=int)
    flag[idx] = 1

    return flag, success


# TODO: the total number of resolution elements R should be given as argument
def _calc_optimal_binning(fwhm: NDArray, counts: NDArray) -> NDArray:
    """Calculate the optimal binning for each channel.

    .. note::
        This is translated from the ``heasp``.
    """
    # The optimal binning from Kaastra & Bleeker 2016, A&A 587, A151.
    # The binning delta/FWHM is defined as
    #    1  if x <= 2.119 otherwise
    #    (0.08 + 7.0/x + 1.8/x^2)/(1 + 5.9/x)
    # where
    #    x = ln[N_r(1 + 0.2 ln R)]
    # and N_r is the number of counts per resolution element and R is the total
    # number of resolution elements.
    Nchan = len(fwhm)

    # Estimate the total number of resolution elements. Since this enters
    # as ln(0.2ln R) it can be a rough estimate. See eqn B.1 of K&B.
    logR = np.log(1 + np.sum(1.0 / fwhm))

    # Calculate the number of counts within the FWHM for each channel then
    # multiply by 1.314 to get the number of counts per resolution element.
    # This assumes the response is gaussian - I could improve this by also
    # including a vector with the fraction within the FWHM for each channel
    # but this is probably not going to make a significant difference
    Nr = np.zeros(Nchan)
    for i in range(Nchan):
        fwhm_i = fwhm[i]
        low = max(0, int(np.round(i - 0.5 * fwhm_i)))
        high = min(Nchan, int(np.round(i + 0.5 * fwhm_i)))
        Nr[i] = 1.314 * counts[low : high + 1].sum()

    # Calculate the optimal bin size at each channel
    b = np.array(fwhm, dtype=np.float64)
    mask = Nr > np.exp(2.119) / (1 + 0.2 * logR)
    x = np.log(Nr[mask] * (1.0 + 0.2 * logR))
    b[mask] *= (0.08 * x + 7.0 + 1.8 / x) / (x + 5.9)
    bint = b.astype(np.int64)
    bint[bint < 1] = 1
    return bint


def group_opt(
    fwhm: NDArray,
    net_counts: NDArray,
    bin_counts: NDArray | None = None,
    n: int | None = None,
) -> tuple[NDArray, bool]:
    """Optimal binning of the spectrum data [1]_.

    Parameters
    ----------
    fwhm : ndarray
        FWHM of the channel.
    net_counts : ndarray
        Net counts of the channel.
    bin_counts : ndarray, optional
        Additional counts to be grouped.
    n : int, optional
        Grouping scale of the `bin_counts`.

    Returns
    -------
    flag : ndarray
        Grouping flag.
    success: bool
        Whether the scale is met for all grouped channels.

    References
    ----------
    .. [1] Kaastra & Bleeker 2016, A&A 587, A151
    """
    assert len(fwhm) == len(net_counts)
    nchan = len(fwhm)

    # Calculate the optimal binsize for each channel based on
    # the FWHM and the counts.
    bint = _calc_optimal_binning(fwhm, net_counts)

    # Initialize the grouping flag
    if n is not None:
        n = int(n)
    else:
        n = 0
    if bin_counts is not None:
        bin_counts = np.array(bin_counts)
    min_flag = (n > 0) and (bin_counts is not None)
    idx = np.empty(nchan, dtype=np.int64)
    ng = 0

    i = 0
    imax = nchan - 1
    while i <= imax:
        idx[ng] = i
        ng += 1

        j = min(imax, i + bint[i] - 1)
        k = np.arange(i + 1, j + 1)
        if k.size:
            j = np.minimum(j, np.min(k + bint[k] - 1))

        # Ensure minimum number of counts and extend the bin
        # if necessary to abide that constraint
        if min_flag:
            cts = bin_counts[i : j + 1].sum()
            if cts < n:
                mask = bin_counts[j + 1 :].cumsum() >= n - cts
                if np.any(mask):
                    # the smallest j for bin_counts[i:j + 1].sum() >= n
                    j += np.flatnonzero(mask)[0] + 1
                else:
                    # if the last group does not have enough counts,
                    # then combine the last two groups to ensure all
                    # groups meet the count requirement
                    if ng > 1:
                        ng -= 1
                    break

        if j > imax:
            # If the left channels are not enough, then group them with
            # the previous channels group.
            if (j - imax) / (j - i) > 1 / 3 and ng > 1:
                ng -= 1
            break

        i = j + 1

    idx = idx[:ng]
    flag = np.full(nchan, -1, dtype=int)
    flag[idx] = 1

    if min_flag and np.any(np.add.reduceat(bin_counts, idx) < n):
        success = False
    else:
        success = True

    return flag, success


class _ScaleGroupData(NamedTuple):
    """Grouped scale values plus additive metadata for regrouping."""

    scale: NDArray
    counts: NDArray
    n_chan: NDArray
    weighted_inv_sum: NDArray
    reciprocal_sum: NDArray
    scale_sum: NDArray


class _ScalePrefixData(NamedTuple):
    """Prefix sums for efficient interval scale aggregation."""

    counts: NDArray
    n_chan: NDArray
    weighted_inv_sum: NDArray
    reciprocal_sum: NDArray
    scale_sum: NDArray
    net: bool


def _prefix_sum(values: NDArray) -> NDArray:
    """Return a prefix-sum array with a leading zero."""
    values = np.asarray(values, dtype=np.float64)
    return np.concatenate(([0.0], np.cumsum(values)))


def _interval_sum(prefix: NDArray, start: int, stop: int) -> np.float64:
    """Return the half-open interval sum from a prefix-sum array."""
    return np.float64(prefix[stop] - prefix[start])


def _build_scale_prefix(data: _ScaleGroupData, net: bool) -> _ScalePrefixData:
    """Build prefix sums used to regroup already-grouped scale metadata."""
    return _ScalePrefixData(
        counts=_prefix_sum(data.counts),
        n_chan=_prefix_sum(data.n_chan),
        weighted_inv_sum=_prefix_sum(data.weighted_inv_sum),
        reciprocal_sum=_prefix_sum(data.reciprocal_sum),
        scale_sum=_prefix_sum(data.scale_sum),
        net=bool(net),
    )


def _scale_from_interval(
    prefix: _ScalePrefixData, start: int, stop: int
) -> np.float64:
    """Evaluate a grouped scale value on ``[start, stop)``.

    NET spectra use an arithmetic mean. Non-NET spectra use the
    count-weighted harmonic mean with a plain harmonic fallback when the
    weighted denominator is zero.
    """
    counts = prefix.counts[stop] - prefix.counts[start]
    n_chan = prefix.n_chan[stop] - prefix.n_chan[start]
    if n_chan == 0.0:
        return np.float64(1.0)

    if prefix.net:
        scale_sum = prefix.scale_sum[stop] - prefix.scale_sum[start]
        return np.float64(scale_sum / n_chan)

    weighted_inv_sum = (
        prefix.weighted_inv_sum[stop] - prefix.weighted_inv_sum[start]
    )
    if weighted_inv_sum != 0.0:
        return np.float64(counts / weighted_inv_sum)

    reciprocal_sum = prefix.reciprocal_sum[stop] - prefix.reciprocal_sum[start]
    if reciprocal_sum == 0.0:
        return np.float64(1.0)

    return np.float64(n_chan / reciprocal_sum)


def _group_scale_data(
    counts: NDArray,
    scale: NDArray,
    grouping: NDArray,
    quality: NDArray | None = None,
    *,
    net: bool,
    n_chan: NDArray | None = None,
    weighted_inv_sum: NDArray | None = None,
    reciprocal_sum: NDArray | None = None,
    scale_sum: NDArray | None = None,
) -> _ScaleGroupData:
    """Group a scale array together with sufficient statistics.

    The returned metadata can be reused to regroup the same channels again
    without reconstructing the original per-channel scale inputs.
    """
    counts = np.asarray(counts, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    grouping = np.asarray(grouping, dtype=np.int64)

    if counts.shape != scale.shape:
        raise ValueError('counts and scale must have the same shape')
    if grouping.shape != counts.shape:
        raise ValueError('grouping must have the same shape as counts')

    if quality is None:
        factor = np.ones(counts.shape, dtype=np.float64)
    else:
        quality = np.asarray(quality)
        if quality.shape != counts.shape:
            raise ValueError('quality must have the same shape as counts')
        factor = quality.astype(np.float64)

    if n_chan is None:
        n_chan = np.ones(counts.shape, dtype=np.float64)
    else:
        n_chan = np.asarray(n_chan, dtype=np.float64)
        if n_chan.shape != counts.shape:
            raise ValueError('n_chan must have the same shape as counts')

    if weighted_inv_sum is None:
        weighted_inv_sum = np.zeros(counts.shape, dtype=np.float64)
        np.divide(
            counts,
            scale,
            out=weighted_inv_sum,
            where=scale != 0.0,
        )
    else:
        weighted_inv_sum = np.asarray(weighted_inv_sum, dtype=np.float64)
        if weighted_inv_sum.shape != counts.shape:
            raise ValueError(
                'weighted_inv_sum must have the same shape as counts'
            )

    if reciprocal_sum is None:
        reciprocal_sum = np.zeros(counts.shape, dtype=np.float64)
        np.divide(
            1.0,
            scale,
            out=reciprocal_sum,
            where=scale != 0.0,
        )
    else:
        reciprocal_sum = np.asarray(reciprocal_sum, dtype=np.float64)
        if reciprocal_sum.shape != counts.shape:
            raise ValueError(
                'reciprocal_sum must have the same shape as counts'
            )

    if scale_sum is None:
        scale_sum = np.asarray(scale, dtype=np.float64)
    else:
        scale_sum = np.asarray(scale_sum, dtype=np.float64)
        if scale_sum.shape != counts.shape:
            raise ValueError('scale_sum must have the same shape as counts')

    group_idx = np.flatnonzero(grouping != -1)
    group_counts = np.add.reduceat(factor * counts, group_idx)
    group_n_chan = np.add.reduceat(factor * n_chan, group_idx)
    group_weighted_inv = np.add.reduceat(factor * weighted_inv_sum, group_idx)
    group_reciprocal = np.add.reduceat(factor * reciprocal_sum, group_idx)
    group_scale_sum = np.add.reduceat(factor * scale_sum, group_idx)

    grouped_scale = np.ones(group_counts.shape, dtype=np.float64)
    if net:
        np.divide(
            group_scale_sum,
            group_n_chan,
            out=grouped_scale,
            where=group_n_chan != 0.0,
        )
    else:
        mask = group_weighted_inv != 0.0
        np.divide(
            group_counts,
            group_weighted_inv,
            out=grouped_scale,
            where=mask,
        )
        harmonic = np.ones(group_counts.shape, dtype=np.float64)
        np.divide(
            group_n_chan,
            group_reciprocal,
            out=harmonic,
            where=group_reciprocal != 0.0,
        )
        grouped_scale[~mask] = harmonic[~mask]

    return _ScaleGroupData(
        scale=grouped_scale,
        counts=group_counts,
        n_chan=group_n_chan,
        weighted_inv_sum=group_weighted_inv,
        reciprocal_sum=group_reciprocal,
        scale_sum=group_scale_sum,
    )


def _slice_scale_data(
    data: _ScaleGroupData, masks: list[NDArray]
) -> list[_ScaleGroupData]:
    """Apply boolean masks to every field of grouped scale metadata."""
    fields = data._fields
    return [
        _ScaleGroupData(
            *[np.asarray(getattr(data, field))[mask] for field in fields]
        )
        for mask in masks
    ]


def _group_back_ratio(
    spec_exposure: float,
    back_exposure: float,
    spec_area: _ScaleGroupData,
    spec_back: _ScaleGroupData,
    back_area: _ScaleGroupData,
    back_back: _ScaleGroupData,
) -> NDArray:
    """Return grouped source-to-background effective exposure ratios."""
    return (
        spec_exposure
        * spec_area.scale
        * spec_back.scale
        / (back_exposure * back_area.scale * back_back.scale)
    )


def _interval_back_ratio(
    start: int,
    stop: int,
    spec_exposure: float,
    back_exposure: float,
    spec_area: _ScalePrefixData,
    spec_back: _ScalePrefixData,
    back_area: _ScalePrefixData,
    back_back: _ScalePrefixData,
) -> np.float64:
    """Return the background ratio for one grouped interval."""
    ratio = spec_exposure / back_exposure
    ratio *= _scale_from_interval(spec_area, start, stop)
    ratio *= _scale_from_interval(spec_back, start, stop)
    ratio /= _scale_from_interval(back_area, start, stop)
    ratio /= _scale_from_interval(back_back, start, stop)
    return np.float64(ratio)


def _interval_ratios(
    idx: NDArray,
    size: int,
    spec_exposure: float,
    back_exposure: float,
    spec_area: _ScalePrefixData,
    spec_back: _ScalePrefixData,
    back_area: _ScalePrefixData,
    back_back: _ScalePrefixData,
) -> NDArray:
    """Return grouped background ratios for all bins defined by ``idx``."""
    edge = np.append(idx, size)
    ratios = np.empty(len(idx), dtype=np.float64)
    for i, (start, stop) in enumerate(zip(edge[:-1], edge[1:], strict=True)):
        ratios[i] = _interval_back_ratio(
            start,
            stop,
            spec_exposure,
            back_exposure,
            spec_area,
            spec_back,
            back_area,
            back_back,
        )
    return ratios


def significance_lima(
    n_on: float | NDArray,
    n_off: float | NDArray,
    a: float | NDArray,
) -> NDArray:
    """Significance using the formula of Li & Ma 1983."""
    n_on = np.asarray(n_on, dtype=np.float64)
    n_off = np.asarray(n_off, dtype=np.float64)
    a = np.broadcast_to(np.asarray(a, dtype=np.float64), n_on.shape)

    term1 = np.zeros_like(n_on, dtype=np.float64)
    mask = n_on > 0.0
    if mask.any():
        a_mask = a[mask]
        term1[mask] = xlogy(
            n_on[mask],
            (1.0 + a_mask) / a_mask * n_on[mask] / (n_on[mask] + n_off[mask]),
        )

    term2 = np.zeros_like(n_on, dtype=np.float64)
    mask = n_off > 0.0
    if mask.any():
        a_mask = a[mask]
        term2[mask] = xlogy(
            n_off[mask],
            (1.0 + a_mask) * n_off[mask] / (n_on[mask] + n_off[mask]),
        )
    sign = np.where(n_on >= a * n_off, 1.0, -1.0)
    return sign * np.sqrt(2.0 * (term1 + term2))


def significance_gv(
    n: float | NDArray,
    b: float | NDArray,
    s: float | NDArray,
    a: float | NDArray,
) -> NDArray:
    """Significance using the formula of Vianello 2018."""
    n = np.asarray(n, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    a = np.broadcast_to(np.asarray(a, dtype=np.float64), n.shape)

    b = b * a
    s = s * a
    s2 = s * s
    s4 = s2 * s2
    b0_mle = 0.5 * (b - s2 + np.sqrt(b * b - 2 * b * s2 + 4 * n * s2 + s4))
    b0_mle = np.clip(b0_mle, 0, None)

    term1 = np.zeros_like(n).astype(float)
    mask = n > 0.0
    if mask.any():
        n_mask = n[mask]
        b0_mle_mask = b0_mle[mask]
        term1[mask] = xlogy(n_mask, n_mask / b0_mle_mask)
    term1 += b0_mle - n

    term2 = np.square(b - b0_mle) / (2 * s2)
    sign = np.where(n >= b, 1.0, -1.0)
    return sign * np.sqrt(2.0 * (term1 + term2))


def group_optsig_normal(
    fwhm: NDArray,
    net_counts: NDArray,
    counts: NDArray,
    errors: NDArray,
    sig: int | float,
) -> tuple[NDArray, bool]:
    """Optimal binning with an extra requirement of a minimum significance.

    Parameters
    ----------
    fwhm : ndarray
        FWHM of the channel.
    net_counts : ndarray
        Net counts of the channel.
    counts : ndarray
        Counts of the channel.
    errors : ndarray
        Uncertainty of the counts.
    sig : float
        Significance threshold.

    Returns
    -------
    flag : ndarray
        Grouping flag.
    success: bool
        Whether the scale is met for all grouped channels.
    """
    assert len(fwhm) == len(net_counts) == len(counts) == len(errors)
    sig = float(sig)
    assert sig > 0.0
    nchan = len(fwhm)

    # Calculate the optimal binsize for each channel based on
    # the FWHM and the counts.
    bint = _calc_optimal_binning(fwhm, net_counts)

    # Initialize the grouping flag
    idx = np.empty(nchan, dtype=np.int64)
    ng = 0

    i = 0
    imax = nchan - 1
    while i <= imax:
        idx[ng] = i
        ng += 1

        j = min(imax, i + bint[i] - 1)
        k = np.arange(i + 1, j + 1)
        if k.size:
            j = np.minimum(j, np.min(k + bint[k] - 1))

        # Ensure minimum significance and extend the bin if necessary to abide
        # that constraint
        cts = counts[i : j + 1].sum()
        err = np.sqrt(np.sum(np.square(errors[i : j + 1])))
        if cts - sig * err < 0.0 or err == 0.0:
            cts = cts + counts[j + 1 :].cumsum()
            err = err + np.sqrt(np.cumsum(np.square(errors[j + 1 :])))
            mask = (cts - sig * err >= 0.0) & (err > 0.0)
            if np.any(mask):
                # the smallest j for the significance threshold
                j += np.flatnonzero(mask)[0] + 1
            else:
                # if the last group does not have a nominal significance,
                # then combine the last two groups to ensure all
                # groups meet the count requirement
                if ng > 1:
                    ng -= 1
                break

        if j > imax:
            # If the left channels are not enough, then group them with
            # the previous channels group.
            if (j - imax) / (j - i) > 1 / 3 and ng > 1:
                ng -= 1
            break

        i = j + 1

    idx = idx[:ng]
    flag = np.full(nchan, -1, dtype=int)
    flag[idx] = 1

    cts = np.add.reduceat(counts, idx)
    err = np.sqrt(np.add.reduceat(errors * errors, idx))
    if np.all(cts - sig * err >= 0.0):
        success = True
    else:
        success = False

    return flag, success


def group_sig_lima(
    n_on: NDArray,
    n_off: NDArray,
    *,
    spec_exposure: float,
    back_exposure: float,
    spec_area: _ScaleGroupData,
    spec_back: _ScaleGroupData,
    back_area: _ScaleGroupData,
    back_back: _ScaleGroupData,
    sig: float,
) -> tuple[NDArray, bool]:
    """Group Poisson source/background data by Li & Ma significance.

    The source-to-background ratio is recomputed for every candidate bin from
    the grouped scale metadata instead of assuming a single constant ratio.
    Significance grouping always treats the source scales as total-spectrum
    scales, so NET averaging is not used here.
    """
    assert len(n_on) == len(n_off)
    sig = float(sig)
    assert sig > 0.0

    spec_area_prefix = _build_scale_prefix(spec_area, net=False)
    spec_back_prefix = _build_scale_prefix(spec_back, net=False)
    back_area_prefix = _build_scale_prefix(back_area, net=False)
    back_back_prefix = _build_scale_prefix(back_back, net=False)
    on_prefix = _prefix_sum(n_on)
    off_prefix = _prefix_sum(n_off)

    nd = len(n_on)
    idx = np.empty(nd, np.int64)
    idx[0] = 0
    ng = 1
    imax = nd - 1
    start = 0

    for i in range(nd):
        group_on = _interval_sum(on_prefix, start, i + 1)
        group_off = _interval_sum(off_prefix, start, i + 1)
        ratio = _interval_back_ratio(
            start,
            i + 1,
            spec_exposure,
            back_exposure,
            spec_area_prefix,
            spec_back_prefix,
            back_area_prefix,
            back_back_prefix,
        )
        group_sig = significance_lima(group_on, group_off, ratio)

        if i == imax:
            if group_sig < sig and ng > 1:
                ng -= 1
            break

        if group_sig >= sig:
            idx[ng] = i + 1
            ng += 1
            start = i + 1

    idx = idx[:ng]
    group_on = np.add.reduceat(n_on, idx)
    group_off = np.add.reduceat(n_off, idx)
    ratios = _interval_ratios(
        idx,
        nd,
        spec_exposure,
        back_exposure,
        spec_area_prefix,
        spec_back_prefix,
        back_area_prefix,
        back_back_prefix,
    )
    success = bool(
        np.all(significance_lima(group_on, group_off, ratios) >= sig)
    )

    flag = np.full(nd, -1, dtype=int)
    flag[idx] = 1
    return flag, success


def group_sig_gv(
    n: NDArray,
    b: NDArray,
    s: NDArray,
    *,
    spec_exposure: float,
    back_exposure: float,
    spec_area: _ScaleGroupData,
    spec_back: _ScaleGroupData,
    back_area: _ScaleGroupData,
    back_back: _ScaleGroupData,
    sig: float,
) -> tuple[NDArray, bool]:
    """Group Poisson data with Gaussian background by GV significance.

    The source-to-background ratio is recomputed for every candidate bin from
    the grouped scale metadata instead of assuming a single constant ratio.
    Significance grouping always treats the source scales as total-spectrum
    scales, so NET averaging is not used here.
    """
    assert len(n) == len(b) == len(s)
    sig = float(sig)
    assert sig > 0.0

    spec_area_prefix = _build_scale_prefix(spec_area, net=False)
    spec_back_prefix = _build_scale_prefix(spec_back, net=False)
    back_area_prefix = _build_scale_prefix(back_area, net=False)
    back_back_prefix = _build_scale_prefix(back_back, net=False)
    n_prefix = _prefix_sum(n)
    b_prefix = _prefix_sum(b)
    s2_prefix = _prefix_sum(s * s)

    nd = len(n)
    idx = np.empty(nd, np.int64)
    idx[0] = 0
    ng = 1
    imax = nd - 1
    start = 0

    for i in range(nd):
        group_n = _interval_sum(n_prefix, start, i + 1)
        group_b = _interval_sum(b_prefix, start, i + 1)
        group_s = np.sqrt(_interval_sum(s2_prefix, start, i + 1))
        ratio = _interval_back_ratio(
            start,
            i + 1,
            spec_exposure,
            back_exposure,
            spec_area_prefix,
            spec_back_prefix,
            back_area_prefix,
            back_back_prefix,
        )
        group_sig = significance_gv(group_n, group_b, group_s, ratio)

        if i == imax:
            if group_sig < sig and ng > 1:
                ng -= 1
            break

        if group_sig >= sig:
            idx[ng] = i + 1
            ng += 1
            start = i + 1

    idx = idx[:ng]
    group_n = np.add.reduceat(n, idx)
    group_b = np.add.reduceat(b, idx)
    group_s = np.sqrt(np.add.reduceat(s * s, idx))
    ratios = _interval_ratios(
        idx,
        nd,
        spec_exposure,
        back_exposure,
        spec_area_prefix,
        spec_back_prefix,
        back_area_prefix,
        back_back_prefix,
    )
    success = bool(
        np.all(significance_gv(group_n, group_b, group_s, ratios) >= sig)
    )

    flag = np.full(nd, -1, dtype=int)
    flag[idx] = 1
    return flag, success


def group_optsig_lima(
    fwhm: NDArray,
    net_counts: NDArray,
    n_on: NDArray,
    n_off: NDArray,
    *,
    spec_exposure: float,
    back_exposure: float,
    spec_area: _ScaleGroupData,
    spec_back: _ScaleGroupData,
    back_area: _ScaleGroupData,
    back_back: _ScaleGroupData,
    sig: float,
) -> tuple[NDArray, bool]:
    """Optimally group Poisson source/background data."""
    assert len(fwhm) == len(net_counts) == len(n_on) == len(n_off)
    sig = float(sig)
    assert sig > 0.0
    nchan = len(fwhm)

    spec_area_prefix = _build_scale_prefix(spec_area, net=False)
    spec_back_prefix = _build_scale_prefix(spec_back, net=False)
    back_area_prefix = _build_scale_prefix(back_area, net=False)
    back_back_prefix = _build_scale_prefix(back_back, net=False)
    on_prefix = _prefix_sum(n_on)
    off_prefix = _prefix_sum(n_off)

    bint = _calc_optimal_binning(fwhm, net_counts)
    idx = np.empty(nchan, dtype=np.int64)
    ng = 0
    i = 0
    imax = nchan - 1
    while i <= imax:
        idx[ng] = i
        ng += 1

        j = min(imax, i + bint[i] - 1)
        k = np.arange(i + 1, j + 1)
        if k.size:
            j = np.minimum(j, np.min(k + bint[k] - 1))

        while j <= imax:
            group_on = _interval_sum(on_prefix, i, j + 1)
            group_off = _interval_sum(off_prefix, i, j + 1)
            ratio = _interval_back_ratio(
                i,
                j + 1,
                spec_exposure,
                back_exposure,
                spec_area_prefix,
                spec_back_prefix,
                back_area_prefix,
                back_back_prefix,
            )
            if significance_lima(group_on, group_off, ratio) >= sig:
                break
            j += 1

        if j > imax:
            if ng > 1:
                ng -= 1
            break

        i = j + 1

    idx = idx[:ng]
    flag = np.full(nchan, -1, dtype=int)
    flag[idx] = 1
    ratios = _interval_ratios(
        idx,
        nchan,
        spec_exposure,
        back_exposure,
        spec_area_prefix,
        spec_back_prefix,
        back_area_prefix,
        back_back_prefix,
    )
    success = bool(
        np.all(
            significance_lima(
                np.add.reduceat(n_on, idx),
                np.add.reduceat(n_off, idx),
                ratios,
            )
            >= sig
        )
    )
    return flag, success


def group_optsig_gv(
    fwhm: NDArray,
    net_counts: NDArray,
    n: NDArray,
    b: NDArray,
    s: NDArray,
    *,
    spec_exposure: float,
    back_exposure: float,
    spec_area: _ScaleGroupData,
    spec_back: _ScaleGroupData,
    back_area: _ScaleGroupData,
    back_back: _ScaleGroupData,
    sig: float,
) -> tuple[NDArray, bool]:
    """Optimally group Poisson data with Gaussian background."""
    assert len(fwhm) == len(net_counts) == len(n) == len(b) == len(s)
    sig = float(sig)
    assert sig > 0.0
    nchan = len(fwhm)

    spec_area_prefix = _build_scale_prefix(spec_area, net=False)
    spec_back_prefix = _build_scale_prefix(spec_back, net=False)
    back_area_prefix = _build_scale_prefix(back_area, net=False)
    back_back_prefix = _build_scale_prefix(back_back, net=False)
    n_prefix = _prefix_sum(n)
    b_prefix = _prefix_sum(b)
    s2_prefix = _prefix_sum(s * s)

    bint = _calc_optimal_binning(fwhm, net_counts)
    idx = np.empty(nchan, dtype=np.int64)
    ng = 0
    i = 0
    imax = nchan - 1
    while i <= imax:
        idx[ng] = i
        ng += 1

        j = min(imax, i + bint[i] - 1)
        k = np.arange(i + 1, j + 1)
        if k.size:
            j = np.minimum(j, np.min(k + bint[k] - 1))

        while j <= imax:
            group_n = _interval_sum(n_prefix, i, j + 1)
            group_b = _interval_sum(b_prefix, i, j + 1)
            group_s = np.sqrt(_interval_sum(s2_prefix, i, j + 1))
            ratio = _interval_back_ratio(
                i,
                j + 1,
                spec_exposure,
                back_exposure,
                spec_area_prefix,
                spec_back_prefix,
                back_area_prefix,
                back_back_prefix,
            )
            if significance_gv(group_n, group_b, group_s, ratio) >= sig:
                break
            j += 1

        if j > imax:
            if ng > 1:
                ng -= 1
            break

        i = j + 1

    idx = idx[:ng]
    flag = np.full(nchan, -1, dtype=int)
    flag[idx] = 1
    ratios = _interval_ratios(
        idx,
        nchan,
        spec_exposure,
        back_exposure,
        spec_area_prefix,
        spec_back_prefix,
        back_area_prefix,
        back_back_prefix,
    )
    success = bool(
        np.all(
            significance_gv(
                np.add.reduceat(n, idx),
                np.add.reduceat(b, idx),
                np.sqrt(np.add.reduceat(s * s, idx)),
                ratios,
            )
            >= sig
        )
    )
    return flag, success


def group_sig_normal(
    count: NDArray,
    error: NDArray,
    sig: int | float,
) -> tuple[NDArray, bool]:
    """Group data by limiting the significance is greater than `sig`.

    Parameters
    ----------
    count : ndarray
        Counts array.
    error : ndarray
        Uncertainty of counts.
    sig : int or float
        Significance threshold.

    Returns
    -------
    flag : ndarray
        Grouping flag.
    success: bool
        Whether the scale is met for all grouped channels.
    """
    assert len(count) == len(error)
    sig = float(sig)
    assert sig > 0.0

    nd = len(count)
    idx = np.empty(nd, np.int64)
    idx[0] = 0
    ng = 1
    imax = nd - 1

    group_count = 0.0
    group_variance = 0.0
    for i, (d, e) in enumerate(zip(count, error, strict=True)):
        group_count += d
        group_variance += e * e
        x = group_count - sig * np.sqrt(group_variance)

        if i == imax:
            if (x < 0.0 or group_variance == 0.0) and ng > 1:
                # if the last group does not have a nominal significance,
                # then combine the last two groups to ensure all
                # groups meet the count requirement
                ng -= 1

            break

        if x >= 0.0 and group_variance > 0.0:
            idx[ng] = i + 1
            ng += 1
            group_count = 0.0
            group_variance = 0.0

    idx = idx[:ng]
    group_count = np.add.reduceat(count, idx)
    group_error = np.sqrt(np.add.reduceat(error * error, idx))
    if np.all(group_count - sig * group_error >= 0):
        success = True
    else:
        success = False

    flag = np.full(nd, -1, dtype=int)
    flag[idx] = 1

    return flag, success
