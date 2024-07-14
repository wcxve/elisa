"""Methods for grouping spectrum."""

from __future__ import annotations

from typing import TYPE_CHECKING

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


def significance_lima(
    n_on: float | NDArray,
    n_off: float | NDArray,
    a: float | NDArray,
) -> NDArray:
    """Significance using the formula of Li & Ma 1983."""
    n_on = np.asarray(n_on)
    n_off = np.asarray(n_off)

    term1 = np.zeros_like(n_on).astype(float)
    mask = n_on > 0.0
    if mask.any():
        term1[mask] = xlogy(
            n_on[mask], (1.0 + a) / a * n_on[mask] / (n_on[mask] + n_off[mask])
        )

    term2 = np.zeros_like(n_on).astype(float)
    mask = n_off > 0.0
    if mask.any():
        term2[mask] = xlogy(
            n_off[mask], (1.0 + a) * n_off[mask] / (n_on[mask] + n_off[mask])
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
    n = np.asarray(n)
    b = np.asarray(b)
    s = np.asarray(s)

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


def group_optsig_lima(
    fwhm: NDArray,
    net_counts: NDArray,
    n_on: NDArray,
    n_off: NDArray,
    a: float,
    sig: int | float,
) -> tuple[NDArray, bool]:
    """Optimal binning with an extra requirement of a minimum significance.

    .. note::
        The formula of Li & Ma 1983 is used to calculate the significance.

    Parameters
    ----------
    fwhm : ndarray
        FWHM of the channel.
    net_counts : ndarray
        Net counts of the channel.
    n_on : ndarray
        Counts array of on measurement.
    n_off : ndarray
        Counts array of off measurement.
    a : float
        Ratio of the on and off exposure.
    sig : int or float
        Significance threshold.

    Returns
    -------
    flag : ndarray
        Grouping flag.
    success: bool
        Whether the scale is met for all grouped channels.
    """
    assert len(fwhm) == len(net_counts) == len(n_on) == len(n_off)
    assert a > 0.0
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
        on = n_on[i : j + 1].sum()
        off = n_off[i : j + 1].sum()
        if significance_lima(on, off, a) < sig:
            on = on + n_on[j + 1 :].cumsum()
            off = off + n_off[j + 1 :].cumsum()
            mask = significance_lima(on, off, a) >= sig
            if np.any(mask):
                # the smallest j for the significance threshold
                j += np.flatnonzero(mask)[0] + 1
            else:
                # if the last group does not have a nominal significance,
                # then combine the last two groups to ensure all
                # groups meet the count requirement
                if ng > 1:
                    ng -= 1

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

    on = np.add.reduceat(n_on, idx)
    off = np.add.reduceat(n_off, idx)
    if np.all(significance_lima(on, off, a) >= sig):
        success = True
    else:
        success = False

    return flag, success


def group_optsig_gv(
    fwhm: NDArray,
    net_counts: NDArray,
    n: NDArray,
    b: NDArray,
    s: NDArray,
    a: float,
    sig: int | float,
) -> tuple[NDArray, bool]:
    """Optimal binning with an extra requirement of a minimum significance.

    .. note::
        The formula of Vianello 2018 is used to calculate the significance.

    Parameters
    ----------
    fwhm : ndarray
        FWHM of the channel.
    net_counts : ndarray
        Net counts of the channel.
    n : ndarray
        Counts array of on measurement.
    b : ndarray
        Estimate of background counts.
    s : ndarray
        Uncertainty of the background estimate.
    a : float
        Ratio of the on and off exposure.
    sig : float
        Significance threshold.

    Returns
    -------
    flag : ndarray
        Grouping flag.
    success: bool
        Whether the scale is met for all grouped channels.
    """
    assert len(fwhm) == len(net_counts) == len(n) == len(b) == len(s)
    assert a > 0.0
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
        n_ = n[i : j + 1].sum()
        b_ = b[i : j + 1].sum()
        s_ = np.sqrt(np.sum(np.square(s[i : j + 1])))
        if significance_gv(n_, b_, s_, a) < sig:
            n_ = n_ + n[j + 1 :].cumsum()
            b_ = b_ + b[j + 1 :].cumsum()
            s_ = s_ + np.sqrt(np.cumsum(np.square(s[j + 1 :])))
            mask = significance_gv(n_, b_, s_, a) >= sig
            if np.any(mask):
                # the smallest j for the significance threshold
                j += np.flatnonzero(mask)[0] + 1
            else:
                # if the last group does not have a nominal significance,
                # then combine the last two groups to ensure all
                # groups meet the count requirement
                if ng > 1:
                    ng -= 1

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

    n_ = np.add.reduceat(n, idx)
    b_ = np.add.reduceat(b, idx)
    s_ = np.sqrt(np.add.reduceat(s * s, idx))
    if np.all(significance_gv(n_, b_, s_, a) >= sig):
        success = True
    else:
        success = False

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
    for i, (d, e) in enumerate(zip(count, error)):
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


def group_sig_lima(
    n_on: NDArray,
    n_off: NDArray,
    a: float,
    sig: float,
) -> tuple[NDArray, bool]:
    """Group data by limiting the significance is greater than `sig`.

    .. note::
        The formula of Li & Ma 1983 is used to calculate the significance.

    Parameters
    ----------
    n_on : ndarray
        Counts array of on measurement.
    n_off : ndarray
        Counts array of off measurement.
    a : float
        Ratio of the on and off exposure.
    sig : int or float
        Significance threshold.

    Returns
    -------
    flag : ndarray
        Grouping flag.
    success: bool
        Whether the scale is met for all grouped channels.
    """
    assert len(n_on) == len(n_off)
    assert a > 0.0
    sig = float(sig)
    assert sig > 0.0

    nd = len(n_on)
    idx = np.empty(nd, np.int64)
    idx[0] = 0
    ng = 1
    imax = nd - 1

    group_on = 0.0
    group_off = 0.0
    for i, (j, k) in enumerate(zip(n_on, n_off)):
        group_on += j
        group_off += k
        group_sig = significance_lima(group_on, group_off, a)

        if i == imax:
            if group_sig < sig and ng > 1:
                # if the last group does not have a nominal significance,
                # then combine the last two groups to ensure all
                # groups meet the count requirement
                ng -= 1

            break

        if group_sig >= sig:
            idx[ng] = i + 1
            ng += 1
            group_on = 0.0
            group_off = 0.0

    idx = idx[:ng]
    group_on = np.add.reduceat(n_on, idx)
    group_off = np.add.reduceat(n_off, idx)
    if np.all(significance_lima(group_on, group_off, a) >= sig):
        success = True
    else:
        success = False

    flag = np.full(nd, -1, dtype=int)
    flag[idx] = 1

    return flag, success


def group_sig_gv(
    n: NDArray,
    b: NDArray,
    s: NDArray,
    a: float,
    sig: float,
) -> tuple[NDArray, bool]:
    """Group data by limiting the significance is greater than `sig`.

    .. note::
        The formula of Vianello 2018 is used to calculate the significance.

    Parameters
    ----------
    n : ndarray
        Counts array of on measurement.
    b : ndarray
        Estimate of background counts.
    s : ndarray
        Uncertainty of the background estimate.
    a : float
        Ratio of the on and off exposure.
    sig : float
        Significance threshold.

    Returns
    -------
    flag : ndarray
        Grouping flag.
    success: bool
        Whether the scale is met for all grouped channels.
    """
    assert len(n) == len(b) == len(s)
    assert a > 0.0
    sig = float(sig)
    assert sig > 0.0

    nd = len(n)
    idx = np.empty(nd, np.int64)
    idx[0] = 0
    ng = 1
    imax = nd - 1

    group_n = 0.0
    group_b = 0.0
    group_var = 0.0
    for i, (ni, bi, si) in enumerate(zip(n, b, s)):
        group_n += ni
        group_b += bi
        group_var += si * si
        group_sig = significance_gv(group_n, group_b, np.sqrt(group_var), a)

        if i == imax:
            if group_sig < sig and ng > 1:
                # if the last group does not have a nominal significance,
                # then combine the last two groups to ensure all
                # groups meet the count requirement
                ng -= 1

            break

        if group_sig >= sig:
            idx[ng] = i + 1
            ng += 1
            group_n = 0.0
            group_b = 0.0
            group_var = 0.0

    idx = idx[:ng]
    group_n = np.add.reduceat(n, idx)
    group_b = np.add.reduceat(b, idx)
    group_var = np.sqrt(np.add.reduceat(s * s, idx))
    group_sig = significance_gv(group_n, group_b, np.sqrt(group_var), a)
    if np.all(group_sig >= sig):
        success = True
    else:
        success = False

    flag = np.full(nd, -1, dtype=int)
    flag[idx] = 1

    return flag, success
