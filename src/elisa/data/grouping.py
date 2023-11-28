"""Various method for grouping spectrum."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = [
    'group_const',
    'group_min',
    'group_sig',
    'group_pos',
    'group_opt',
    'group_optmin',
    'group_optsig',
]

GroupResultType = tuple[np.ndarray, bool]
NDArray = np.ndarray


def group_min(data: NDArray, n: int) -> GroupResultType:
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
    nd = len(data)
    nc_minus_1 = nd - 1
    idx = np.empty(nd, np.int64)
    idx[0] = 0

    group_counts = 0
    ng = 1

    for i, di in enumerate(data):
        group_counts += di

        if i == nc_minus_1:
            if group_counts < n:
                # if the last group does not have enough counts,
                # then combine the last two groups to ensure all
                # groups meet the counts requirement
                ng -= 1

            break

        if group_counts >= n:
            idx[ng] = i + 1

            group_counts = 0
            ng += 1

    idx = idx[:ng]
    if np.all(np.add.reduceat(data, idx) >= n):
        success = True
    else:
        success = False

    flag = np.full(nd, -1, dtype=int)
    flag[idx] = 1

    return flag, success


def group_pos(data: NDArray, error: NDArray, p: float) -> GroupResultType:
    """Group data by limiting the negative part of counts CDF is less than `p`.

    Parameters
    ----------
    data: array_like
        Counts array.
    error : array_like
        Uncertainty of data.
    p: float
        Maximum area of negative part of CDF in each channel after grouping.
        Must be less than 1.

    Returns
    -------
    flag : ndarray
        Grouping flag.
    success: bool
        Whether the scale is met for all grouped channels.

    """
    n_sigma = norm.isf(p)

    nd = len(data)
    nc_minus_1 = nd - 1
    idx = np.empty(nd, np.int64)
    idx[0] = 0

    grp_data = 0.0
    grp_var = 0.0
    ng = 1

    for i, (ci, ei) in enumerate(zip(data, error)):
        grp_data += ci
        grp_var += ei * ei
        x = grp_data - n_sigma * np.sqrt(grp_var)

        if i == nc_minus_1:
            if x < 0.0:
                # if the error of last group is not small enough,
                # then combine the last two groups to ensure all
                # groups meet the scale requirement
                ng -= 1

            break

        if x > 0.0:
            idx[ng] = i + 1

            grp_data = 0.0
            grp_var = 0.0
            ng += 1

    idx = idx[:ng]
    grp_data = np.add.reduceat(data, idx)
    grp_err = np.sqrt(np.add.reduceat(np.square(error), idx))
    if np.all(grp_data - n_sigma * grp_err > 0):
        success = True
    else:
        success = False

    flag = np.full(nd, -1, dtype=int)
    flag[idx] = 1

    return flag, success


def group_sig() -> GroupResultType:
    raise NotImplementedError


def group_const() -> GroupResultType:
    raise NotImplementedError


def group_opt() -> GroupResultType:
    raise NotImplementedError


def group_optmin() -> GroupResultType:
    raise NotImplementedError


def group_optsig() -> GroupResultType:
    raise NotImplementedError
