from __future__ import annotations

import numpy as np


def counts():
    ...


def error():
    ...


def significance():
    ...


def constant():
    ...


def optimal():
    ...


def optimal_counts():
    ...


def optimal_significance():
    ...


def _counts_grouping_idx(counts, group_scale):
    n = len(counts)
    n_minus_1 = n - 1
    grouping = np.empty(n, np.int64)
    grouping[0] = 0

    group_counts = 0
    ngroup = 1

    for i, ci in enumerate(counts):
        group_counts += ci

        if i == n_minus_1:
            if group_counts < group_scale:
                # if the last group does not have enough counts,
                # then combine the last two groups to ensure all
                # groups meet the scale requirement
                ngroup -= 1

            break

        if group_counts >= group_scale:
            grouping[ngroup] = i + 1

            group_counts = 0
            ngroup += 1

    return grouping[:ngroup]


def _counts_grouping_flag(counts, group_scale):
    idx = _counts_grouping_idx(counts, group_scale)
    flag = np.zeros(len(counts), dtype=int)
    flag[idx] = 1

    return flag


def _error_grouping_idx(counts, error, n_sigma):
    n = len(counts)
    n_minus_1 = n - 1
    grouping = np.empty(n, np.int64)
    grouping[0] = 0

    grp_counts = 0.0
    grp_error2 = 0.0
    ngroup = 1

    for i, (ci, ei) in enumerate(zip(counts, error)):
        grp_counts += ci
        grp_error2 += ei * ei
        x = grp_counts - n_sigma * np.sqrt(grp_error2)

        if i == n_minus_1:
            if x < 0.0:
                # if the error of last group is not small enough,
                # then combine the last two groups to ensure all
                # groups meet the scale requirement
                ngroup -= 1

            break

        if x > 0.0:
            grouping[ngroup] = i + 1

            grp_counts = 0.0
            grp_error2 = 0.0
            ngroup += 1

    return grouping[:ngroup]


def _error_grouping_flag(counts, error, n_sigma):
    idx = _error_grouping_idx(counts, error, n_sigma)
    flag = np.zeros(len(counts), dtype=int)
    flag[idx] = 1

    return flag
