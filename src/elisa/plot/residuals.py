"""Residual calculation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import xlogy
from scipy.stats import chi2, norm, poisson


def combine_residuals(
    r1: NDArray,
    r2: NDArray,
    dof: int | float | NDArray,
    sign: NDArray | None = None,
) -> NDArray:
    if isinstance(dof, (float, int)):
        dof = float(dof)
    else:
        dof = np.asarray(dof, float)
        if not (dof.shape == r1.shape == r2.shape):
            raise ValueError('shape of r1, r2, and dof should be the same')

    if sign is None:
        sign = 1.0

    r2_2d = r1 * r1 + r2 * r2
    if dof == 1.0:
        return sign * np.sqrt(r2_2d)

    mask = r2_2d <= dof
    r = np.empty(r1.shape)
    if isinstance(dof, float):
        r[mask] = norm.isf(0.5 - 0.5 * chi2.cdf(r2_2d[mask], dof))
        r[~mask] = norm.isf(0.5 * chi2.sf(r2_2d[~mask], dof))
    else:
        r[mask] = norm.isf(0.5 - 0.5 * chi2.cdf(r2_2d[mask], dof[mask]))
        r[~mask] = norm.isf(0.5 * chi2.sf(r2_2d[~mask], dof[~mask]))

    return sign * r


def pearson_residuals(
    observed: NDArray,
    expected: NDArray,
    std: NDArray | None = None,
) -> NDArray:
    """Calculate Pearson residuals.

    Parameters
    ----------
    observed : ndarray
        Observed counts.
    expected : ndarray
        Expected counts.
    std : ndarray, optional
        Standard deviation of the observed counts. If None, assumed to be the
        square root of the expected counts.

    Returns
    -------
    ndarray
        Pearson residuals.
    """
    if std is None:
        std = np.sqrt(expected)
    return (observed - expected) / std


def pit_poisson(
    k: NDArray,
    lam: NDArray,
    minus: bool = False,
) -> NDArray | tuple[NDArray, NDArray]:
    """Probability integral transform of poisson data fit.

    Parameters
    ----------
    k : ndarray
        The data values.
    lam : ndarray
        The model values.
    minus : bool, optional
        Whether to calculate the PIT of `k` - 1.

    Returns
    -------
    ndarray, or tuple of ndarray
        The probability integral transform values.
    """
    pit = poisson.cdf(k, lam)
    if not minus:
        return pit
    else:
        pit_minus = poisson.cdf(k - 1, lam)
        return pit_minus, pit


def quantile_residuals_poisson(
    k: NDArray,
    lam: NDArray,
    keep_sign: bool = False,
    random: bool = True,
    seed: int = 42,
) -> NDArray:
    """Normalized quantile residuals for fit of Poisson data.

    Parameters
    ----------
    k : ndarray
        Data value.
    lam : ndarray
        Model value.
    keep_sign : bool, optional
        Whether to keep sign of the residuals as ``sign(k - lam)``. The default
        is False.
    random : bool, optional
        Whether to add random noise so that residuals are normally distributed.
    seed : int, optional
        Random seed to use in adding noise. The default is 42.

    Returns
    -------
    ndarray
        The quantile residuals.

    Notes
    -----
    When `random` is True, the residuals are theoretically guaranteed to be
    normally distributed. If `keep_sign` is also True, then the normality of
    the residuals may not be preserved. The recommended use is only `random`
    or `keep_sign` being True.

    References
    ----------
    .. [1] doi:10.2307/1390802
    .. [2] doi:10.1371/journal.pone.0181790
    .. [3] doi:10.1111/j.1541-0420.2009.01191.x
    .. [4] doi:10.1186/s12874-020-01055-2
    """
    k_shape = k.shape
    lam_shape = lam.shape
    assert k_shape == lam_shape[-len(k_shape) :]

    if k_shape != lam_shape:
        k = np.full(lam_shape, k)

    mask1 = k < lam
    mask2 = ~mask1
    k1 = k[mask1]
    k2 = k[mask2]
    lam1 = lam[mask1]
    lam2 = lam[mask2]
    cdf = poisson.cdf(k1, lam1)
    sf = poisson.sf(k2, lam2)

    if random:  # randomized probability integral transform
        rng = np.random.default_rng(seed)
        cdf_right = cdf
        cdf_left = poisson.cdf(k1 - 1, lam1)  # when k < 0, poisson cdf is 0
        sf_right = sf
        sf_left = poisson.sf(k2 - 1, lam2)  # when k < 0, poisson sf is 1
        cdf = rng.uniform(cdf_left, cdf_right)
        sf = rng.uniform(sf_right, sf_left)

    r = np.empty(lam_shape)
    r[mask1] = np.where(
        cdf != 0.0,
        norm.ppf(cdf),
        (k1 - lam1) / np.sqrt(lam1),
    )
    r[mask2] = np.where(
        sf != 0.0,
        norm.isf(sf),
        (k2 - lam2) / np.sqrt(lam2),
    )

    if keep_sign:
        r = np.where(k >= lam, 1.0, -1.0) * np.abs(r)

    return r


def pit_poisson_poisson(
    k1: NDArray,
    k2: NDArray,
    lam1: NDArray,
    lam2: NDArray,
    ratio: float | NDArray,
    seed: int = 42,
    minus: bool = False,
    nsim: int = 10000,
) -> NDArray | tuple[NDArray, NDArray]:
    """Probability integral transform of two poisson data fit.

    Parameters
    ----------
    k1, k2 : ndarray
        The first and second data values.
    lam1, lam2 : ndarray
        The first and second model values.
    ratio : float or ndarray
        Background ratio.
    seed : int, optional
        Random seed to use in simulation. The default is 42.
    minus : bool, optional
        Whether to calculate the PIT of the next small quantity given `m1` and
        `m2`. The default is False.
    nsim : int, optional
        The number of simulations to generate. The default is 10000.

    Returns
    -------
    ndarray, or tuple of ndarray
        The probability integral transform values.
    """
    k_shape = np.shape(k1)
    lam_shape = np.shape(lam1)
    assert k_shape == np.shape(k2)
    assert lam_shape == np.shape(lam2)
    assert k_shape == lam_shape[-len(k_shape) :]

    obs = k1 - ratio * k2
    rng = np.random.default_rng(seed)
    sim = rng.poisson(lam1, size=(nsim,) + lam_shape) - ratio * rng.poisson(
        lam2, size=(nsim,) + lam_shape
    )
    pit = np.count_nonzero(sim <= obs, axis=0) / nsim
    if not minus:
        return pit
    else:
        obs_minus = np.maximum(k1 - 1.0 - ratio * k2, k1 - ratio * (k2 + 1.0))
        pit_minus = np.count_nonzero(sim <= obs_minus, axis=0) / nsim
        return pit_minus, pit


def quantile_residuals_poisson_poisson(
    k1: NDArray,
    k2: NDArray,
    lam1: NDArray,
    lam2: NDArray,
    ratio: float | NDArray,
    random: bool = True,
    seed: int = 42,
    nsim: int = 10000,
) -> tuple[NDArray, NDArray, NDArray]:
    """Normalized quantile residuals for joint fit of two Poisson data.

    .. note::
        The calculation is based on Monte Carlo simulation. It is also possible
        to calculate the quantile residuals by inverting the Cornish-Fisher
        expansion. See the following link for more details:

            * https://stats.stackexchange.com/a/73070
            * https://www.value-at-risk.net/the-cornish-fisher-expansion/
            * https://faculty.washington.edu/ezivot/econ589/ssrn-id1997178.pdf

    Parameters
    ----------
    k1, k2 : ndarray
        The first and second data values.
    lam1, lam2 : ndarray
        The first and second model values.
    ratio : float or ndarray
        Background ratio.
    random : bool, optional
        Whether to add random noise so that residuals are normally distributed.
    seed : int, optional
        Random seed to use in adding noise and simulation. The default is 42.
    nsim : int, optional
        The number of simulations to generate. The default is 10000.

    Returns
    -------
    tuple of ndarray
        The quantile residuals, and flags to mark if the residuals are lower or
        upper limit.
    """
    res = pit_poisson_poisson(k1, k2, lam1, lam2, ratio, seed, random, nsim)
    if random:
        pit = np.random.default_rng(seed).uniform(*res)
    else:
        pit = res
    r = norm.ppf(pit)
    lower_mask = pit == 0.0
    upper_mask = pit == 1.0
    r[lower_mask] = norm.ppf(1.0 / nsim)
    r[upper_mask] = norm.ppf(1.0 - 1.0 / nsim)
    lower = np.full(lam1.shape, False)
    upper = np.full(lam1.shape, False)
    lower[lower_mask] = True
    upper[upper_mask] = True
    return r, lower, upper


def pit_poisson_normal(
    k: NDArray,
    lam: NDArray,
    v: NDArray,
    mu: NDArray,
    sigma: NDArray,
    ratio: float | NDArray,
    seed: int = 42,
    nsim: int = 10000,
) -> NDArray:
    """Probability integral transform of poisson and normal data fit.

    Parameters
    ----------
    k : ndarray
        The Poisson data value.
    lam : ndarray
        The Poisson model value.
    v : ndarray
        The normal data value.
    mu : ndarray
        The normal model value.
    sigma : ndarray
        The normal model sigma.
    ratio : float or ndarray
        Background ratio.
    seed : int, optional
        Random seed to use in simulation. The default is 42.
    nsim : int, optional
        The number of simulations to generate. The default is 10000.

    Returns
    -------
    ndarray, or tuple of ndarray
        The probability integral transform values.
    """
    k_shape = np.shape(k)
    lam_shape = np.shape(lam)
    assert k_shape == np.shape(v)
    assert lam_shape == np.shape(mu)
    assert k_shape == np.shape(sigma) == lam_shape[-len(k_shape) :]

    obs = k - ratio * v
    rng = np.random.default_rng(seed)
    sim = rng.poisson(lam, size=(nsim,) + lam_shape) - ratio * rng.normal(
        mu, sigma, size=(nsim,) + lam_shape
    )
    pit = np.count_nonzero(sim <= obs, axis=0) / nsim
    return pit


def quantile_residuals_poisson_normal(
    k: NDArray,
    lam: NDArray,
    v: NDArray,
    mu: NDArray,
    sigma: NDArray,
    ratio: float | NDArray,
    seed: int = 42,
    nsim: int = 10000,
) -> tuple[NDArray, NDArray, NDArray]:
    """Normalized quantile residuals for joint fit of Poisson and normal data.

    .. note::
        The calculation is based on Monte Carlo simulation. It is also possible
        to calculate the quantile residuals by inverting the Cornish-Fisher
        expansion. See the following link for more details:

            * https://stats.stackexchange.com/a/73070
            * https://www.value-at-risk.net/the-cornish-fisher-expansion/
            * https://faculty.washington.edu/ezivot/econ589/ssrn-id1997178.pdf

    Parameters
    ----------
    k : ndarray
        The Poisson data value.
    lam : ndarray
        The Poisson model value.
    v : ndarray
        The normal data value.
    mu : ndarray
        The normal model value.
    sigma : ndarray
        The normal model sigma.
    ratio : float or ndarray
        Background ratio.
    seed : int, optional
        Random seed to use in simulation. The default is 42.
    nsim : int, optional
        The number of simulations to generate. The default is 10000.

    Returns
    -------
    tuple of ndarray
        The quantile residuals, and flags to mark if the residuals are lower or
        upper limit.
    """
    pit = pit_poisson_normal(k, lam, v, mu, sigma, ratio, seed, nsim)
    r = norm.ppf(pit)
    lower_mask = pit == 0.0
    upper_mask = pit == 1.0
    r[lower_mask] = norm.ppf(1.0 / nsim)
    r[upper_mask] = norm.ppf(1.0 - 1.0 / nsim)
    lower = np.full(lam.shape, False)
    upper = np.full(lam.shape, False)
    lower[lower_mask] = True
    upper[upper_mask] = True
    return r, lower, upper


def deviance_residuals_poisson(k: NDArray, lam: NDArray) -> NDArray:
    """Deviance residuals for fit of Poisson data.

    Parameters
    ----------
    k : ndarray
        Data value.
    lam : ndarray
        Model value.

    Returns
    -------
    ndarray
        The deviance residuals.

    References
    ----------
    .. [1] McCullagh P, Nelder JA. Generalized Linear Models, 2nd ed., pp. 39
    .. [2] Spiegelhalter et al., https://doi.org/10.1111/1467-9868.00353
    """
    deviance = 2.0 * (xlogy(k, k / lam) - (k - lam))
    return np.where(k >= lam, 1.0, -1.0) * np.sqrt(deviance)


def deviance_residuals_poisson_poisson(
    k1: NDArray,
    k2: NDArray,
    lam1: NDArray,
    lam2: NDArray,
    sign: NDArray | None = None,
    dof: int | float | NDArray = 2.0,
) -> NDArray:
    """Deviance residuals [1]_ [2]_ for joint fit of two Poisson data.

    Parameters
    ----------
    k1, k2 : ndarray
        The first and second data value.
    lam1, lam2 : ndarray
        The first and second model value.
    sign : ndarray, optional
        The sign of output residuals. The default is None.
    dof : int, float, or NDArray, optional
        The degree of freedom of each data pair. The default is 2.

    Returns
    -------
    ndarray
        The deviance residuals.

    References
    ----------
    .. [1] McCullagh P, Nelder JA. Generalized Linear Models, 2nd ed., pp. 39
    .. [2] Spiegelhalter et al., https://doi.org/10.1111/1467-9868.00353
    """
    d = 2.0 * (xlogy(k1, k1 / lam1) - (k1 - lam1))
    d += 2.0 * (xlogy(k2, k2 / lam2) - (k2 - lam2))

    if np.all(dof == 1.0):
        r = np.sqrt(d)
    else:
        if isinstance(dof, (float, int)):
            dof = np.full(k1.shape, dof)

        mask = d <= dof
        r = np.empty(k1.shape)
        r[mask] = norm.ppf(chi2.cdf(d[mask], dof[mask]))
        r[~mask] = norm.isf(chi2.sf(d[~mask], dof[~mask]))

    if sign is not None:
        r = sign * np.abs(r)

    return r


def deviance_residuals_poisson_normal(
    k: NDArray,
    lam: NDArray,
    v: NDArray,
    mu: NDArray,
    sigma: NDArray,
    sign: NDArray | None = None,
    dof: int | float | NDArray = 2.0,
) -> NDArray:
    """Deviance residuals [1]_ [2]_ for joint fit of Poisson and normal data.

    Parameters
    ----------
    k : ndarray
        The first data value.
    lam : ndarray
        The first model value.
    v : ndarray
        Data value.
    mu : ndarray
        Model value.
    sigma : ndarray
        Model sigma.
    sign : ndarray, optional
        The sign of output residuals. The default is None.
    dof : int, float, or NDArray, optional
        The degree of freedom of each data pair. The default is 2.

    Returns
    -------
    ndarray
        The deviance residuals.

    References
    ----------
    .. [1] McCullagh P, Nelder JA. Generalized Linear Models, 2nd ed., pp. 39
    .. [2] Spiegelhalter et al., https://doi.org/10.1111/1467-9868.00353
    """
    d = 2.0 * (xlogy(k, k / lam) - (k - lam))
    v = (v - mu) / sigma
    d += v * v

    if np.all(dof == 1.0):
        r = np.sqrt(d)
    else:
        if isinstance(dof, (float, int)):
            dof = np.full(k.shape, dof)

        mask = d <= dof
        r = np.empty(k.shape)
        r[mask] = norm.ppf(chi2.cdf(d[mask], dof[mask]))
        r[~mask] = norm.isf(chi2.sf(d[~mask], dof[~mask]))

    if sign is not None:
        r = sign * np.abs(r)

    return r
