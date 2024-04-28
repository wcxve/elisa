"""Helper functions for plotting."""

from __future__ import annotations

from itertools import cycle

import numpy as np
import seaborn as sns

from elisa.util.typing import NumPyArray as NDArray


def get_colors(
    n: int, palette: str = 'husl'
) -> list[tuple[float, float, float]]:
    if len(colors := sns.color_palette(palette)) >= n:
        return colors[:n]
    else:
        return sns.color_palette(palette, n)


def get_markers(n: int) -> list[str]:
    markers_cycle = cycle(['s', 'o', 'D', '^', 'd', 'p', 'h', 'H', 'D'])
    return [marker for marker, _ in zip(markers_cycle, range(int(n)))]


def _clip(num):
    return int(np.clip(num, 0, 255))


def _scale_color(color: str, factor: float) -> str:
    color = str(color)
    factor = float(factor)

    if (not color.startswith('#')) or (len(color) != 7):
        raise ValueError('color must be in hex format "#RRGGBB"')

    if factor <= 0.0:
        raise ValueError('factor must be positive')

    r = _clip(int(color[1:3], 16) * factor)
    g = _clip(int(color[3:5], 16) * factor)
    b = _clip(int(color[5:], 16) * factor)

    return f'#{r:02x}{g:02x}{b:02x}'


def get_color_gradient(
    color: str, n: int, factor_min: float = 0.9, factor_max: float = 1.5
) -> list[str]:
    """Create a sequence of color gradient."""
    color = str(color)
    n = int(n)
    factor_min = float(factor_min)
    factor_max = float(factor_max)

    if (not color.startswith('#')) or (len(color) != 7):
        raise ValueError('color must be in hex format "#RRGGBB"')

    if factor_min <= 0.0:
        raise ValueError('factor_min must be positive')

    if factor_max <= 0.0:
        raise ValueError('factor_min must be positive')

    if factor_min >= factor_max:
        raise ValueError('factor_min must be less than factor_max')

    scales = np.geomspace(factor_max, factor_min, n)
    return [_scale_color(color, scale) for scale in scales]


def get_contour_colors(
    color: str,
    n: int,
    factor_min: float = 0.9,
    factor_max: float = 1.5,
    factor_f: float = 0.72,
) -> tuple:
    """Create two sets of colors for contour and contourf plots."""
    color = str(color)
    n = int(n)
    factor_min = float(factor_min)
    factor_max = float(factor_max)
    f = float(factor_f)

    contourf_colors = get_color_gradient(color, n, factor_min, factor_max)
    contour_colors = get_color_gradient(
        color, n, f * factor_min, f * factor_max
    )

    return contour_colors, contourf_colors


def gaussian_kernel_smooth(
    x: NDArray,
    y: NDArray,
    sigma: int | float,
    x_eval: NDArray | None = None,
    null_thresh: float = 0.683,
) -> NDArray:
    """Apply Gaussian kernel regression to data and then interpolate it.

    .. note::
        The regression here is also known as Nadaraya-Watson kernel regression
        [1]_. This helper function is adapted from [2]_.

    Parameters
    ----------
    x, y : ndarray
        Arrays of x- and y-coordinates of data. Must be 1d and have the same
        length.
    sigma : float
        Standard deviation of the Gaussian to apply to each data point. Larger
        values yield a smoother curve.
    x_eval : ndarray, optional
        Array of x-coordinates at which to evaluate the smoothed result. The
        default is `x`.
    null_thresh : float
        For evaluation points far from data points, the estimate will be
        based on very little data. If the total weight is below this threshold,
        return np.nan at this location. Zero means always return an estimate.
        The default of 0.6 corresponds to approximately one sigma away from
        the nearest datapoint.

    Returns
    -------
    smoothed : ndarray
        Smoothed data at `x_eval`.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Kernel_regression
    .. [2] https://stackoverflow.com/a/61394682
    """
    # The distance between every combination of x and x_eval
    # each row corresponds to a value in x_eval
    # each col corresponds to a value in x
    if x_eval is None:
        x_eval = x

    # from statsmodels.nonparametric.kernel_regression import KernelReg
    # return KernelReg(y, x, 'c', 'lc', [sigma]).fit(x_eval)[0]

    delta_x = x_eval[:, None] - x

    # Calculate weight of every value in delta_x using Gaussian
    # Maximum weight is 1.0 where delta_x is 0
    weights = np.exp(-0.5 * ((delta_x / sigma) ** 2))

    # Multiply each weight by every data point, and sum over data points
    smoothed = np.dot(weights, y)

    # Nullify the result when the total weight is below threshold
    # This happens at evaluation points far from any data
    # 1-sigma away from a data point has a weight of ~0.683
    nan_mask = weights.sum(1) < null_thresh
    smoothed[nan_mask] = np.nan

    # Normalize by dividing by the total weight at each evaluation point
    # Nullification above avoids divide by zero warnings here
    smoothed = smoothed / weights.sum(1)

    return smoothed
