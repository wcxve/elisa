"""Corner plot."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from corner import corner


def plot_corner(data, axes_scale='linear', labels=None, color=None, weights=None):
    """log_scale : bool, whether to plot vars in log which is log uniform"""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['text.usetex'] = True
    levels = [
        [0.683, 0.954, 0.997],  # 1/2/3-sigma of 1d normal
        [0.393, 0.865, 0.989],  # 1/2/3-sigma of 2d normal
        [0.683, 0.9],           # 1-sigma and 90% of 2d normal
        [0.393, 0.683, 0.9]     # 1-sigma, 68.3% and 90% of 2d normal
    ][-1]

    # def to_hex(c):
    #     rgb_hex = ''.join(f'{round(i * 255):02x}' for i in c[:3])
    #     return f'#{rgb_hex}'
    # cmap = plt.get_cmap('Blues')
    # colors2 = [cmap(i*0.8 + 0.1) for i in levels]
    # colors1 = [scale_color(to_hex(c), 0.95) for c in colors1]
    if color is None:
        color = '#2f68c4'
    else:
        color = str(color)
    colors1, colors2 = _contour_colors(color, len(levels), 0.8, 2.0)

    fig = corner(
        data,
        bins=25,
        axes_scale=axes_scale,
        weights=weights,
        color=color,
        hist_bin_factor=2,
        labels=labels,
        show_titles=True,
        quantiles=[0.15865, 0.5, 0.84135],
        use_math_text=True,
        labelpad=-0.08,
        # kwargs for corner.hist2d
        levels=levels,
        plot_datapoints=True,
        plot_density=False,
        plot_contours=True,
        fill_contours=True,
        no_fill_contours=True,
        contour_kwargs={'colors': colors1},
        contourf_kwargs={'colors': ['white'] + colors2, 'alpha': 0.75},
        data_kwargs={'color': colors2[0], 'alpha': 0.75}
    )


def _scale_color(color: str, factor: float) -> str:
    color = str(color)
    factor = float(factor)

    if (not color.startswith('#')) or (len(color) != 7):
        raise ValueError('color format must be "#RRGGBB"')

    if factor <= 0.0:
        raise ValueError('factor must be positive')

    def clip(num):
        return int(np.clip(num, 0, 255))

    r = clip(int(color[1:3], 16) * factor)
    g = clip(int(color[3:5], 16) * factor)
    b = clip(int(color[5:], 16) * factor)

    return f'#{r:02x}{g:02x}{b:02x}'


def _gradient_colors(
    color: str,
    n: int,
    factor_min: float = 0.9,
    factor_max: float = 1.5
) -> list[str]:
    color = str(color)
    n = int(n)
    factor_min = float(factor_min)
    factor_max = float(factor_max)

    if (not color.startswith('#')) or (len(color) != 7):
        raise ValueError('color format must be "#RRGGBB"')

    if factor_min <= 0.0:
        raise ValueError('factor_min must be positive')

    if factor_max <= 0.0:
        raise ValueError('factor_min must be positive')

    if factor_min >= factor_max:
        raise ValueError('factor_min must be less than factor_max')

    scales = np.geomspace(factor_max, factor_min, n)
    return [_scale_color(color, scale) for scale in scales]


def _contour_colors(
    color: str,
    n: int,
    factor_min: float = 0.9,
    factor_max: float = 1.5,
    factor_f: float = 0.72
) -> tuple:
    color = str(color)
    n = int(n)
    factor_min = float(factor_min)
    factor_max = float(factor_max)
    f = float(factor_f)

    contourf_colors = _gradient_colors(color, n, factor_min, factor_max)
    contour_colors = _gradient_colors(color, n, f*factor_min, f*factor_max)

    return contour_colors, contourf_colors
