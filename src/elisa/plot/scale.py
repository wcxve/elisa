from __future__ import annotations

import numpy as np
from matplotlib.axis import Axis
from matplotlib.scale import ScaleBase
from matplotlib.ticker import (
    AutoLocator,
    Locator,
    LogFormatterSciNotation,
    LogLocator,
    NullLocator,
    ScalarFormatter,
)
from matplotlib.transforms import Transform


class _DummyAxis:
    __name__ = 'dummy'

    def __init__(
        self,
        axis: Axis,
        base: float,
        lin_thresh: float,
        lin_scale: float,
        is_log: bool,
    ):
        base = float(base)
        lin_thresh = float(lin_thresh)
        lin_scale = float(lin_scale)
        is_log = bool(is_log)

        self._axis = axis
        self._base = base
        self._log_base = np.log(base)
        self._lin_thresh = lin_thresh
        self._lin_scale_adj = lin_scale / (1.0 - 1.0 / base)
        self._is_log = is_log

    def get_view_interval(self) -> tuple[float, float]:
        lin_thresh = self._lin_thresh
        vmin, vmax = self._axis.get_view_interval()
        if vmin < lin_thresh < vmax:
            return (lin_thresh, vmax) if self._is_log else (vmin, lin_thresh)
        else:
            return vmin, vmax

    def get_minpos(self) -> float:
        return self._axis.get_minpos()

    def get_data_interval(self) -> tuple[float, float]:
        lin_thresh = self._lin_thresh
        vmin, vmax = self._axis.get_data_interval()
        if vmin < lin_thresh < vmax:
            return (lin_thresh, vmax) if self._is_log else (vmin, lin_thresh)
        else:
            return vmin, vmax

    def get_tick_space(self) -> int:
        transformed_lin_thresh = self._lin_thresh * self._lin_scale_adj
        view_interval = np.asarray(self._axis.get_view_interval())
        vmin, vmax = self.transform_non_affine(view_interval)
        if vmin < transformed_lin_thresh < vmax:
            factor = (transformed_lin_thresh - vmin) / (vmax - vmin)
            if self._is_log:
                factor = 1.0 - factor
        else:
            factor = 1.0
        return round(self._axis.get_tick_space() * factor)

    def transform_non_affine(self, values: np.ndarray) -> np.ndarray:
        mask = values > self._lin_thresh
        out = np.empty_like(values)
        with np.errstate(divide='ignore', invalid='ignore'):
            out[mask] = self._lin_thresh * (
                self._lin_scale_adj
                + np.log(values[mask] / self._lin_thresh) / self._log_base
            )
        out[~mask] = values[~mask] * self._lin_scale_adj
        return out


class LinLogTransform(Transform):
    input_dims = output_dims = 1

    def __init__(self, base: float, lin_thresh: float, lin_scale: float):
        base = float(base)
        lin_thresh = float(lin_thresh)
        lin_scale = float(lin_scale)

        if base <= 1.0:
            raise ValueError('`base` must be larger than 1')
        if lin_thresh <= 0.0:
            raise ValueError('`lin_thresh` must be positive')
        if lin_scale <= 0.0:
            raise ValueError('`lin_scale` must be positive')

        super().__init__()

        self.base = base
        self._log_base = np.log(base)
        self.lin_thresh = lin_thresh
        self.lin_scale = lin_scale
        self._lin_scale_adj = lin_scale / (1.0 - 1.0 / base)

    def transform_non_affine(self, values: np.ndarray):
        mask = values > self.lin_thresh
        out = np.empty_like(values)
        with np.errstate(divide='ignore', invalid='ignore'):
            out[mask] = self.lin_thresh * (
                self._lin_scale_adj
                + np.log(values[mask] / self.lin_thresh) / self._log_base
            )
        out[~mask] = values[~mask] * self._lin_scale_adj
        return out

    def inverted(self):
        return InvertedLinLogTransform(
            self.base, self.lin_thresh, self.lin_scale
        )


class InvertedLinLogTransform(Transform):
    input_dims = output_dims = 1

    def __init__(self, base: float, lin_thresh: float, lin_scale: float):
        base = float(base)
        lin_thresh = float(lin_thresh)
        lin_scale = float(lin_scale)

        if base <= 1.0:
            raise ValueError('`base` must be larger than 1')
        if lin_thresh <= 0.0:
            raise ValueError('`lin_thresh` must be positive')
        if lin_scale <= 0.0:
            raise ValueError('`lin_scale` must be positive')

        super().__init__()

        self.base = base
        self.lin_thresh = lin_thresh
        self.lin_scale = lin_scale
        self._lin_scale_adj = lin_scale / (1.0 - 1.0 / self.base)
        self.inv_lin_thresh = lin_thresh * self._lin_scale_adj

    def transform_non_affine(self, values: np.ndarray):
        mask = values > self.inv_lin_thresh
        out = np.empty_like(values)
        with np.errstate(divide='ignore', invalid='ignore'):
            out[mask] = self.lin_thresh * (
                np.power(
                    self.base,
                    values[mask] / self.lin_thresh - self._lin_scale_adj,
                )
            )
        out[~mask] = values[~mask] / self._lin_scale_adj
        return out

    def inverted(self):
        return LinLogTransform(self.base, self.lin_thresh, self.lin_scale)


class _LinLogFormatter(LogFormatterSciNotation):
    """Formatter for LinLogScale axes ticks."""

    def __init__(
        self,
        base: float,
        lin_thresh: float,
        lin_scale: float,
        label_only_base: bool = False,
    ):
        base = float(base)
        lin_thresh = float(lin_thresh)
        lin_scale = float(lin_scale)

        if base <= 1.0:
            raise ValueError('`base` must be larger than 1')
        if lin_thresh <= 0.0:
            raise ValueError('`lin_thresh` must be positive')
        if lin_scale <= 0.0:
            raise ValueError('`lin_scale` must be positive')

        self.__base = base
        self.__lin_thresh = lin_thresh
        self.__lin_scale = lin_scale
        self._formatter_lin = ScalarFormatter()
        super().__init__(
            base=base,
            labelOnlyBase=label_only_base,
            linthresh=lin_thresh,
        )

    def __call__(self, x: float, pos: int | None = None):
        if x >= self.__lin_thresh:
            return super().__call__(x, pos)
        else:
            s = self._formatter_lin(x, pos)
            try:
                if float(s) == 0.0:
                    s = '0'
            except ValueError:
                pass
            return s

    def set_axis(self, axis: Axis):
        self._formatter_lin.set_axis(
            _DummyAxis(
                axis, self.__base, self.__lin_thresh, self.__lin_scale, False
            )
        )
        super().set_axis(
            _DummyAxis(
                axis, self.__base, self.__lin_thresh, self.__lin_scale, True
            )
        )

    def create_dummy_axis(self, **kwargs):
        self._formatter_lin.create_dummy_axis(**kwargs)
        super().create_dummy_axis(**kwargs)

    def set_locs(self, locs=None):
        """Set the locations of the ticks."""
        super().set_locs(locs)
        lin_locs = [i for i in locs if i <= self.__lin_thresh]
        self._formatter_lin.set_locs(lin_locs)


class LinLogLocator(Locator):
    """Determine the tick locations for LinLogScale axes."""

    def __init__(
        self,
        lin_thresh: float,
        lin_scale: float,
        base: float = 10.0,
        subs=(1.0,),
        numticks=None,
        is_minor: bool = False,
    ):
        base = float(base)
        lin_thresh = float(lin_thresh)
        lin_scale = float(lin_scale)
        is_minor = bool(is_minor)

        if base <= 1.0:
            raise ValueError('`base` must be larger than 1')
        if lin_thresh <= 0.0:
            raise ValueError('`lin_thresh` must be positive')
        if lin_scale <= 0.0:
            raise ValueError('`lin_scale` must be positive')

        self._base = base
        self._log_base = np.log(base)
        self._lin_thresh = lin_thresh
        self._lin_scale = lin_scale
        self._lin_scale_adj = lin_scale / (1.0 - self._base**-1)
        self._locator_log = LogLocator(base, subs, numticks=numticks)
        self._is_minor = is_minor

        # no minor ticks at linear range
        if is_minor:
            self._locator_lin = NullLocator()
        else:
            self._locator_lin = AutoLocator()

    def set_axis(self, axis: Axis):
        log_dummy = _DummyAxis(
            axis, self._base, self._lin_thresh, self._lin_scale, True
        )
        self._locator_log.set_axis(log_dummy)
        lin_dummy = _DummyAxis(
            axis, self._base, self._lin_thresh, self._lin_scale, False
        )
        self._locator_lin.set_axis(lin_dummy)
        super().set_axis(axis)

    def create_dummy_axis(self, **kwargs):
        self._locator_log.create_dummy_axis(**kwargs)
        self._locator_lin.create_dummy_axis(**kwargs)
        super().create_dummy_axis(**kwargs)

    def set_params(
        self,
        base=None,
        subs=None,
        numticks=None,
    ):
        """Set parameters for log locator."""
        self._locator_log.set_params(base, subs, numticks=numticks)

    def __call__(self):
        """Return the locations of the ticks."""
        # Note, these are untransformed coordinates
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin: float, vmax: float):
        if vmin > vmax:
            vmin, vmax = vmax, vmin

        lin_thresh = self._lin_thresh

        ticks = []
        # Lower linear range is present
        if vmin < lin_thresh:
            lin_upper_lim = min(lin_thresh, vmax)
            ticks_lin = self._locator_lin()
            mask = np.less(ticks_lin, lin_upper_lim)
            if mask.any():
                ticks.append(ticks_lin[mask])

        # Upper log range is present
        if vmax > lin_thresh:
            log_lower_lim = max(vmin, lin_thresh)
            ticks_log = self._locator_log()
            mask = np.greater(ticks_log, log_lower_lim)
            if mask.any():
                ticks.append(ticks_log[mask])

        if ticks:
            ticks = np.unique(np.hstack(ticks))

        return self.raise_if_exceeds(ticks)

    def view_limits(self, vmin, vmax):
        """Try to choose the view limits intelligently."""
        return self._locator_lin.view_limits(vmin, vmax)

    def transform_non_affine(self, values: np.ndarray) -> np.ndarray:
        mask = values > self._lin_thresh
        out = np.empty_like(values)
        with np.errstate(divide='ignore', invalid='ignore'):
            out[mask] = self._lin_thresh * (
                self._lin_scale_adj
                + np.log(values[mask] / self._lin_thresh) / self._log_base
            )
        out[~mask] = values[~mask] * self._lin_scale_adj
        return out


class LinLogScale(ScaleBase):
    """The linear-logarithmic scale is logarithmic above positive `lin_thresh`
    and linear otherwise.

    Parameters
    ----------
    base : float, default: 10
        The base of the logarithm.
    lin_thresh : float, default: 1.0
        Defines the range ``(lin_thresh, +inf)``, within which the plot is
        logarithmic.
    lin_scale : float, optional
        This allows the linear range ``(-inf, lin_thresh)`` to be stretched
        relative to the logarithmic range. Its value is the number of decades
        to use for the linear range. For example, when *lin_scale* == 1.0, the
        space used for ``[0, lin_thresh]`` range will be equal to one decade
        in the logarithmic range.
    subs : sequence of int
        Where to place the subticks between each major tick.
        For example, in a log10 scale: ``[2, 3, 4, 5, 6, 7, 8, 9]`` will place
        8 logarithmically spaced minor ticks between each major tick.
    """

    name = 'linlog'

    def __init__(
        self,
        axis,
        *,
        base: float = 10.0,
        lin_thresh: float = 1.0,
        lin_scale=None,
        subs=None,
    ):
        base = float(base)
        lin_thresh = float(lin_thresh)

        if lin_scale is None:
            lin_scale = 1.0 / np.log(base)
        else:
            lin_scale = float(lin_scale)

        if base <= 1.0:
            raise ValueError('`base` must be larger than 1')
        if lin_thresh <= 0.0:
            raise ValueError('`lin_thresh` must be positive')
        if lin_scale <= 0.0:
            raise ValueError('`lin_scale` must be positive')

        self._transform = LinLogTransform(base, lin_thresh, lin_scale)
        self.subs = subs
        super().__init__(axis)

    base = property(lambda self: self._transform.base)
    lin_thresh = property(lambda self: self._transform.lin_thresh)
    lin_scale = property(lambda self: self._transform.lin_scale)

    def set_default_locators_and_formatters(self, axis):
        lin_thresh = self.lin_thresh
        lin_scale = self.lin_scale
        base = self.base

        major_locator = LinLogLocator(lin_thresh, lin_scale, base)
        axis.set_major_locator(major_locator)
        axis.set_major_formatter(_LinLogFormatter(base, lin_thresh, lin_scale))

        minor_locator = LinLogLocator(
            lin_thresh, lin_scale, base, 'auto', is_minor=True
        )
        axis.set_minor_locator(minor_locator)
        axis.set_minor_formatter(
            _LinLogFormatter(
                base,
                lin_thresh,
                lin_scale,
                label_only_base=(self.subs is not None),
            )
        )

    def get_transform(self):
        """Return the `.LinLogTransform` associated with this scale."""
        return self._transform


def get_scale(
    base: float, thresh: float, vmin: float, vmax: float, f: float
) -> float:
    """Get scale of `.LinLogScale` so that linear part takes up `f` of axis."""
    base = float(base)
    thresh = float(thresh)
    vmin = float(vmin)
    vmax = float(vmax)

    assert base > 1.0
    assert vmin < thresh < vmax
    assert 0.0 < f < 1.0

    numerator = (base - 1.0) * f * thresh * np.log(vmax / thresh)
    denominator = base * (f - 1.0) * (thresh - vmin) * np.log(base)
    scale = -numerator / denominator
    return scale
