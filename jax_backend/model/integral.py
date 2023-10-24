"""Numerical method for additive models integral, whose closed form expressions
do not exist.
"""

from functools import wraps
from typing import Callable, Union

__all__ = ['integral']


def trapezoid(func: Callable) -> Callable:
    """Trapezoid method."""

    @wraps(func)
    def _(grid, *args, **kwargs):
        dx = grid[1:] - grid[:-1]
        f_grid = func(grid, *args, **kwargs)
        return (f_grid[:-1] + f_grid[1:]) / 2.0 * dx

    return _


def simpson(func: Callable) -> Callable:
    """Simpson's 1/3 rule."""

    @wraps(func)
    def _(grid, *args, **kwargs):
        dx = grid[1:] - grid[:-1]
        x_mid = (grid[:-1] + grid[1:]) / 2.0
        f_grid = func(grid, *args, **kwargs)
        f_mid = func(x_mid, *args, **kwargs)
        return dx / 6.0 * (f_grid[:-1] + 4.0 * f_mid + f_grid[1:])

    return _


METHODS: dict[Union[str, None]] = {
    'default': trapezoid,
    'trapezoid': trapezoid,
    'simpson': simpson
}


def integral(f: Callable, method: str) -> Callable:
    """Wrap the integrand with specified numerical integral method."""

    if method not in METHODS:
        methods = '"' + '", "'.join(METHODS.keys()) + '"'
        raise ValueError(
            f'available numerical integral options are: {methods}, but got '
            f'"{method}"'
        )

    return METHODS[method](f)
