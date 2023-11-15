"""Numerical method for additive continnum integral.

Various numerical integral method is defined here, calculating integral whose
closed form expressions do not exist.

"""

import inspect

from functools import wraps
from typing import Callable

__all__ = ['integral', 'list_methods']


_trapezoid = """
@wraps(func)
def _(egrid, {def_str}):
    de = egrid[1:] - egrid[:-1]
    f_grid = func(egrid, {call_str})
    return (f_grid[:-1] + f_grid[1:]) / 2.0 * de
"""

_simpson = """
@wraps(func)
def _(egrid, {def_str}):
    de = egrid[1:] - egrid[:-1]
    e_mid = (egrid[:-1] + egrid[1:]) / 2.0
    f_grid = func(egrid, {call_str})
    f_mid = func(e_mid, {call_str})
    return de / 6.0 * (f_grid[:-1] + 4.0 * f_mid + f_grid[1:])
"""


_template: dict = {
    'default': _trapezoid,
    'trapezoid': _trapezoid,
    'simpson': _simpson
}


def integral(f: Callable, method: str) -> Callable:
    """Wrap the integrand with specified numerical integral method."""

    if method not in _template:
        methods = '"' + '", "'.join(list_methods()) + '"'
        raise ValueError(
            f'available numerical integral options are: {methods}, but got '
            f'"{method}"'
        )

    params = list(inspect.signature(f).parameters.keys())[1:]
    def_str = ', '.join(params)
    call_str = ', '.join(map(lambda s: f'{s}={s}', params))
    tmp = {'wraps': wraps, 'func': f}
    exec(_template[method].format(def_str=def_str, call_str=call_str), tmp)

    return tmp['_']


def list_methods() -> tuple:
    """List available numerical integral options."""
    return tuple(_template.keys())
