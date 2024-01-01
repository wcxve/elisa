"""Helper functions for inference module."""
from __future__ import annotations

from typing import Callable, Optional, Sequence, TypeVar

import re
from functools import reduce

from jax import lax
from jax.experimental import host_callback
from prettytable import PrettyTable
from tqdm.auto import tqdm

T = TypeVar('T')


def order_composite(sample: dict, composite: dict) -> dict:
    """Return ordered composite parameters."""
    ordered = {}

    remains = list(composite.items())
    while remains:
        site = sample | ordered
        i = remains.pop(0)
        name, (arg_names, func) = i
        if all(arg_name in site for arg_name in arg_names):
            ordered[name] = (arg_names, func)
        else:
            remains.append(i)

    return ordered


def make_pretty_table(fields: Sequence[str], rows: Sequence) -> PrettyTable:
    """Creates an instance of :class:`prettytable.PrettyTable`.

    Parameters
    ----------
    fields : sequence of str
        The names of fields.
    rows : sequence
        The sequence of data corresponding to the `fields`.

    Returns
    -------
    table : PrettyTable
        The pretty table.

    """
    table = PrettyTable(
        fields,
        align='c',
        hrules=1,  # 1 for all, 0 for frame
        vrules=1,
        padding_width=1,
        vertical_char='│',
        horizontal_char='─',
        junction_char='┼',
        top_junction_char='┬',
        bottom_junction_char='┴',
        right_junction_char='┤',
        left_junction_char='├',
        top_right_junction_char='┐',
        top_left_junction_char='┌',
        bottom_right_junction_char='┘',
        bottom_left_junction_char='└'
    )
    table.add_rows(rows)
    return table


def replace_string(mapping: dict[str, str], value: T) -> T:
    """Replace all string in `value` appeared in `mapping`.

    Parameters
    ----------
    mapping : dict
        Mapping of str value to be replaced and replacement.
    value : iterable or mapping
        Whose str value needs to be replaced.

    Returns
    -------
    replaced : iterable or mapping
        Value of `value` replaced with `mapping`.

    """
    mapping = mapping.items()

    def replace_with_mapping(s: str):
        """Replace all k in s with v, as in mapping."""
        return reduce(lambda x, kv: x.replace(*kv), mapping, s)

    def replace_dict(d: dict):
        """Replace key and value of a dict."""
        return {replace(k): replace(v) for k, v in d.items()}

    def replace_iterable(it: tuple | list):
        """Replace element of a dict."""
        return type(it)(map(replace, it))

    def replace(v):
        """Main replace function."""
        if isinstance(v, dict):
            return replace_dict(v)
        elif isinstance(v, (list, tuple)):
            return replace_iterable(v)
        elif isinstance(v, str):
            return replace_with_mapping(v)
        else:
            return v

    replaced = replace(value)

    return replaced


def progress_bar_factory(
    neval: int,
    ncores: int,
    init_str: Optional[str] = None,
    run_str: Optional[str] = None
) -> Callable:
    """Add a progress bar to fori_loop kernel.
    Adapt from: https://www.jeremiecoullon.com/2021/01/29/jax_progress_bar/
    """
    neval = int(neval)
    ncores = int(ncores)
    neval_single = neval // ncores

    if neval % ncores != 0:
        raise ValueError('neval must be multiple of ncores')

    if init_str is None:
        init_str = 'Compiling... '
    else:
        init_str = str(init_str)

    if run_str is None:
        run_str = 'Running'
    else:
        run_str = str(run_str)

    process_re = re.compile(r"\d+$")

    if neval > 20:
        print_rate = int(neval_single / 20)
    else:
        print_rate = 1

    remainder = neval_single % print_rate
    finished = [False] * 4
    bar = tqdm(range(neval))
    bar.set_description(init_str, refresh=True)

    def _update_tqdm(arg, transform):
        bar.set_description(run_str, refresh=False)
        bar.update(arg)

    def _close_tqdm(arg, transform, device):
        match = process_re.search(str(device))
        assert match
        i = int(match.group())
        bar.update(arg)
        finished[i] = True
        if all(finished):
            bar.close()

    def _update_progress_bar(iter_num):
        _ = lax.cond(
            iter_num == 1,
            lambda _: host_callback.id_tap(
                _update_tqdm, 0, result=iter_num
            ),
            lambda _: iter_num,
            operand=None,
        )
        _ = lax.cond(
            iter_num % print_rate == 0,
            lambda _: host_callback.id_tap(
                _update_tqdm, print_rate, result=iter_num
            ),
            lambda _: iter_num,
            operand=None,
        )
        _ = lax.cond(
            iter_num == neval_single,
            lambda _: host_callback.id_tap(
                _close_tqdm, remainder, result=iter_num, tap_with_device=True
            ),
            lambda _: iter_num,
            operand=None,
        )

    def progress_bar_fori_loop(fn):
        """Decorator that adds a progress bar to `body_fun` used in
        `lax.fori_loop`.
        Note that `body_fun` must be looping over a tuple who's first element
        is `np.arange(num_samples)`.
        This means that `iter_num` is the current iteration number
        """

        def _wrapper_progress_bar(i, vals):
            result = fn(i, vals)
            _update_progress_bar(i + 1)
            return result

        return _wrapper_progress_bar

    return progress_bar_fori_loop
