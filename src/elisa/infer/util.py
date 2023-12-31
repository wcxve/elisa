"""Helper functions for inference module."""
from __future__ import annotations

from typing import Callable, Sequence, TypeVar

import re
from functools import reduce

from jax import lax
from jax.experimental import host_callback
from prettytable import PrettyTable
from tqdm.auto import tqdm as tqdm_auto

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
    fn: Callable,
    num: int,
    num_process: int,
    init_fmt=None,
    run_fmt=None
) -> Callable:
    """Adapted from :func:`numpyro.util.progress_bar_factory`.
    Factory that builds a progress bar decorator along with the
    `set_tqdm_description` and `close_tqdm` functions
    """
    num = int(num)
    num_process = int(num_process)

    process_re = re.compile(r"\d+$")

    if init_fmt is None:
        init_fmt = 'Compiling... '
    else:
        init_fmt = str(init_fmt)

    if run_fmt is None:
        run_fmt = 'Running {process}'
    else:
        init_fmt = str(init_fmt)

    if num > 20:
        print_rate = int(num / 20)
    else:
        print_rate = 1

    remainder = num % print_rate

    tqdm_bars = {}
    finished = []
    for i in range(num_process):
        tqdm_bars[i] = tqdm_auto(range(num), position=i)
        tqdm_bars[i].set_description(init_fmt.format(process=i), refresh=True)

    def _update_tqdm(arg, transform, device):
        match = process_re.search(str(device))
        assert match
        p = int(match.group())
        tqdm_bars[p].set_description(run_fmt.format(process=p), refresh=False)
        tqdm_bars[p].update(arg)

    def _close_tqdm(arg, transform, device):
        match = process_re.search(str(device))
        assert match
        p = int(match.group())
        tqdm_bars[p].update(arg)
        finished.append(p)
        if len(finished) == num_process:
            for i in range(num_process):
                tqdm_bars[i].close()

    def _update_progress_bar(iter_num):
        """Updates tqdm progress bar of a JAX loop only if the iteration number
        is a multiple of the print_rate
        Usage: carry = progress_bar((iter_num, print_rate), carry)
        """

        _ = lax.cond(
            iter_num == 1,
            lambda _: host_callback.id_tap(
                _update_tqdm, 0, result=iter_num, tap_with_device=True
            ),
            lambda _: iter_num,
            operand=None,
        )
        _ = lax.cond(
            iter_num % print_rate == 0,
            lambda _: host_callback.id_tap(
                _update_tqdm, print_rate, result=iter_num, tap_with_device=True
            ),
            lambda _: iter_num,
            operand=None,
        )
        _ = lax.cond(
            iter_num == num,
            lambda _: host_callback.id_tap(
                _close_tqdm, remainder, result=iter_num, tap_with_device=True
            ),
            lambda _: iter_num,
            operand=None,
        )

    def progress_bar_fori_loop(func):
        """Decorator that adds a progress bar to `body_fun` used in
        `lax.fori_loop`.
        Note that `body_fun` must be looping over a tuple who's first element
        is `np.arange(num_samples)`.
        This means that `iter_num` is the current iteration number
        """

        def wrapper_progress_bar(i, vals):
            result = func(i, vals)
            _update_progress_bar(i + 1)
            return result

        return wrapper_progress_bar

    return progress_bar_fori_loop(fn)
