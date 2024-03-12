"""Miscellaneous helper functions."""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import reduce
from typing import Callable, TypeVar

from jax import lax
from jax.experimental import host_callback
from prettytable import PrettyTable
from tqdm.auto import tqdm

T = TypeVar('T')


def build_namespace(
    names: Sequence[str],
    latex: bool = False,
    prime: bool = False,
) -> list[str]:
    """Build a namespace from a sequence of names.

    Parameters
    ----------
    names : sequence of str
        A sequence of names.
    latex : bool, optional
        If True, `names` are assumed to be LaTeX strings. The default is False.
    prime : bool, optional
        If True, primes are used as suffix for duplicate names, otherwise
        a number is used. The default is False.

    Returns
    -------
    namespace: list of str
        A list of non-duplicate names.

    """
    # 'ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᵠᴿˢᵀᵁⱽᵂᕽʸᶻ'
    # 'ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖᵠʳˢᵗᵘᵛʷˣʸᶻ'
    # '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾'
    # '₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎'
    namespace = []
    names_ = []
    suffixes = []
    counter = {}

    for name in names:
        names_.append(name)

        if name not in namespace:
            counter[name] = 1
            namespace.append(name)
        else:
            counter[name] += 1
            namespace.append(f'{name}#{counter[name]}')

        suffixes.append(counter[name])

    if prime:
        suffixes = [i - 1 for i in suffixes]
        if latex:
            suffixes = ["'" * n for n in suffixes]
        else:
            suffixes = ['"' * (n // 2) + "'" * (n % 2) for n in suffixes]
    else:
        template = '_{%d}' if latex else '_%d'
        suffixes = [template % n if n > 1 else '' for n in suffixes]

    return [name + suffix for name, suffix in zip(names_, suffixes)]


def make_pretty_table(fields: Sequence[str], rows: Sequence) -> PrettyTable:
    """Make a :class:`prettytable.PrettyTable`.

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
        bottom_left_junction_char='└',
    )
    table.add_rows(rows)
    return table


# def ordered_composition(
#     basic_mapping: Sequence[ParamID],
#     composite_mapping: dict[ParamID, Sequence[ParamID]],
# ) -> tuple[ParamID, ...]:
#     """Get ordered composition.
#
#     Parameters
#     ----------
#     basic_mapping : sequence
#         Sequence of basic parameter id.
#     composite_mapping : dict
#         Mapping of composite parameter id to sequence of parameter id, which
#         may include id existing in `composite_mapping`.
#
#     Returns
#     -------
#     ordered : tuple
#         Ordered keys of `composite_mapping`.
#
#     """
#     nodes = set(i for pids in composite_mapping.values() for i in pids)
#     available_nodes = set(basic_mapping) | set(composite_mapping)
#     if nodes - available_nodes != set():
#         raise ValueError(
#             '`composition` id must be a subset of union of `basic_mapping` '
#             'and `composition`'
#         )
#
#     ordered = list(basic_mapping)
#     remains = dict(composite_mapping)
#     while remains:
#         to_del = []
#         for k, v in remains:
#             if all(pid in ordered for pid in v):
#                 to_del.append(k)
#             ordered.append(k)
#
#         for k in to_del:
#             del remains[k]
#
#     return tuple(k for k in ordered if k not in basic_mapping)


def replace_string(value: T, mapping: dict[str, str]) -> T:
    """Replace all strings in `value` appeared in `mapping`.

    Parameters
    ----------
    value : str, iterable or mapping
        Object whose str value needs to be replaced.

    mapping : dict
        Mapping of str value to be replaced and replacement.

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

    def replace_sequence(it: tuple | list):
        """Replace elements of iterable."""
        return type(it)(map(replace, it))

    def replace(v):
        """The main replace function."""
        if isinstance(v, str):
            return replace_with_mapping(v)
        elif isinstance(v, (list, tuple)):
            return replace_sequence(v)
        elif isinstance(v, dict):
            return replace_dict(v)
        else:
            return v

    return replace(value)


def progress_bar_factory(
    neval: int,
    ncores: int,
    init_str: str | None = None,
    run_str: str | None = None,
) -> Callable[[Callable], Callable]:
    """Add a progress bar to JAX ``fori_loop`` kernel, see [1]_ for details.

    Parameters
    ----------
    neval : int
        The total number of evaluations.
    ncores : int
        The number of cores.
    init_str : str, optional
        The string displayed before progress bar when initialization.
    run_str : str, optional
        The string displayed before progress bar when run.

    Returns
    -------
    progress_bar_fori_loop : callable
        Factory that adds a progress bar to function input.

    References
    ----------
    .. [1] `How to add a progress bar to JAX scans and loops
            <https://www.jeremiecoullon.com/2021/01/29/jax_progress_bar/>`_

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

    process_re = re.compile(r'\d+$')

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
            lambda _: host_callback.id_tap(_update_tqdm, 0, result=iter_num),
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
