"""Miscellaneous helper functions."""

from __future__ import annotations

import math
import re
import warnings
from functools import reduce
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from astropy.units import Unit
from jax import lax, tree_util
from jax.custom_derivatives import SymbolicZero
from jax.experimental import host_callback
from jax.flatten_util import ravel_pytree
from prettytable import PrettyTable
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Callable, Literal, TypeVar

    from elisa.util.typing import CompEval

    T = TypeVar('T')

_SUPERSCRIPT = dict(
    zip(
        'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-/=()',
        'ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᵠᴿˢᵀᵁⱽᵂᕽʸᶻᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖᵠʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ᐟ⁼⁽⁾',
    )
)
_SUBSCRIPT = dict(
    zip(
        'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-/=()',
        'ᴀʙᴄᴅᴇғɢʜɪᴊᴋʟᴍɴᴏᴘǫʀsᴛᴜᴠᴡxʏᴢₐᵦ𝒸𝒹ₑ𝒻𝓰ₕᵢⱼₖₗₘₙₒₚᵩᵣₛₜᵤᵥ𝓌ₓᵧ𝓏₀₁₂₃₄₅₆₇₈₉₊₋⸝₌₍₎',
    )
)


def add_suffix(
    strings: str | Sequence[str],
    suffix: str | Sequence[str],
    subscript: bool = True,
    unicode: bool = False,
    latex: bool = False,
    mathrm: bool = False,
) -> str | list[str]:
    """Add suffix to a sequence of strings.

    Parameters
    ----------
    strings : sequence of str
        The sequence of strings.
    suffix : sequence of str
        The sequence of suffixes.
    subscript : bool, optional
        If True, add suffix as subscript, otherwise superscript.
        The default is True.
    latex : bool, optional
        If True, add suffix following LaTeX format. The default is False.
    unicode : bool, optional
        If True, add suffix with Unicode string. The default is False.
    mathrm : bool, optional
        If True, add suffix in mathrm when latex is True. The default is False.

    Returns
    -------
    str or list of str
        The strings with suffix added.
    """
    return_list = False

    if isinstance(strings, str):
        strings = [strings]
    else:
        strings = list(strings)
        return_list = True

    if isinstance(suffix, str):
        suffix = [suffix]
    else:
        suffix = list(suffix)
        return_list = True

    if len(strings) != len(suffix):
        raise ValueError('length of `strings` and `suffix` must be the same')

    def to_unicode(string: str):
        """Replace suffix with unicode."""
        if subscript:
            return ''.join(f'{_SUBSCRIPT.get(i, i)}' for i in string)
        else:
            return ''.join(f'{_SUPERSCRIPT.get(i, i)}' for i in string)

    if latex:
        symbol = '_' if subscript else '^'
        rm = r'\mathrm' if mathrm else ''
        strings = [
            rf'{{{i}}}{symbol}{rm}{{{j}}}' if j else i
            for i, j in zip(strings, suffix)
        ]
    elif unicode:
        strings = [
            f'{i}{to_unicode(j)}' if j else i for i, j in zip(strings, suffix)
        ]
    else:
        symbol = '_' if subscript else '^'
        strings = [
            f'{i}{symbol}{j}' if j else i for i, j in zip(strings, suffix)
        ]

    if return_list:
        return strings
    else:
        return strings[0]


def build_namespace(
    names: Sequence[str],
    latex: bool = False,
    prime: bool = False,
) -> dict[str, list[str | int]]:
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
    namespace: dict
        A dict of non-duplicate names and suffixes in original name order.
    """
    namespace = []
    names_ = []
    suffixes_n = []
    counter = {}

    for name in names:
        names_.append(name)

        if name not in namespace:
            counter[name] = 1
            namespace.append(name)
        else:
            counter[name] += 1
            namespace.append(f'{name}#{counter[name]}')

        suffixes_n.append(counter[name])

    if prime:
        suffixes = [i - 1 for i in suffixes_n]
        if latex:
            suffixes = ["'" * n for n in suffixes]
        else:
            suffixes = ['"' * (n // 2) + "'" * (n % 2) for n in suffixes]
    else:
        template = '_{%d}' if latex else '_%d'
        suffixes = [template % n if n > 1 else '' for n in suffixes_n]

    return {
        'namespace': list(map(''.join, zip(names_, suffixes))),
        'suffix_num': [str(n) if n > 1 else '' for n in suffixes_n],
    }


def define_fdjvp(
    fn: CompEval,
    method: Literal['central', 'forward'] = 'central',
) -> CompEval:
    """Define JVP using finite differences."""
    if method not in {'central', 'forward'}:
        raise ValueError(
            f"supported methods are 'central' and 'forward', but got "
            f"'{method}'"
        )

    def fdjvp(primals, tangents):
        egrid, params = primals
        egrid_tangent, params_tangent = tangents

        if not isinstance(egrid_tangent, SymbolicZero):
            raise NotImplementedError('JVP for energy grid is not implemented')

        primals_out = fn(egrid, params)

        tvals, _ = tree_util.tree_flatten(params_tangent)
        if any(jnp.shape(v) != () for v in tvals):
            raise NotImplementedError(
                'JVP for non-scalar parameter is not implemented'
            )

        non_zero_tangents = [not isinstance(v, SymbolicZero) for v in tvals]
        idx = [i for i, v in enumerate(non_zero_tangents) if v]
        idx_arr = jnp.array(idx)
        nbatch = sum(non_zero_tangents)
        nparam = len(tvals)
        params_ravel, revert = ravel_pytree(params)
        free_params_values = params_ravel[idx_arr]
        free_params_abs = jnp.where(
            jnp.equal(free_params_values, 0.0),
            jnp.ones_like(free_params_values),
            jnp.abs(free_params_values),
        )
        free_params_abs = jnp.expand_dims(free_params_abs, axis=-1)
        row_idx = jnp.arange(nbatch)
        perturb_idx = jnp.zeros((nbatch, nparam)).at[row_idx, idx_arr].set(1.0)
        params_batch = jnp.full((nbatch, nparam), params_ravel)

        eps = jnp.finfo(egrid.dtype).eps
        f_vmap = jax.vmap(fn, in_axes=(None, 0), out_axes=0)
        revert = jax.vmap(revert, in_axes=0, out_axes=0)

        # See Numerical Recipes Chapter 5.7
        if method == 'central':
            perturb = free_params_abs * eps ** (1.0 / 3.0)
            params_pos_perturb = revert(params_batch + perturb_idx * perturb)
            out_pos_perturb = f_vmap(egrid, params_pos_perturb)
            params_neg_perturb = revert(params_batch - perturb_idx * perturb)
            out_neg_perturb = f_vmap(egrid, params_neg_perturb)
            d_out = (out_pos_perturb - out_neg_perturb) / (2.0 * perturb)
        else:
            perturb = free_params_abs * jnp.sqrt(eps)
            params_perturb = revert(params_batch + perturb_idx * perturb)
            out_perturb = f_vmap(egrid, params_perturb)
            d_out = (out_perturb - primals_out) / perturb

        free_params_tangent = jnp.array([tvals[i] for i in idx])
        tangents_out = free_params_tangent @ d_out
        return primals_out, tangents_out

    fn = jax.custom_jvp(fn)
    fn.defjvp(fdjvp, symbolic_zeros=True)

    return fn


def get_parallel_number(n: int | None) -> int:
    """Check and return the available parallel number in JAX.

    Parameters
    ----------
    n : int, optional
        The desired number of parallel processes in JAX.

    Returns
    -------
    int
        The available number of parallel processes.
    """
    n_max = jax.local_device_count()

    if n is None:
        return n_max
    else:
        n = int(n)
        if n <= 0:
            raise ValueError(
                f'number of parallel processes must be positive, got {n}'
            )

    if n > n_max:
        warnings.warn(
            f'number of parallel processes ({n}) is more than the number of '
            f'available devices ({jax.local_device_count()}), reset to '
            f'{jax.local_device_count()}',
            Warning,
        )
        n = jax.local_device_count()

    return n


def get_unit_latex(unit: str, throw: bool = True) -> str:
    """Get latex string of a unit.

    Parameters
    ----------
    unit : str
        The unit string.
    throw : bool, optional
        If True, raise ValueError if the unit is invalid. The default is True.

    Returns
    -------
    str
        The latex string of the unit.
    """
    ustr = str(unit)
    if ustr:
        try:
            unit = Unit(ustr)
            max_index = len(ustr)
            pattern = r'(?:[^a-zA-Z]*){}(?:[^a-zA-Z]*)'
            index = [
                min(
                    r.start(0)
                    if (r := re.search(pattern.format(s), ustr)) is not None
                    else max_index
                    for s in [b.name] + b.aliases
                )
                for b in unit.bases
            ]
            index = sorted(range(len(index)), key=index.__getitem__)
            bases = [unit.bases[i].name for i in index]
            powers = [unit.powers[i] for i in index]
            ustr = r'\ '.join(
                b + (f'^{{{p}}}' if p != 1 else '')
                for b, p in zip(bases, powers)
            )
            scale = Unit(unit.scale).to_string('latex_inline')[9:-2]
            if scale != '':
                scale = scale.replace(r'1 \times ', '')
                scale += r'\ '
            ustr = rf'$\mathrm{{{scale}{ustr}}}$'
        except ValueError as ve:
            if throw:
                raise ve
            ustr = ''

    return ustr


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
        top_right_junction_char='╮',
        top_left_junction_char='╭',
        bottom_right_junction_char='╯',
        bottom_left_junction_char='╰',
    )
    table.add_rows(rows)
    return table


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


def report_interval(
    vmid: float,
    vmin: float,
    vmax: float,
    precision: int = 2,
    min_exponent: int = 1,
    max_exponent: int = 2,
) -> str:
    r"""Report parameter interval in :math:`\LaTeX` format.

    Parameters
    ----------
    vmid : float
        The mid value.
    vmin : float
        The lower bound.
    vmax : float
        The upper bound.
    precision : int, optional
        The precision of the mid value. The default is 2.
    min_exponent : int, optional
        The minimum exponent to use scientific notation. The default is 1.
    max_exponent : int, optional
        The maximum exponent to use scientific notation. The default is 2.

    Returns
    -------
    str
        The interval in :math:`|LaTeX` format.
    """
    vmid = float(vmid)
    vmin = float(vmin)
    vmax = float(vmax)
    precision = int(precision)
    min_exponent = int(min_exponent)
    max_exponent = int(max_exponent)

    # assert vmin <= vmid <= vmax
    assert precision > 0
    assert min_exponent > 0
    assert max_exponent > 0

    def get_sci_notation_exponent(num: float) -> int:
        """Get the exponent of a number in scientific notation."""
        return int(f'{num:.{precision}e}'.split('e')[1])

    def get_sci_notation_significand(num: float, exp: int) -> str:
        """Get the significand of a number in scientific notation."""
        significand = num * 10**-exp
        if abs(num) < 10 ** (exp - precision):
            p = abs(get_sci_notation_exponent(num) - exp)
            return f'{significand:+.{p}f}'.rstrip('0')
        else:
            p = precision
            return f'{significand:+.{p}f}'

    lower = vmin - vmid
    upper = vmax - vmid
    exponent = math.log10(abs(vmid))

    if exponent <= -min_exponent or exponent >= max_exponent:
        str_mid = f'{vmid:.{precision}e}'.split('e')[0]
        base_exponent = math.floor(exponent)
        suffix = rf' \times 10^{{{base_exponent}}}'
    else:
        str_mid = f'{vmid:.{precision}f}'
        base_exponent = 0
        suffix = ''

    if lower != 0:
        str_lower = get_sci_notation_significand(lower, base_exponent)
    else:
        str_lower = '-0'
    if upper != 0:
        str_upper = get_sci_notation_significand(upper, base_exponent)
    else:
        str_upper = '+0'

    return f'${str_mid}_{{{str_lower}}}^{{{str_upper}}}{suffix}$'


def progress_bar_factory(
    neval: int,
    ncores: int,
    init_str: str | None = None,
    run_str: str | None = None,
    update_rate: int = 50,
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
    update_rate : int, optional
        The update rate of the progress bar. The default is 50.

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

    if neval > update_rate:
        print_rate = int(neval_single / update_rate)
    else:
        print_rate = 1

    remainder = neval_single % print_rate
    finished = [False] * ncores
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
