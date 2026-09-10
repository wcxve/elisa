"""Compatibility workaround for Lineax 0.1.1 with JAX 0.10 and 0.11.

Lineax 0.1.1 compares the sharding metadata of ``ShapeDtypeStruct`` values.
Inside ``pmap``, this can reject mathematically compatible pytrees with
``ValueError: pytree does not match out_structure``.

This module adapts the relevant behavior from the Apache-2.0-licensed upstream
fix at https://github.com/patrick-kidger/lineax/pull/246.  Remove it once a
released Lineax containing that fix passes ELISA's parallel fitting and
bootstrap regression tests without this workaround.
"""

from importlib.metadata import version

import equinox as eqx
import jax
import jax.tree_util as jtu


def _strip_weak_dtype_and_sharding(tree):
    return jtu.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype)
        if type(x) is jax.ShapeDtypeStruct
        else x,
        tree,
    )


def _structure_equal(x, y) -> bool:
    x = _strip_weak_dtype_and_sharding(jax.eval_shape(lambda: x))
    y = _strip_weak_dtype_and_sharding(jax.eval_shape(lambda: y))
    return eqx.tree_equal(x, y) is True


def apply_lineax_workaround() -> None:
    """Ignore sharding only for the confirmed Lineax/JAX combinations."""
    jax_release = jax.__version_info__[:2]
    if version('lineax') != '0.1.1' or not (
        (0, 10) <= jax_release < (0, 12)
    ):
        return

    from lineax import _misc as lineax_misc
    from lineax import _solve as lineax_solve
    from lineax._solver import cg, gmres, misc

    lineax_misc.structure_equal = _structure_equal
    misc.structure_equal = _structure_equal
    cg.structure_equal = _structure_equal
    gmres.structure_equal = _structure_equal
    lineax_solve.strip_weak_dtype = _strip_weak_dtype_and_sharding
