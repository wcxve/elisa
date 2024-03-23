"""Helper functions for computation environment configuration."""

from __future__ import annotations

import warnings
from multiprocessing import cpu_count

from numpyro import enable_x64, set_host_device_count


def jax_enable_x64(use_x64: bool) -> None:
    """Changes the default float precision of arrays in JAX.

    Parameters
    ----------
    use_x64 : bool
        When `True`, JAX arrays will use 64 bits else 32 bits.
    """
    enable_x64(bool(use_x64))


def set_cpu_cores(n: int) -> None:
    """Set CPU number to use, should be called before running JAX codes.

    Parameters
    ----------
    n : int
        CPU number to use.
    """
    n = int(n)
    total_cores = cpu_count()

    if n > total_cores:
        msg = f'only {total_cores} CPUs available, '
        msg += f'will use {total_cores - 1} CPUs'
        warnings.warn(msg, Warning)
        n = total_cores - 1

    set_host_device_count(n)
