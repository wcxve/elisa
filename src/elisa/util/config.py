"""Helper functions for computation environment configuration."""

from __future__ import annotations

import os
import re
import warnings
from multiprocessing import cpu_count
from typing import TYPE_CHECKING

import jax

if TYPE_CHECKING:
    from typing import Literal


def jax_enable_x64(use_x64: bool) -> None:
    """Changes the default float precision of arrays in JAX.

    Parameters
    ----------
    use_x64 : bool
        When ``True``, JAX arrays will use 64 bits else 32 bits.
    """
    if not use_x64:
        use_x64 = os.getenv('JAX_ENABLE_X64', 0)
    jax.config.update('jax_enable_x64', bool(use_x64))


def set_jax_platform(platform: Literal['cpu', 'gpu', 'tpu'] | None = None):
    """Set JAX platform to CPU, GPU, or TPU.

    .. warning::
        This utility takes effect only before running any JAX program.

    Parameters
    ----------
    platform : {'cpu', 'gpu', 'tpu'}, optional
        Either ``'cpu'``, ``'gpu'``, or ``'tpu'``.
    """
    if platform is None:
        platform = os.getenv('JAX_PLATFORM_NAME', 'cpu')

    assert platform in {'cpu', 'gpu', 'tpu', None}

    jax.config.update('jax_platform_name', platform)

    if platform == 'gpu':
        # see https://jax.readthedocs.io/en/latest/gpu_performance_tips.html
        xla_gpu_flags = (
            '--xla_gpu_enable_triton_softmax_fusion=true '
            '--xla_gpu_triton_gemm_any=True '
            '--xla_gpu_enable_async_collectives=true '
            '--xla_gpu_enable_latency_hiding_scheduler=true '
            '--xla_gpu_enable_highest_priority_async_stream=true'
        )
        xla_flags = os.getenv('XLA_FLAGS', '')
        if xla_gpu_flags not in xla_flags:
            os.environ['XLA_FLAGS'] = f'{xla_flags} {xla_gpu_flags}'


def set_cpu_cores(n: int) -> None:
    """Set device number to use in JAX.

    .. warning::
        This utility takes effect only for CPU platform and before running any
        JAX program.

    Parameters
    ----------
    n : int
        Device number to use.
    """
    n = int(n)
    total_cores = cpu_count()

    if n > total_cores:
        msg = f'only {total_cores} CPUs available, '
        msg += f'will use {total_cores - 1} CPUs'
        warnings.warn(msg, Warning)
        n = total_cores - 1

    xla_flags = os.getenv('XLA_FLAGS', '')
    xla_flags = re.sub(
        r'--xla_force_host_platform_device_count=\S+', '', xla_flags
    ).split()
    os.environ['XLA_FLAGS'] = ' '.join(
        [f'--xla_force_host_platform_device_count={n}'] + xla_flags
    )


def jax_debug_nans(flag: bool):
    """Automatically detect when NaNs are produced when running JAX codes.

    See JAX `docs <https://jax.readthedocs.io/en/latest/debugging/flags.html>`_
    for details.

    Parameters
    ----------
    flag : bool
        When ``True``, raise an error when NaNs are detected in JAX.
    """
    jax.config.update('jax_debug_nans', bool(flag))
