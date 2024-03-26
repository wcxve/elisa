"""Helper functions for computation environment configuration."""

from __future__ import annotations

import os
import warnings
from multiprocessing import cpu_count

import jax


def jax_enable_x64(use_x64: bool) -> None:
    """Changes the default float precision of arrays in JAX.

    Parameters
    ----------
    use_x64 : bool
        When `True`, JAX arrays will use 64 bits else 32 bits.
    """
    if not use_x64:
        use_x64 = os.getenv('JAX_ENABLE_X64', 0)
    jax.config.update('jax_enable_x64', use_x64)


def set_platform(platform: str = 'cpu'):
    """
    Changes platform to CPU, GPU, or TPU. This utility only takes
    effect at the beginning of your program.

    :param str platform: either 'cpu', 'gpu', or 'tpu'. The default is 'cpu'.
    """

    jax.config.update('jax_platform_name', platform)

    # <https://jax.readthedocs.io/en/latest/gpu_performance_tips.html>
    if platform == 'gpu':
        os.environ['XLA_FLAGS'] = (
            '--xla_gpu_enable_triton_softmax_fusion=true '
            '--xla_gpu_triton_gemm_any=True '
            '--xla_gpu_enable_async_collectives=true '
            '--xla_gpu_enable_latency_hiding_scheduler=true '
            '--xla_gpu_enable_highest_priority_async_stream=true '
        )


def set_cpu_cores(n: int) -> None:
    """Set CPU number to use, should be called before running JAX codes.
    This utility only takes effect at CPU platform and the beginning
    of your program.

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

    os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={n}'


def set_debug_nan(flag: bool):
    """Automatically detect when NaNs are produced.
    <https://jax.readthedocs.io/en/latest/debugging/flags.html>

    Parameters
    ----------
    flag : bool
        When `True`, raises an error when NaNs is detected.
    """
    jax.config.update('jax_debug_nans', flag)
