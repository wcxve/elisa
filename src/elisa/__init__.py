from .model import *  # noqa: F403
from .util import jax_enable_x64, set_cpu_cores

jax_enable_x64(True)
set_cpu_cores(4)
