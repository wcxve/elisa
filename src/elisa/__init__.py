from .__about__ import __version__ as __version__
from .data import *  # noqa F403
from .infer import *  # noqa F403
from .model import *  # noqa F403
from .util import jax_enable_x64, set_cpu_cores

jax_enable_x64(True)
set_cpu_cores(4)
