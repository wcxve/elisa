from .data import *
from .model import *
from .infer import *
from .util import *
from .__about__ import __version__

jax_enable_x64(True)
set_cpu_cores(4)
