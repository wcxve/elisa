from .__about__ import __version__
from .data import *
from .infer import *
from .model import *
from .util import *

jax_enable_x64(True)
set_cpu_cores(4)
