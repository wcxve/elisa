from .data import *
from .model import *
from .infer import *
from .util import *

__version__ = '0.0.1.dev1'

jax_enable_x64(True)
set_cpu_cores(4)
