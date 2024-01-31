from . import data, infer, model, util
from .data import *  # noqa F403
from .infer import *  # noqa F403
from .model import *  # noqa F403
from .util import jax_enable_x64, set_cpu_cores

__all__ = data.__all__ + infer.__all__ + model.__all__ + util.__all__

jax_enable_x64(True)
set_cpu_cores(4)
