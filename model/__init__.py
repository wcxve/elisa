from .base import UniformParameter
from .model import *

import bayespec.model.xspec.xs as xs

__all__ = ['UniformParameter', 'xs']
__all__.extend(model.__all__)
