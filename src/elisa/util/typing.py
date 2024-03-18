"""Typing aliases to shorten hints."""

from typing import Callable, TypeVar, Union

import numpy as np
from jax import Array
from jax.typing import ArrayLike

__all__ = [
    'PyFloat',
    'JAXFloat',
    'Float',
    'PRNGKey',
    'NumPyArray',
    'JAXArray',
    'Array',
    'ArrayLike',
    'CompID',
    'CompName',
    'CompParamName',
    'ParamID',
    'ParamName',
    'NameValMapping',
    'CompIDParamValMapping',
    'CompIDStrMapping',
    'ParamIDStrMapping',
    'ParamIDValMapping',
    'ParamNameValMapping',
    'CompEval',
    'ConvolveEval',
    'ModelEval',
    'ModelCompiledFn',
    'NameLaTeX',
    'AdditiveFn',
]

T = TypeVar('T')

PyFloat = Union[float, np.inexact]  # must include 0-d NDArray with float dtype
JAXFloat = Array
Float = Union[PyFloat, JAXFloat]

PRNGKey = Array

NumPyArray = np.ndarray
JAXArray = Array
Array = Union[NumPyArray, JAXArray]

ArrayLike = ArrayLike

# Type aliases for parameter and model module
CompID = CompName = CompParamName = ParamID = ParamName = str
NameValMapping = dict[CompParamName, JAXFloat]
CompIDParamValMapping = dict[CompID, NameValMapping]
CompIDStrMapping = dict[CompID, str]
ParamIDStrMapping = dict[ParamID, str]
ParamIDValMapping = dict[ParamID, JAXFloat]
ParamNameValMapping = dict[ParamName, JAXFloat]
CompEval = Callable[[JAXArray, NameValMapping], JAXArray]
ConvolveEval = Callable[
    [JAXArray, NameValMapping, Callable[[JAXArray], JAXArray]], JAXArray
]
ModelEval = Callable[[JAXArray, CompIDParamValMapping], JAXArray]
ModelCompiledFn = Callable[[JAXArray, ParamIDValMapping], JAXArray]
NameLaTeX = tuple[str, str]
AdditiveFn = Callable[[JAXArray, ParamIDValMapping], dict[NameLaTeX, JAXArray]]
