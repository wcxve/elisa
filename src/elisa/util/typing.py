from typing import Union

import jax.numpy as jnp
import numpy as np
from jax import Array

FloatType = jnp.result_type(float)
IntType = jnp.result_type(int)

JAXFloat = Array
PRNGKey = Array
JAXArray = Array
NumpyArray = np.ndarray
Array = Union[NumpyArray, JAXArray]
