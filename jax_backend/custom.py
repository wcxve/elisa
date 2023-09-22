from jax.core import Primitive

import jax.experimental.host_callback as hcb
import jax.numpy as jnp
import jax

def myprint(x):
  print('inside myprint:', type(x), x)
  return x

@jax.jit
def device_fun(x):
  return hcb.call(myprint, x, result_shape=x)

device_fun(jnp.arange(10))