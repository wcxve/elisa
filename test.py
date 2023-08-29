import jax
from jax import jit
from xspec_models_cxc import powerlaw

# @jit
def pl(PhoIndex, energies):
    print(dir(PhoIndex), energies)
    return powerlaw([PhoIndex], energies)