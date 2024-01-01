import jax
from elisa import *
path = '/Users/xuewc/ObsData/FRB221021/HXMT/'
LE = Data(
    [5, 10],
    f'{path}/LE_optbmin5.fits',
    f'{path}/LE_phabkg20s_g0_0-94.pha',
    f'{path}/LE_rsp.rsp',
    group='bmin',
    scale=10
)

ME = Data(
    [10, 35],
    f'{path}/ME_optbmin5.fits',
    f'{path}/ME_phabkg20s_g0_0-53.pha',
    f'{path}/ME_rsp.rsp',
    group='bmin',
    scale=10
)

HE = Data(
    [28, 250],
    f'{path}/HE_optbmin5.fits',
    f'{path}/HE_phabkg20s_g0_0-12.pha',
    f'{path}/HE_rsp.rsp',
    group='bmin',
    scale=10
)

model = Powerlaw()
fit = LikelihoodFit([LE, ME, HE], model, 'wstat')

from elisa.infer.simulation import Simulator
simulator = Simulator(fit._numpyro_model, fit._seed)
mle = fit.mle()
# res=mle._simfit.run_one_set(mle._result['constr'], 40000)
import numpy as np
params = jax.tree_map(
    lambda x: x + np.random.normal(loc=0, scale=0.1, size=80000),
    mle._result['constr']
)
res2=mle._simfit.run_multi_sets(params)

raise RuntimeError

import numpy as np
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import Mesh, PositionalSharding
import time

# simulator = Simulator(fit._numpyro_model, fit._seed)
# fn = simulator._get_dist
sim = simulator.sample_from_one_set(fit._mle['constr'], 10000)

np.random.default_rng(42)
params = jax.device_put(jax.tree_map(
    lambda x: x + np.random.normal(loc=0, scale=0.1, size=80000),
    fit._mle['constr']
))

t0 = time.time()
jax.block_until_ready(jax.vmap(simulator._get_dist)(params))
print(time.time() - t0)

t0 = time.time()
sharding = PositionalSharding(create_device_mesh((4,)))
sharded_params = jax.device_put(params, sharding)
jax.block_until_ready(jax.vmap(simulator._get_dist)(sharded_params))
print(time.time() - t0)


t0 = time.time()
params_pmap = jax.tree_map(
    lambda x: x.reshape((4, -1) + x.shape[1:]),
    params
)
jax.block_until_ready(jax.pmap(jax.vmap(simulator._get_dist))(params_pmap))
print(time.time() - t0)

t0 = time.time()
jax.block_until_ready(shard_map(
    jax.vmap(simulator._get_dist),
    Mesh(create_device_mesh((4,)), ('i',)),
    P('i'),
    P('i'),
    False
)(params))
print(time.time() - t0)

np.random.default_rng(42)
params = jax.tree_map(
    lambda x: x + np.random.normal(loc=0, scale=0.1, size=80000),
    fit._mle['constr']
)
t0 = time.time()
simulator.sample_from_multi_sets(params)
print(time.time() - t0)
