import jax
import numpyro
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", False)
numpyro.set_host_device_count(4)
from elisa.data.ogip import Data
from elisa.infer import LikelihoodFit, BayesianFit
from numpyro.distributions import Normal, LogNormal, Uniform
from elisa.model import *
import arviz as az
import jaxopt
import jax.numpy as jnp
from jax import lax
from numpyro import handlers
from jax_tqdm import loop_tqdm
import time

path = '/Users/xuewc/ObsData/FRB221021/HXMT/'

LE = Data([5, 10], f'{path}/LE_optbmin5.fits',
          f'{path}/LE_phabkg20s_g0_0-94.pha', f'{path}/LE_rsp.rsp',
          group='bmin', scale=10
          )

ME = Data([10, 35], f'{path}/ME_optbmin5.fits',
          f'{path}/ME_phabkg20s_g0_0-53.pha', f'{path}/ME_rsp.rsp',
          group='bmin', scale=10
          )

HE = Data([28, 250], f'{path}/HE_optbmin5.fits',
          f'{path}/HE_phabkg20s_g0_0-12.pha', f'{path}/HE_rsp.rsp',
          group='bmin', scale=10
          )
model = Powerlaw()
fit2 = LikelihoodFit([LE, ME, HE], model, 'wstat')
fit2.mle()

n=10000
mle_array = jnp.array(list(fit2._mle['unconstr'].values()))

print('Start simulate')
t0 = time.time()
simulation = fit2.boot(1000)
print(f'Finish simulate: {time.time() - t0:.2f} s')

print('Start simulate')
t0 = time.time()
simulation = fit2._simulator(fit2._mle['constr'], n)
print(f'Finish simulate: {time.time() - t0:.2f} s')
deviance = fit2._deviance
residual = fit2._residual


def fit_one_sim(i, args):
    sim_data, result, init = args
    residual_for_sim = handlers.substitute(
        residual,
        jax.tree_map(lambda x: x[i], sim_data)
    )
    res = jaxopt.LevenbergMarquardt(
        residual_fun=residual_for_sim,
        stop_criterion='grad-l2-norm'
    ).run(init)
    result['params'] = result['params'].at[i].set(res.params)
    result['stat'] = result['stat'].at[i].set(2.0*res.state.value)
    result['grad'] = result['grad'].at[i].set(res.state.error)
    result['residual'] = result['residual'].at[i].set(res.state.residual)
    return sim_data, result, init

# nworkers = jax.device_count()
# body_pbar = progress_bar_factory(n, 1)(fit_one_sim)

# from jax.experimental import mesh_utils
# from jax.sharding import PositionalSharding
# devices1 = mesh_utils.create_device_mesh((nworkers, 1))
# sharding1 = PositionalSharding(devices1)
# devices2 = mesh_utils.create_device_mesh((nworkers,))
# sharding2 = PositionalSharding(devices2)
# sharded_simulation = jax.device_put(simulation, sharding1)
# sharded_result = {
#     'params': jax.device_put(jnp.empty((n, len(fit2._free_names))), sharding1),
#     'stat': jax.device_put(jnp.empty(n), sharding2),
#     'grad': jax.device_put(jnp.empty(n), sharding2),
#     'residual': jax.device_put(jnp.empty((n, fit2._nchan['total']*2)), sharding1)
# }
# result = {
#     'params': jnp.empty((n, len(fit2._free_names))),
#     'stat': jnp.empty(n),
#     'grad': jnp.empty(n),
#     'residual': jnp.empty((n, fit2._nchan['total']*2))
# }
# body_pbar = progress_bar_factory(n, 1)(fit_one_sim)
# res = lax.fori_loop(0, n, body_pbar, (simulation, result, mle_array))

from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map

mesh = Mesh(mesh_utils.create_device_mesh((4,)), axis_names=('i',))
result = {
    'params': jnp.empty((n, len(fit2._free_names))),
    'stat': jnp.empty(n),
    'grad': jnp.empty(n),
    'residual': jnp.empty((n, fit2._nchan['total']*2))
}
from functools import partial
def fit_one_sim(i, args):
    sim_data, result, init = args
    residual_for_sim = handlers.substitute(
        residual,
        jax.tree_map(lambda x: x[i], sim_data)
    )
    res = jaxopt.LevenbergMarquardt(
        residual_fun=residual_for_sim,
        stop_criterion='grad-l2-norm'
    ).run(init)
    result['params'] = result['params'].at[i].set(res.params)
    result['stat'] = result['stat'].at[i].set(2.0*res.state.value)
    result['grad'] = result['grad'].at[i].set(res.state.error)
    result['residual'] = result['residual'].at[i].set(res.state.residual)
    return sim_data, result, init
# body_pbar = progress_bar_factory(n//4, 4)(fit_one_sim)

print('Start')
t0 = time.time()
res = shard_map(
    lambda *x: lax.fori_loop(0, n//4, fit_one_sim, x)[1],
    mesh,
    in_specs=(P('i'), P('i'), P()),
    out_specs=P('i'),
    check_rep=False
)(simulation, result, mle_array)
print(time.time() - t0)
