import jax
import numpy as np
import numpyro
from numpyro.distributions import Normal, LogNormal, Uniform
from elisa import *
import arviz as az
import jaxopt
import jax.numpy as jnp



a = UniformParameter('a', 'a', 1.0, 0.1, 2, log=1)
a.max = 3
b = UniformParameter('a', 'a', 1.0, 0.1, 2, log=1, frozen=1)
c = UniformParameter('c', 'c', 1.0, 0, 2)
d = a+b
e = c*d
f = generate_parameter('f', 'f', 2.0, Normal(), dist_expr='Normal(mu=0.0, sigma=1.0)')
g = e*f
m = Constant()*PhFlux(1,10)*Powerlaw()
m1 = Bbody(fmt=r'\mathrm{BB}')
m2 = Bbody(K=1, fmt=r'\mathrm{BB}')
m2.kT = m1.kT * g
m3 = m2 + m1
det = 'n7'
spec = f'/Users/xuewc/ObsData/GRB231115A/GBM/{det}.pha'
back = f'/Users/xuewc/ObsData/GRB231115A/GBM/{det}.bak'
resp = f'/Users/xuewc/ObsData/GRB231115A/GBM/{det}.rsp'
d = Data((8, 900), spec, back, resp, name=det, group='bpos', scale=1e-7)
# fit1 = LikelihoodFit(d, BlackBodyRad(), 'pgstat')

import time
t0 = time.time()
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

# print(time.time() - t0)
model = Powerlaw()
model.alpha.max=5
# print(time.time() - t0)
fit2 = LikelihoodFit([LE, ME, HE], model, 'wstat')

print(time.time() - t0)
mle = fit2.mle()
print(time.time() - t0)
raise ValueError
# t0 = time.time()
# fit2.boot(10000)
# la = jax.experimental.multihost_utils.process_allgather(fit2._boot.params)
# print(fit2._boot.params['powerlaw_K'][-1])
# fit2.ci(method='boot')
# print(time.time() - t0)
model.K.log=True
fit3 = BayesianFit([LE, ME, HE], model, 'wstat')
# raise
fit3.nuts()
from numpyro import handlers
fit2.mle()
mle_array = jnp.array(list(fit2._mle['unconstr'].values()))

t0 = time.time()
n=10000
simulation = fit2._simulator(fit2._mle['constr'], n)
print(time.time() - t0)

rep = {}
for k, v in simulation.items():
    rep[k.replace('Non', 'spec').replace('Noff', 'back')] = v
in_axes = ({k: 0 for k in simulation.keys()},)
deviance = fit2._deviance
residual = fit2._residual

@jax.jit
# @jax.vmap
def fit_one_sim1(sim_data):
    residual_for_sim = handlers.substitute(residual, sim_data)
    res = jaxopt.LevenbergMarquardt(
        residual_fun=residual_for_sim,
        stop_criterion='madsen-nielsen'
    ).run(mle_array)
    return res.params, 2.0*res.state.value, res.state.error, res.state.residual


@jax.jit
def fit_one_sim2(sim_data):
    deviance_for_sim = handlers.substitute(deviance, sim_data)
    res = jaxopt.BFGS(fun=deviance_for_sim).run(mle_array)
    return res  # .params, res.state.fun_val, res.state.success


def fit_one_sim(i, args):
    sim_data, result, init = args
    residual_for_sim = handlers.substitute(
        residual,
        jax.tree_map(lambda x: x[i], sim_data)
    )
    res = jaxopt.LevenbergMarquardt(
        residual_fun=residual_for_sim,
        stop_criterion='madsen-nielsen'
    ).run(init)
    result['params'] = result['params'].at[i].set(res.params)
    result['stat'] = result['stat'].at[i].set(2.0*res.state.value)
    result['grad'] = result['grad'].at[i].set(res.state.error)
    result['residual'] = result['residual'].at[i].set(res.state.residual)
    return sim_data, result, init

nworkers = 4
from elisa.infer.util import progress_bar_factory
body_pbar = progress_bar_factory(n // nworkers, nworkers)(fit_one_sim)

sim_data = jax.tree_map(
    lambda x: jnp.reshape(x, (nworkers, -1, x.shape[-1])),
    simulation
)
result = {
    'params': jnp.empty((nworkers, n//nworkers, len(fit2._free_names))),
    'stat': jnp.empty((nworkers, n//nworkers)),
    'grad': jnp.empty((nworkers, n//nworkers)),
    'residual': jnp.empty((nworkers, n//nworkers, fit2._nchan['total']*2))
}
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
devices1 = mesh_utils.create_device_mesh((nworkers, 1, 1))
sharding1 = PositionalSharding(devices1)
devices2 = mesh_utils.create_device_mesh((nworkers, 1))
sharding2 = PositionalSharding(devices2)
sim = jax.device_put(sim_data, sharding1)
res = {
    'params': jax.device_put(jnp.empty((nworkers, n//nworkers, len(fit2._free_names))), sharding1),
    'stat': jax.device_put(jnp.empty((nworkers, n//nworkers)), sharding2),
    'grad': jax.device_put(jnp.empty((nworkers, n//nworkers)), sharding2),
    'residual': jax.device_put(jnp.empty((nworkers, n//nworkers, fit2._nchan['total']*2)), sharding1)
}
init = mle_array
loop = lambda sim, res: jax.lax.fori_loop(0, n//nworkers, body_pbar, (sim, res, init))[1]
pmap_loop = jax.pmap(loop)
t0 = time.time()
r = pmap_loop(sim, res)
print(time.time() - t0)

# fit3 = BayesianFit([LE, ME, HE], model, 'wstat')
# from elisa.inference.nested_sampling import reparam_loglike
# loglik, transform, names = reparam_loglike(fit2._numpyro_model, jax.random.PRNGKey(42))

# @jax.jit
# def loglik_fn(params):
#     params = {name: value for name, value in zip(names, params)}
#     return loglik(**params)

# import corner
# import ultranest
#
# sampler = ultranest.ReactiveNestedSampler(names, jax.jit(loglik_fn))
# result = sampler.run(show_status=False, viz_callback=False)
# sampler.print_results()
# idata = az.from_dict(transform({k: v for k, v in zip(names, result['samples'].T)}))
# corner.corner(idata, show_titles=1)
# from nautilus import Prior
# from nautilus import Sampler
#
#
# prior = Prior()
# [prior.add_parameter(k) for k in names]
#
# sampler = Sampler(prior, jax.jit(loglik_fn), n_live=1000, pass_dict=False)
# sampler.run(verbose=True, discard_exploration=True)
# idata2 = az.from_dict(transform({k: v for k, v in zip(names, sampler.posterior(equal_weight=True)[0].T)}))
# corner.corner(idata2, show_titles=1)
