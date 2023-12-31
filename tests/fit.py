import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.distributions import Normal, LogNormal, Uniform
from elisa import *


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
fit1 = LikelihoodFit(d, Bbodyrad(), 'pgstat')

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

model = Powerlaw()
model.alpha.max=5
fit2 = LikelihoodFit([LE, ME, HE], model, 'wstat')

mle = fit2.mle()
# fit2.ci(method='boot')
model.K.log=True
fit3 = BayesianFit([LE, ME, HE], model, 'wstat')
post1 = fit3.nuts()
post2 = fit3.ns()
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
