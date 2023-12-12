import jax
import numpyro
jax.config.update("jax_enable_x64", True)
numpyro.set_host_device_count(4)
from elisa.data.ogip import Data
from elisa.inference.fit import LikelihoodFit, BayesianFit
from numpyro.distributions import Normal, LogNormal
from elisa.model import *
import arviz as az


a = UniformParameter('a', 'a', 1.0, 0.1, 2, log=1)
b = UniformParameter('b', 'b', 1.0, 0.1, 2, log=1, frozen=1)
c = UniformParameter('c', 'c', 1.0, 0, 2)
d = a+b
e = c*d
f = generate_parameter('f', 'f', 2.0, Normal())
g = e*f
m = Constant()*PhFlux(1,10)*Powerlaw()
m1 = BlackBody(K=1, fmt='BB')
m2 = BlackBody(fmt='BB')
m2.kT = m1.kT * UniformParameter('f', 'f', 0.5, 0.001, 1, log=True)
m3 = m2 + m1
det = 'n7'
spec = f'/Users/xuewc/ObsData/GRB231115A/GBM/{det}.pha'
back = f'/Users/xuewc/ObsData/GRB231115A/GBM/{det}.bak'
resp = f'/Users/xuewc/ObsData/GRB231115A/GBM/{det}.rsp'
d = Data((8, 900), spec, back, resp, name=det, group='bpos', scale=0.001)
fit1 = LikelihoodFit(d, m3, 'pgstat')
path = '/Users/xuewc/ObsData/FRB221021/HXMT/'
LE = Data([5, 10], f'{path}/LE_optbmin5.fits',
          f'{path}/LE_phabkg20s_g0_0-94.pha', f'{path}/LE_rsp.rsp',
          # group='bmin', scale=
          )

ME = Data([10, 35], f'{path}/ME_optbmin5.fits',
          f'{path}/ME_phabkg20s_g0_0-53.pha', f'{path}/ME_rsp.rsp',
          # group='bmin', scale=10
          )

HE = Data([28, 250], f'{path}/HE_optbmin5.fits',
          f'{path}/HE_phabkg20s_g0_0-12.pha', f'{path}/HE_rsp.rsp',
          # group='bmin', scale=10
          )
# model = CutoffPowerlaw(method='simpson')
model = Powerlaw()
# model.alpha.max = 5
fit2 = LikelihoodFit([LE, ME, HE], model, 'wstat')
fit3 = BayesianFit([LE, ME, HE], model, 'wstat')
from elisa.inference.nested_sampling import reparam_loglike
loglik, transform, names = reparam_loglike(fit2._numpyro_model, jax.random.PRNGKey(42))

@jax.jit
def loglik_fn(params):
    params = {name: value for name, value in zip(names, params)}
    return loglik(**params)

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
