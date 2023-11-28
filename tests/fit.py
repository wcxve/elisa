from elisa.model.base import UniformParameter, generate_parameter
from elisa.model.add import BlackBody
from elisa.data.ogip import Data
from elisa.inference.fit import LikelihoodFit
from numpyro.distributions import Normal

a = UniformParameter('a', 'a', 1.0, 0.1, 2, log=1)
b = UniformParameter('b', 'b', 1.0, 0.1, 2, log=1, frozen=1)
c = UniformParameter('c', 'c', 1.0, 0, 2)
d = a+b
e = c*d
f = generate_parameter('f', 'f', 2.0, Normal())
g = e*f

m1 = BlackBody(fmt='BB')
m2 = BlackBody(fmt='BB')
m2.kT = m1.kT * UniformParameter('f', 'f', 0.5, 0.001, 1, log=True)
m3 = m2 + m1
det = 'n7'
spec = f'/Users/xuewc/ObsData/GRB231115A/GBM/{det}.pha'
back = f'/Users/xuewc/ObsData/GRB231115A/GBM/{det}.bak'
resp = f'/Users/xuewc/ObsData/GRB231115A/GBM/{det}.rsp'
d = Data((8, 900), spec, back, resp, name=det, group='bpos', scale=0.001)

f = LikelihoodFit(d, m3, 'pgstat')
sample_func = f._generate_sample()
determ_func = f._generate_deterministic()

def model():
    sample_site = sample_func()
    determ_func(sample_site)
    x = numpyro.sample('x', Normal())
    numpyro.deterministic('_x', x)

import jax
jax.config.update("jax_enable_x64", True)
import arviz as az
import numpyro
numpyro.set_host_device_count(2)
from numpyro import infer



sampler = infer.MCMC(
    infer.NUTS(model),
    num_warmup=2000,
    num_samples=2000,
    num_chains=2,
    progress_bar=True,
)
sampler.run(jax.random.PRNGKey(0))
idata = az.from_numpyro(sampler)