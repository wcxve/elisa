import arviz as az
import jax
import numpy as np
import numpyro
import xarray as xr
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS
jax.config.update("jax_enable_x64", True)

np.random.default_rng(42)
data = np.random.poisson(lam=3, size=50)

@jax.jit
def model(data):
    lam = numpyro.sample(
        'lam',
        dist.LogUniform(0.1, 200)
    )

    numpyro.sample(
        'obs',
        dist.Poisson(lam),
        obs=data
    )

sampler = MCMC(NUTS(model), num_warmup=2000, num_samples=20000, num_chains=2)
sampler.run(jax.random.PRNGKey(42), data)
idata = az.from_numpyro(sampler)

np.random.default_rng(42)
ppc_sample = np.random.poisson(
    idata.posterior.lam.values[:, :, None],
    size=idata.posterior.lam.values[:, :].shape + (50,)
)

idata_ppc = az.from_dict(
    posterior={'lam': idata.posterior.lam.values},
    posterior_predictive={'obs': ppc_sample},
    log_likelihood={'obs': idata.log_likelihood.obs.data},
    observed_data={'obs': data},
    coords={'chain': np.arange(2),
        'draw': np.arange(20000),
        'obs_dim_0': np.arange(50)},
    dims={'lam': ['chain', 'draw'],
       'obs': ['obs_dim_0']},
    pred_dims={'obs': ['chain', 'draw', 'obs_dim_0']}
)
# az.plot_loo_pit(idata_ppc, 'obs', ecdf=True, hdi_prob=0.95)
prob = 0.95
ess_p = az.ess(idata_ppc.posterior, method="mean")
n_chain = 2
if n_chain > 1:
    reff = np.hstack([ess_p[v].values.flatten() for v in ess_p.data_vars]).mean() / 40000
else:
    reff = 1.0
log_weights = az.psislw(-idata_ppc.log_likelihood['obs'].stack(__sample__=("chain", "draw")), reff=reff)[0]
az.loo_pit(y=idata_ppc.observed_data['obs'],
           y_hat=idata_ppc.posterior_predictive['obs'].stack(__sample__=("chain", "draw")),
           log_weights=log_weights)
loo_pit = np.sort(az.loo_pit(idata_ppc, y='obs'))
ndata = loo_pit.size
ecdf_diff = loo_pit - np.arange(ndata) / ndata
from scipy import stats
unif_ecdf = np.arange(ndata + 1)
upper = stats.beta.ppf(0.5 + prob / 2, unif_ecdf + 1, ndata - unif_ecdf + 1)
lower = stats.beta.ppf(0.5 - prob / 2, unif_ecdf + 1, ndata - unif_ecdf + 1)
unif_ecdf = unif_ecdf / ndata
import matplotlib.pyplot as plt
# plt.figure()
# plt.fill_between(unif_ecdf, lower-unif_ecdf, upper-unif_ecdf, alpha=0.1, color='r', step='mid')
# plt.plot(np.hstack((0, loo_pit, 1)), np.hstack((0, ecdf_diff, 0)), 'ro:', ds='steps-mid')
