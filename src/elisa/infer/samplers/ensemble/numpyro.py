from __future__ import annotations

import jax
from numpyro.infer.ensemble import AIES, ESS
from numpyro.infer.mcmc import MCMCKernel
from numpyro.util import is_prng_key


class NumpyroEnsembleSampler(MCMCKernel):
    """Wrapper kernel to run the ensemble sampler as a single MCMC chain.

    To collect the posterior correctly, get samples in shape of
    (n_parallel, n_steps, n_walkers) by

    .. code-block:: python

        from numpyro.infer import MCMC
        kernel = NumpyroEnsembleSampler(...)
        mcmc = MCMC(kernel, ...)
        mcmc.run(...)
        samples = mcmc.get_samples(group_by_chain=True)

    Then combine the walkers from the same sampler by

    .. code-block:: python

        import jax
        import jax.numpy as jnp
        samples = jax.tree.map(lambda x: jnp.swapaxes(x, 1, 2), samples)
        samples = jax.tree.map(
            lambda x: jnp.reshape(
                x,
                (x.shape[0], x.shape[1] * x.shape[2], *x.shape[3:]),
            ),
            samples,
        )
    """

    _kernel: type[AIES | ESS]

    def __init__(self, walkers: int, *args, **kwargs):
        self._walkers = int(walkers)
        self._sampler = self._kernel(*args, **kwargs)

    @property
    def sample_field(self):
        return 'z'

    def postprocess_fn(self, args, kwargs):
        return jax.vmap(self._sampler.postprocess_fn(args, kwargs))

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        if not is_prng_key(rng_key):
            raise NotImplementedError(
                "EnsembleSampler only supports chain_method='parallel' or "
                "chain_method='sequential'. Please put in a feature request "
                'if it would be useful to be used in vectorized mode.'
            )
        rng_keys = jax.random.split(rng_key, self._walkers)
        if init_params is not None:
            if not all(
                param.shape[0] == self._walkers
                for param in jax.tree.leaves(init_params)
            ):
                raise ValueError(
                    'The batch dimension of each param must match chains'
                )

        return self._sampler.init(
            rng_keys, num_warmup, init_params, model_args, model_kwargs
        )

    def sample(self, state, model_args, model_kwargs):
        return self._sampler.sample(state, model_args, model_kwargs)


class NumPyroAIES(NumpyroEnsembleSampler):
    _kernel = AIES

    def get_diagnostics_str(self, state):
        return f'acc. prob={state.inner_state.mean_accept_prob:.2f}'


class NumPyroESS(NumpyroEnsembleSampler):
    _kernel = ESS
