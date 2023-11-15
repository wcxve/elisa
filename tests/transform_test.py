if __name__ == '__main__':
    import jax
    import numpy as np
    import numpyro
    from numpyro.distributions import Uniform, Normal
    from numpyro.infer import MCMC, NUTS
    from numpyro.infer.util import constrain_fn, unconstrain_fn, log_density, \
        log_likelihood

    numpyro.set_host_device_count(4)
    jax.config.update("jax_enable_x64", True)

    def model(data):
        mu = numpyro.sample('mu', Uniform(0, 3.1))
        with numpyro.plate('data', len(data)):
            numpyro.sample('y', Normal(mu, 1.0), obs=data)

    data = np.random.normal(2.1, 1, size=20)

    sampler = MCMC(
        NUTS(model),
        num_warmup=2000,
        num_samples=2000,
        num_chains=2,
        progress_bar=True,
    )
    # sampler.run(jax.random.PRNGKey(0), data)



    to_unconstrain = jax.jit(lambda params: unconstrain_fn(model, (data,), {}, params))
    to_constrain = jax.jit(lambda params: constrain_fn(model, (data,), {}, params))
    print(to_constrain(to_unconstrain({'mu': 1.2})))

    from numpyro.distributions import constraints
    from numpyro.distributions.transforms import ComposeTransform, AffineTransform, SigmoidTransform
    def transform_interval(low, high):
        scale = high - low
        return ComposeTransform(
            [
                SigmoidTransform(),
                AffineTransform(
                    low, scale, domain=constraints.unit_interval
                ),
            ]
        )
    # import pymc as pm
    # pm_trans = pm.distributions.transforms.Interval(lower=0, upper=3.1)
    trans = transform_interval(0, 3.1)

    grad = jax.grad(trans.inv)

    ll = jax.jit(
        lambda params, data:
            jax.tree_util.tree_reduce(
                lambda x, y: x + y,
                jax.tree_map(
                    lambda x: x.sum(),
                    log_likelihood(model, params, data)
                )
            )
    )
    print(ll({'mu': 2.1}, data+2))
