if __name__ == '__main__':
    import jax
    import numpyro
    import numpyro.distributions as dist
    import jax.numpy as jnp
    from numpyro import infer

    def linear_model(x, yerr, y=None):
        theta = numpyro.sample("theta", dist.Uniform(-0.5 * jnp.pi, 0.5 * jnp.pi))
        b_perp = numpyro.sample("b_perp", dist.Normal(0, 1))

        m = numpyro.deterministic("m", jnp.tan(theta))
        b = numpyro.deterministic("b", b_perp / jnp.cos(theta))

        with numpyro.plate("data", len(x)):
            numpyro.sample("y", dist.Normal(m * x + b, yerr), obs=y)

    import numpy as np
    import matplotlib.pyplot as plt

    # We'll choose the parameters of our synthetic data.
    # The outlier probability will be 80%:
    true_frac = 0.8

    # The linear model has unit slope and zero intercept:
    true_params = [1.0, 0.0]

    # The outliers are drawn from a Gaussian with zero mean and unit variance:
    true_outliers = [0.0, 1.0]

    # For reproducibility, let's set the random number seed and generate the data:
    np.random.seed(12)
    x = np.sort(np.random.uniform(-2, 2, 15))
    yerr = 0.2 * np.ones_like(x)
    y = true_params[0] * x + true_params[1] + yerr * np.random.randn(len(x))

    # Those points are all drawn from the correct model so let's replace some of
    # them with outliers.
    m_bkg = np.random.rand(len(x)) > true_frac
    y[m_bkg] = true_outliers[0]
    y[m_bkg] += np.sqrt(true_outliers[1] + yerr[m_bkg] ** 2) * np.random.randn(sum(m_bkg))

    # Then save the *true* line.
    x0 = np.linspace(-2.1, 2.1, 200)
    y0 = np.dot(np.vander(x0, 2), true_params)

    sampler = infer.MCMC(
        infer.NUTS(linear_model),
        num_warmup=2000,
        num_samples=2000,
        num_chains=2,
        progress_bar=True,
    )

    sampler.run(jax.random.PRNGKey(0), x, yerr, y=y)
