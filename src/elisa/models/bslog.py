import numpy as np

from jax import lax, vmap
from jax.experimental.sparse import BCOO
from jax.lax import scan
import jax.nn as nn
import jax.numpy as jnp
import jax.random as random
from jax.scipy.linalg import cho_solve, solve_triangular
from jax.scipy.special import (
    betaln,
    expi,
    expit,
    gammainc,
    gammaln,
    logit,
    multigammaln,
    ndtr,
    ndtri,
    xlog1py,
    xlogy,
)
from jax.scipy.stats import norm as jax_norm

from numpyro.distributions import Distribution, LogUniform, Uniform
from numpyro.distributions import constraints
from numpyro.distributions.discrete import _to_logits_bernoulli
from numpyro.distributions.distribution import Distribution, TransformedDistribution
from numpyro.distributions.transforms import (
    AffineTransform,
    CorrMatrixCholeskyTransform,
    ExpTransform,
    PowerTransform,
    SigmoidTransform,
)
from numpyro.distributions.util import (
    betainc,
    betaincinv,
    cholesky_of_inverse,
    gammaincinv,
    lazy_property,
    matrix_to_tril_vec,
    promote_shapes,
    signed_stick_breaking_tril,
    validate_sample,
    vec_to_tril_matrix,
)
from numpyro.util import is_prng_key

# # Bi-Symmetric log transformation
# # https://iopscience.iop.org/article/10.1088/0957-0233/24/2/027001
def BiSymmetricLog(x: np.float64,
                   C: np.float64=1/np.log(10)) -> np.float64:
    return np.sign(x) * np.log10( 1 + np.abs(x/C) )


def InverseBiSymmetricLog(y: np.float64,
                            C: np.float64=1/np.log(10)) -> np.float64:
    return np.sign(y) * C * ( -1 + np.power(10,np.abs(y)) )


class BiSymmetricLogUniform(Distribution):
    arg_constraints = {"low": constraints.dependent, "high": constraints.dependent}
    reparametrized_params = ["low", "high"]
    pytree_data_fields = ("low", "high", "_support")

    def __init__(self, low=-1.0, high=1.0, *, validate_args=None):
        self._c = 1/jnp.log(10)
        self.low, self.high = promote_shapes(low, high)
        batch_shape = lax.broadcast_shapes(jnp.shape(low), jnp.shape(high))
        self._support = constraints.interval(low, high)
        super().__init__(batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    def sample(self, key, sample_shape=()):
        log_low = self.bsLog(self.low)
        log_high = self.bsLog(self.high)
        shape = sample_shape + self.batch_shape
        return self.bsLog_inverse(
            random.uniform(key, shape=shape, minval=log_low, maxval=log_high)
            )

    @validate_sample
    def log_prob(self, value):
        shape = lax.broadcast_shapes(jnp.shape(value), self.batch_shape)
        return -jnp.broadcast_to(self.bsLog(self.high - self.low), shape)

    def cdf(self, value):
        log_value = self.bsLog(value)
        log_low = self.bsLog(self.low)
        log_high = self.bsLog(self.high)
        cdf = (log_value - log_low) / (log_high - log_low)
        return jnp.clip(cdf, a_min=0.0, a_max=1.0)

    def icdf(self, value):
        log_value = self.bsLog(value)
        log_low = self.bsLog(self.low)
        log_high = self.bsLog(self.high)
        return log_low + log_value * (log_high - log_low)

    @property
    def mean(self):
        log_low = self.bsLog(self.low)
        log_high = self.bsLog(self.high)
        return self.bsLog_inverse( log_low + (log_high - log_low) / 2.0 )

    @property
    def variance(self):
        log_low = self.bsLog(self.low)
        log_high = self.bsLog(self.high)
        return self.bsLog_inverse( (log_high - log_low) ** 2 / 12.0 )

    @staticmethod
    def infer_shapes(low=(), high=()):
        batch_shape = lax.broadcast_shapes(low, high)
        event_shape = ()
        return batch_shape, event_shape

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, value):
        self._c = value

    def bsLog(self, x):
        return jnp.sign(x) * jnp.log10( 1 + jnp.abs(x/self._c) )

    def bsLog_inverse(self, y):
        return jnp.sign(y) * self._c * ( -1 + jnp.power(10,jnp.abs(y)) )





