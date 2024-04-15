import jax
from numpyro.distributions import constraints,Uniform
from numpyro.distributions.distribution import TransformedDistribution,Transform

from numpyro.distributions.util import promote_shapes

# # Bi-Symmetric log transformation
# # https://iopscience.iop.org/article/10.1088/0957-0233/24/2/027001
@jax.jit
def log(x, c=0.43429448190325176):
    '''
    transformation  x -> y
    c default to 1/log(10)
    '''
    import jax.numpy as jnp
    return jnp.sign(x) * jnp.log10( 1 + jnp.abs(x/c) )

@jax.jit
def pow(y, c=0.43429448190325176):
    '''
    inverse transformation  y -> x
    c default to 1/log(10)
    '''
    import jax.numpy as jnp
    return jnp.sign(y) * c * ( -1 + jnp.power(10,jnp.abs(y)) )


class BiSymTransform(Transform):
    # TODO: refine domain/codomain Bi-Symmetric logic through setters, especially when
    # transforms for inverses are supported
    def __init__(self, domain=constraints.real):
        self.domain = domain
        self._c = 0.43429448190325176 # 1/log(10)

    @property
    def codomain(self):
        if self.domain is constraints.ordered_vector:
            return constraints.positive_ordered_vector
        elif self.domain is constraints.real:
            return constraints.positive
        elif isinstance(self.domain, constraints.greater_than):
            return constraints.greater_than(self.__call__(self.domain.lower_bound))
        elif isinstance(self.domain, constraints.interval):
            return constraints.interval(
                self.__call__(self.domain.lower_bound),
                self.__call__(self.domain.upper_bound),
            )
        else:
            raise NotImplementedError

    def __call__(self, x):
        # XXX consider to clamp from below for stability if necessary
        return pow(x, self._c)

    def _inverse(self, y):
        return log(y, self._c)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return x

    def tree_flatten(self):
        return (self.domain,), (("domain",), dict())

    def __eq__(self, other):
        if not isinstance(other, BiSymTransform):
            return False
        return self.domain == other.domain

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, value):
        self._c = value

class BiSymLogUniform(TransformedDistribution):
    '''
    the input should be transfor before input

    for example:
    import numpy as np
    import bslogu as bs
    from numpyro.distributions import LogUniform, Uniform
    
    Uniform( low , high )
    LogUniform( np.log(low) , np.log(high) )
    BiSymLogUniform( bs.log(low) , bs.log(high) )
    '''
    arg_constraints = {"low": constraints.dependent, "high": constraints.dependent}
    reparametrized_params = ["low", "high"]
    pytree_data_fields = ("low", "high", "_support")

    def __init__(self, low, high, *, validate_args=None):
        self._c = 0.43429448190325176 # 1/log(10)
        base_dist = Uniform(log(low, self._c), log(high, self._c))
        self.low, self.high = promote_shapes(low, high)
        self._support = constraints.interval(self.low, self.high)
        super(BiSymLogUniform, self).__init__(
            base_dist, BiSymTransform(), validate_args=validate_args
        )

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    @property
    def mean(self):
        return (self.high - self.low) / log(self.high / self.low, self._c)

    @property
    def variance(self):
        return (
            0.5 * (self.high**2 - self.low**2) / log(self.high / self.low, self._c)
            - self.mean**2
        )

    def cdf(self, x):
        return self.base_dist.cdf(log(x, self._c))

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, value):
        self._c = value
        for s in self.transforms:
            if hasattr(s, 'c'):
                s.c = value

