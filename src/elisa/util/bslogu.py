"""Bi-Symmetric log transformation for uniform distribution.

See https://iopscience.iop.org/article/10.1088/0957-0233/24/2/027001.

Contributed by @xiesl97 (https://github.com/xiesl97).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from numpyro.distributions import Uniform, constraints
from numpyro.distributions.distribution import (
    Transform,
    TransformedDistribution,
)
from numpyro.distributions.util import promote_shapes


@jax.jit
def log(x, c=0.43429448190325176):
    """
    transformation  x -> y
    c default to 1/log(10)
    """
    return jnp.sign(x) * jnp.log10(1 + jnp.abs(x / c))


@jax.jit
def pow(y, c=0.43429448190325176):
    """
    inverse transformation  y -> x
    c default to 1/log(10)
    """
    return jnp.sign(y) * c * (-1 + jnp.power(10, jnp.abs(y)))


class BiSymTransform(Transform):
    # TODO: refine domain/codomain Bi-Symmetric logic through setters,
    #  especially when transforms for inverses are supported
    def __init__(self, domain=constraints.real, c=0.43429448190325176):
        self.domain = domain
        self._c = c

    @property
    def codomain(self):
        if self.domain is constraints.ordered_vector:
            return constraints.positive_ordered_vector
        elif self.domain is constraints.real:
            return constraints.positive
        elif isinstance(self.domain, constraints.greater_than):
            return constraints.greater_than(
                self.__call__(self.domain.lower_bound)
            )
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
        return jnp.log(jnp.pow(10.0, jnp.abs(x)))

    def tree_flatten(self):
        return (self.domain,), (('domain',), {})

    def __eq__(self, other):
        if not isinstance(other, BiSymTransform):
            return False
        return self.domain == other.domain


class BiSymLogUniform(TransformedDistribution):
    """
    the input should be transfor before input

    for example:
    import numpy as np
    import bslogu as bs
    from numpyro.distributions import LogUniform, Uniform

    Uniform( low , high )
    LogUniform( np.log(low) , np.log(high) )
    BiSymLogUniform( bs.log(low) , bs.log(high) )
    """

    arg_constraints = {
        'low': constraints.dependent,
        'high': constraints.dependent,
    }
    reparametrized_params = ['low', 'high']
    pytree_data_fields = ('low', 'high', '_support')

    def __init__(
        self,
        low,
        high,
        *,
        c=0.43429448190325176,
        validate_args=None,
    ):
        base_dist = Uniform(log(low, c), log(high, c))
        self._c = c
        self.low, self.high = promote_shapes(low, high)
        self._support = constraints.interval(self.low, self.high)
        super().__init__(
            base_dist, BiSymTransform(c=c), validate_args=validate_args
        )

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    @property
    def mean(self):
        raise NotImplementedError

    @property
    def variance(self):
        raise NotImplementedError

    def cdf(self, x):
        return self.base_dist.cdf(log(x, self._c))
