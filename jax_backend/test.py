from model.add import *


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.tree_util import tree_map, tree_reduce

    BB = BlackBody(kT=1)
    BB2 = BlackBody(kT=2)
    m = BB+BB2

    # class BlackBody:
    #
    #     def _eval(self, kT, norm, e):
    #         print('eval')
    #         return norm * 8.0525 * e*e / (kT*kT*kT*kT * jnp.expm1(e / kT))
    #
    # BB = BlackBody()

    # funcs = {
    #     'BB': BB._eval
    # }
    # pars = {
    #     'BB': [np.r_[1.0, 1.5], np.r_[1.0, 1.]]
    # }

    # eval_map = lambda e: tree_map(
    #     lambda f, p: f(*p, e),
    #     funcs,
    #     pars,
    #     is_leaf=lambda v: isinstance(v, list)
    # )
    # eval_reduce = lambda e: tree_reduce()

    # model1 = Model1(parA=1.1, parB=None)
    # model2 = Model2(parA=None)
    # model = model1 * model2
    # signature1: model(egrid, params=None)
    # signature2: model(egrid, params={'model1': {'parB': 1.0}})

    # components = [
    #     (BB, 2, 1),
    #     ('add', BB, 3, 2),
    #     ('add', BB, 5, 2),
    #     ('mul', BB, 3, 1),
    #     ('mul', BB, 3, 1.5)
    # ]
    #
    # def operation(prev, args):
    #     op, component, *params = args
    #     if op == 'add':
    #         return lambda e: print('add') or (prev(e) + component._eval(*params, e))
    #     elif op == 'mul':
    #         return lambda e: print('mul') or (prev(e) * component._eval(*params, e))
    #     elif op == 'con':
    #         return lambda e: prev(e, flux=component._eval(*params, e))
    #     else:
    #         raise NotImplementedError(f'{op} not implemented')
    #
    # def initializer(components):
    #     component, *params = components[0]
    #     return lambda e: component._eval(*params, e)
    #
    # f = tree_reduce(
    #     operation,
    #     components[1:],
    #     initializer(components),
    #     is_leaf=lambda x: isinstance(x, tuple)
    # )
    # jit_f = jax.jit(f)
    # egrid = jnp.arange(1.0, 10.0, 0.1)
    # print(jit_f(egrid))
    #
    # exec('f = lambda x: x*x', tmp := {})
    # jax.jit(tmp['f'])(1)
