if __name__ == '__main__':
    ...
    # BB = BlackBody()
    # BB2 = BlackBody()
    # m = BB+BB2
    # TODO: syntax: m.BB.kT

    # Done!
    # model1 = Model1(parA=1.1, parB=None)
    # model2 = Model2(parA=None)
    # model = model1 * model2
    # signature1: model(egrid, params=None)
    # signature2: model(egrid, params={'model1': {'parB': 1.0}})

    # from jax.tree_util import tree_map, tree_reduce
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
    #         return lambda e: prev(e) + component._eval(*params, e)
    #     elif op == 'mul':
    #         return lambda e: prev(e) * component._eval(*params, e)
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

    # >>> using haiku >>>
    # import haiku as hk
    # class A(hk.Module):
    #     def __call__(self, e):
    #         alpha = hk.get_parameter('alpha', (), e.dtype, jnp.ones)
    #         return e**-alpha
    #
    # func = hk.without_apply_rng(hk.transform(lambda e: A()(e)))
    #
    # def eval_pl(components):
    #     for c in components:
    #         params = [
    #             hk.get_parameter(name, (), jnp.float64, jnp.ones)
    #             for name in c.params
    #         ]
    #         func = lambda e: c.eval(e, params)
    #     return pl(e, alpha)
    #
    # def pl(e, alpha):
    #     return e ** -alpha
    #
    # func2 = hk.without_apply_rng(hk.transform(lambda e: eval_pl(e)))
    # <<< using haiku <<<

    # >>> using equinox >>>
    # import equinox as eqx
    # from typing import Callable
    # class PL(eqx.Module):
    #     default = (('alpha', 1, 2, 3, False),)
    #     alpha: float
    #     _eval: Callable
    #
    #     def __init__(self, alpha=None):
    #         super().__init__()
    #         self.alpha = self.default[0][1] if alpha is None else alpha
    #         self._eval = self.eval
    #
    #     @staticmethod
    #     def eval(e, alpha):
    #         return e**-alpha
    #
    # pl = PL()
    # params, static = eqx.partition(pl.eval, eqx.is_array)
    # <<< using equinox <<<
