"""Data simulation and fit."""
from __future__ import annotations

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np

from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import PositionalSharding
from numpyro import handlers

from .likelihood import PoissonWithGoodness
from .util import OptFn, progress_bar_factory


def _random_normal(seed, loc, scale, n):
    np.random.default_rng(seed)
    shape = (n,) + loc.shape
    return np.random.normal(loc, scale, shape)


def _random_poisson(seed, lam, n):
    np.random.default_rng(seed)
    shape = (n,) + lam.shape
    return np.random.poisson(lam, shape).astype(np.float64)


class Simulator:
    """Observation simulator for numpyro model.

    This simulator uses :mod:`numpy.random` instead of :mod:`jax.random` for
    the performance.

    Parameters
    ----------
    model : callable
        A callable containing numpyro primitives.
    seed : int
        Random number seed.

    """

    def __init__(self, model: Callable, seed: int):
        self._model = model
        self._seed = int(seed)
        self._get_dist = jax.jit(self._make_dist_fn())
        device = create_device_mesh((jax.device_count(),))
        self._sharding = PositionalSharding(device)

    def sample_from_one_set(
        self,
        params: dict[str, float],
        n: int,
    ) -> dict[str, np.ndarray]:
        """Sample from one set of parameters.

        Parameters
        ----------
        params : dict
            Parameters in constrained space.
        n : int
            Sample size.

        Returns
        -------
        sample : dict
            Simulated observation samples.

        """
        seed = self._seed
        dist = self._get_dist(params)

        poisson_sample = jax.tree_map(
            lambda v: _random_poisson(seed, v, n),
            dist['poisson']
        )

        normal_sample = jax.tree_map(
            lambda v: _random_normal(seed, v[0], v[1], n),
            dist['normal']
        )

        sample = poisson_sample | normal_sample

        return sample

    def sample_from_multi_sets(
        self,
        params: dict[str, np.ndarray],
        n: int = 1,
    ) -> dict[str, np.ndarray]:
        """Sample from multiple sets of parameters.

        Parameters
        ----------
        params : dict
            Parameters in constrained space. Note that the parameter size must
            be multiple of :func:`jax.device_count()`.
        n : int
            Sample size.

        Returns
        -------
        sample : dict
            Simulated observation samples.

        """
        seed = self._seed
        sharded_params = jax.device_put(params, self._sharding)
        dist = jax.vmap(self._get_dist)(sharded_params)

        poisson_sample = jax.tree_map(
            lambda v: _random_poisson(seed, v, n),
            dist['poisson']
        )

        normal_sample = jax.tree_map(
            lambda v: _random_normal(seed, v[0], v[1], n),
            dist['normal']
        )

        sample = poisson_sample | normal_sample

        return sample

    def _make_dist_fn(self) -> Callable:
        model = self._model

        def get_dist(params: dict[str, float]) -> dict:
            """Get sample distributions of observations given parameters.

            Parameters
            ----------
            params : dict
                Parameters in constrained space.

            Returns
            -------
            dist : dict
                Sample distributions.

            """
            m = handlers.substitute(model, params)
            trace = handlers.trace(m).get_trace()

            dist = {'poisson': {}, 'normal': {}}

            for k, v in trace.items():
                flag1 = k.endswith('_Non')
                flag2 = k.endswith('_Noff')

                if flag1 or flag2:
                    if flag1:
                        k2 = k[:-4] + '_spec'
                    else:
                        k2 = k[:-5] + '_back'

                    fn = v['fn']
                    if isinstance(fn, PoissonWithGoodness):
                        dist['poisson'][k2] = fn.rate
                    else:  # instance of NormalWithGoodness
                        dist['normal'][k2] = (fn.loc, fn.scale)

            return dist

        return get_dist


class SimFit:
    def __init__(
        self,
        params: Sequence[str],
        ndata: dict[str, int],
        optfn: OptFn,
    ):
        self._params = params
        self._ndata = ndata
        self._optfn = optfn

        @jax.jit
        def fit_one_simulation(i, args):
            """Fit the i-th simulation."""
            sim_data, result, init = args

            residual_for_sim = jax.jit(
                handlers.substitute(
                    optfn.residual,
                    jax.tree_map(lambda x: x[i], sim_data)
                )
            )

            res = jaxopt.LevenbergMarquardt(
                residual_fun=residual_for_sim,
                stop_criterion='grad-l2-norm'
            ).run(init)

            state = res.state
            valid = jnp.bitwise_not(
                jnp.isnan(state.value)
                | jnp.isnan(state.error)
                | jnp.greater(state.error, 1e-3)
            )

            params = optfn.to_params_dict(optfn.to_dict(res.params))
            stat_info = optfn.deviance_unconstr_info(res.params)
            stat_group = stat_info['group']
            stat_point = stat_info['point']

            result['stat'] = result['stat'].at[i].set(2.0 * state.value)
            result['grad'] = result['grad'].at[i].set(state.error)
            result['residual'] = result['residual'].at[i].set(state.residual)
            result['valid'] = result['valid'].at[i].set(valid)

            for k in result['params']:
                result['params'][k] = result['params'][k].at[i].set(params[k])

            group = result['stat_group']
            point = result['stat_point']
            for k in self._data:
                group[k] = group[k].at[i].set(stat_group[k])
                point[k] = point[k].at[i].set(stat_point[k])

            return sim_data, result, init

        # batch_fit = lambda *args: shard_map(
        #     lambda *x: jax.lax.fori_loop(0, args[0], fit_one_simulation, x)[1],
        #     mesh,
        #     in_specs=(PartitionSpec('i'), PartitionSpec('i'), PartitionSpec()),
        #     out_specs=PartitionSpec('i'),
        #     check_rep=False
        # )(*args[1:])
        #
        # def batch_fit_simulation(simulation, result_container, init):
        #     """Batch fit the simulation."""
        #     batch_size = len(result_container['stat']) // cores
        #     return batch_fit(batch_size, simulation, result_container, init)

    def run(self, params, n, seed):
        ...
        # net counts, deviance_rep, deviance_fit22

    def _make_result_container(self, n: int):
        result_container = {
            'params': {k: jnp.empty(n) for k in self._params},
            'stat': jnp.empty(n),
            'stat_group': {k: jnp.empty(n) for k in self._ndata.keys()},
            'stat_point': {
                k: jnp.empty((n, v)) for k, v in self._ndata.items()
            },
            'stat_sign': {
                k: jnp.empty((n, v)) for k, v in self._ndata.items()
            },
            'valid': jnp.full(n, True, bool)
        }

        return result_container

    def _make_result(self, result_container):
        # net_counts ...
        ...
