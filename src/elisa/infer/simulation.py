"""Data simulation and fit."""
from __future__ import annotations

from typing import Callable, Optional

import jax
import numpy as np
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import PositionalSharding
from numpyro import handlers

from . import fit as _fit
from .likelihood import PoissonWithGoodness


def _random_normal(seed, loc, scale, n=None):
    np.random.default_rng(seed)
    if n is None:
        shape = loc.shape
    else:
        shape = (n,) + loc.shape
    return np.random.normal(loc, scale, shape)


def _random_poisson(seed, lam, n=None):
    np.random.default_rng(seed)
    if n is None:
        shape = lam.shape
    else:
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
        self, params: dict[str, float], n: int, seed: Optional[int] = None
    ) -> dict[str, np.ndarray]:
        """Sample from one set of parameters.

        Parameters
        ----------
        params : dict
            Parameters in constrained space.
        n : int
            Sample size.
        seed : int, optional
            Random number seed used in sampling. By default, use the seed given
            in the initialization.

        Returns
        -------
        sample : dict
            Simulated observation samples.

        """
        if seed is None:
            seed = self._seed
        else:
            seed = int(seed)

        dist = self._get_dist(params)

        poisson_sample = jax.tree_map(
            lambda v: _random_poisson(seed, v, n), dist['poisson']
        )

        normal_sample = jax.tree_map(
            lambda v: _random_normal(seed, v[0], v[1], n), dist['normal']
        )

        sample = poisson_sample | normal_sample

        return sample

    def sample_from_multi_sets(
        self,
        params: dict[str, np.ndarray | jax.Array],
        seed: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        """Sample from multiple sets of parameters.

        Parameters
        ----------
        params : dict
            Parameters in constrained space. Note that the parameter size must
            be multiple of :func:`jax.device_count()`.
        seed : int, optional
            Random number seed used in sampling. By default, use the seed given
            in the initialization.

        Returns
        -------
        sample : dict
            Simulated observation samples.

        """
        if seed is None:
            seed = self._seed
        else:
            seed = int(seed)

        sharded_params = jax.device_put(params, self._sharding)
        dist = jax.vmap(self._get_dist)(sharded_params)

        poisson_sample = jax.tree_map(
            lambda v: _random_poisson(seed, v), dist['poisson']
        )

        normal_sample = jax.tree_map(
            lambda v: _random_normal(seed, v[0], v[1]), dist['normal']
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
    def __init__(self, fit: _fit.BaseFit):
        self._free_names = fit._free_names
        self._to_unconstr = fit._helper.to_unconstr_array
        self._net_counts = fit._helper.net_counts
        self._simulator = Simulator(fit._helper.numpyro_model, fit._seed)
        self._result_container = fit._helper.sim_result_container
        self._sequence_fitter = fit._helper.sim_sequence_fit
        self._parallel_fitter = fit._helper.sim_parallel_fit
        self._fit = fit

    def run_one_set(
        self,
        params: dict[str, float],
        n: int,
        seed: Optional[int] = None,
        parallel: bool = True,
        run_str: Optional[str] = None,
    ):
        """Simulate from one set of parameters and fit the simulation.

        Parameters
        ----------
        params : dict
            Parameters in constrained space.
        n : int
            Perform simulation and fit `n` times.
        seed : int, optional
            Random number seed used in sampling. By default, use the seed given
            by the fitting context.
        parallel : bool, optional
            Whether to run the fit in parallel. The default is True.
        run_str : str, optional
            Description of the progress bar.

        Returns
        -------
        results : dict
            The simulation results.

        """
        p_in = set(params.keys())
        p_all = set(self._free_names)
        if p_in != p_all:
            raise ValueError(f'require params {p_in - p_all}')

        if seed is None:
            seed = self._fit._seed
        else:
            seed = int(seed)

        sim_data = self._simulator.sample_from_one_set(params, n, seed)
        result = self._result_container(n)
        init_unconstr = self._to_unconstr(
            [params[i] for i in self._free_names]
        )
        init_unconstr = np.full((n, len(init_unconstr)), init_unconstr)

        if parallel:
            fitter = self._parallel_fitter
        else:
            fitter = self._sequence_fitter

        fit_result = fitter(sim_data, result, init_unconstr, run_str)

        return self._make_result(sim_data, fit_result)

    def run_multi_sets(
        self,
        params: dict[str, jax.Array],
        seed: Optional[int] = None,
        parallel: bool = True,
        run_str: Optional[str] = None,
    ):
        """Simulate from multiple sets of parameters and fit the simulation.

        Parameters
        ----------
        params : dict
            Parameters in constrained space.
        seed : int, optional
            Random number seed used in sampling. By default, use the seed given
            by the fitting context.
        parallel : bool, optional
            Whether to run the fit in parallel. The default is True.
        run_str : str, optional
            Description of the progress bar.

        Returns
        -------
        results : dict
            The simulation results.

        """
        p_in = set(params.keys())
        p_all = set(self._free_names)
        if p_in != p_all:
            raise ValueError(f'require params {p_in - p_all}')

        if seed is None:
            seed = self._fit._seed
        else:
            seed = int(seed)

        sim_data = self._simulator.sample_from_multi_sets(params, seed)
        init_unconstr = jax.vmap(self._to_unconstr)(
            np.column_stack([params[i] for i in self._free_names])
        )
        result = self._result_container(len(init_unconstr))

        if parallel:
            fitter = self._parallel_fitter
        else:
            fitter = self._sequence_fitter

        fit_result = fitter(sim_data, result, init_unconstr, run_str)

        return self._make_result(sim_data, fit_result)

    def _make_result(self, sim_data, result_container):
        result_container['data'] = sim_data | self._net_counts(sim_data)
        return result_container
