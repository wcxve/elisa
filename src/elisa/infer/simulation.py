"""Data simulation and fit."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import numpy as np
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import PositionalSharding
from numpyro import handlers

from .likelihood import PoissonWithGoodness

if TYPE_CHECKING:
    from .fit import BaseFit


def _random_normal(seed, loc, scale, n=None):
    rng = np.random.default_rng(seed)
    if n is None:
        shape = loc.shape
    else:
        shape = (n,) + loc.shape
    return rng.normal(loc, scale, shape)


def _random_poisson(seed, lam, n=None):
    rng = np.random.default_rng(seed)
    if n is None:
        shape = lam.shape
    else:
        shape = (n,) + lam.shape
    return rng.poisson(lam, shape).astype(np.float64)


class Simulator:
    """Observation simulator for a numpyro model.

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
        device = create_device_mesh((jax.local_device_count(),))
        self._sharding = PositionalSharding(device)

    def samples_from_one_set(
        self,
        params: dict[str, float],
        n: int,
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Samples from one set of parameters.

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
        samples : dict
            Simulated observation samples.

        """
        if seed is None:
            seed = self._seed
        else:
            seed = int(seed)

        dist = self._get_dist(params)

        samples = {
            k: _random_poisson(seed, *v, n) for k, v in dist['poisson'].items()
        } | {k: _random_normal(seed, *v, n) for k, v in dist['normal'].items()}

        return samples

    def samples_from_multi_sets(
        self,
        params: dict[str, np.ndarray | jax.Array],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Samples from multiple sets of parameters.

        Parameters
        ----------
        params : dict
            Parameters in constrained space. Note that the parameter size must
            be multiple of :func:`jax.local_device_count()`.
        seed : int, optional
            Random number seed used in sampling. By default, use the seed given
            in the initialization.

        Returns
        -------
        samples : dict
            Simulated observation samples.

        """
        if seed is None:
            seed = self._seed
        else:
            seed = int(seed)

        sharded_params = jax.device_put(params, self._sharding)
        dist = jax.vmap(self._get_dist)(sharded_params)

        samples = {
            k: _random_poisson(seed, *v) for k, v in dist['poisson'].items()
        } | {k: _random_normal(seed, *v) for k, v in dist['normal'].items()}

        return samples

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
            prior : dict
                Sample distributions.

            """
            m = handlers.substitute(model, params)
            trace = handlers.trace(m).get_trace()

            dist = {'poisson': {}, 'normal': {}}

            for k, v in trace.items():
                if k.endswith('_Non') or k.endswith('_Noff'):
                    fn = v['fn']
                    if isinstance(fn, PoissonWithGoodness):
                        dist['poisson'][f'{k}_data'] = (fn.rate,)
                    else:  # instance of NormalWithGoodness
                        dist['normal'][f'{k}_data'] = (fn.loc, fn.scale)

            return dist

        return get_dist


class SimFit:
    def __init__(self, fit: BaseFit):
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
        seed: int | None = None,
        parallel: bool = True,
        run_str: str | None = None,
        progress: bool = True,
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
        progress : bool, optional
            Whether to display progress bar. The default is True.

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

        sim_data = self._simulator.samples_from_one_set(params, n, seed)
        result = self._result_container(n)
        init_unconstr = self._to_unconstr(
            [params[i] for i in self._free_names]
        )
        init_unconstr = np.full((n, len(init_unconstr)), init_unconstr)

        if parallel:
            fitter = self._parallel_fitter
        else:
            fitter = self._sequence_fitter

        fit_result = fitter(sim_data, result, init_unconstr, run_str, progress)

        return self._make_result(sim_data, fit_result)

    def run_multi_sets(
        self,
        params: dict[str, jax.Array | np.ndarray],
        seed: int | None = None,
        parallel: bool = True,
        run_str: str | None = None,
        progress: bool = True,
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
        progress : bool, optional
            Whether to display progress bar. The default is True.

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

        sim_data = self._simulator.samples_from_multi_sets(params, seed)
        init_unconstr = jax.vmap(self._to_unconstr)(
            np.column_stack([params[i] for i in self._free_names])
        )
        result = self._result_container(len(init_unconstr))

        if parallel:
            fitter = self._parallel_fitter
        else:
            fitter = self._sequence_fitter

        fit_result = fitter(sim_data, result, init_unconstr, run_str, progress)

        return self._make_result(sim_data, fit_result)

    def fit_data(
        self,
        params: dict,
        data: dict,
        parallel: bool = True,
        run_str: str | None = None,
        progress: bool = True,
    ):
        """Fit the simulation data.

        Parameters
        ----------
        params : dict
            Parameters in constrained space.
        data : dict
            Simulation data based on `params`.
        parallel : bool, optional
            Whether to run the fit in parallel. The default is True.
        run_str : str, optional
            Description of the progress bar.
        progress : bool, optional
            Whether to display progress bar. The default is True.

        Returns
        -------
        results : dict
            The fit results.

        """
        p_in = set(params.keys())
        p_all = set(self._free_names)
        if p_in != p_all:
            raise ValueError(f'require params {p_in - p_all}')

        params = jax.tree_map(np.atleast_1d, params)
        data = jax.tree_map(np.atleast_2d, data)

        params_shape = list(jax.tree_map(np.shape, params).values)
        data_shape = list(jax.tree_map(np.shape, data).values)

        if any(i != params_shape[0] for i in params_shape):
            raise ValueError('params shape must be all equal')

        if any(i != data_shape[0] for i in data_shape):
            raise ValueError('data shape must be all equal')

        if params_shape[0] == 1:
            init_unconstr = self._to_unconstr(
                np.array(params[i] for i in self._free_names)
            )
            init_unconstr = np.full(
                (data_shape[0], len(init_unconstr)), init_unconstr
            )
        else:
            if params_shape[0] != data_shape[0]:
                raise ValueError('params and data size must be equal')

            init_unconstr = jax.vmap(self._to_unconstr)(
                np.column_stack([params[i] for i in self._free_names])
            )

        result = self._result_container(data_shape[0])

        if parallel:
            fitter = self._parallel_fitter
        else:
            fitter = self._sequence_fitter

        fit_result = fitter(data, result, init_unconstr, run_str, progress)

        return self._make_result(data, fit_result)

    def _make_result(self, sim_data, result):
        result['data'] = sim_data | self._net_counts(sim_data)
        valid = result.pop('valid')
        result = jax.tree_map(lambda x: x[valid], result)
        result['valid'] = valid
        result['n_valid'] = int(valid.sum())
        return result
