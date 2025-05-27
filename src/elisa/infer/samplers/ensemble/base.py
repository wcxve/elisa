from __future__ import annotations

import threading
import warnings
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import multiprocess as mp
import numpy as np
from numpyro.infer.initialization import init_to_value
from tqdm.auto import tqdm

from elisa.infer.samplers.util import get_model_info

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from multiprocessing import Queue
    from typing import Any

    from numpy.typing import NDArray


class EnsembleSamplerState(NamedTuple):
    """Ensemble sampler state."""

    coords: NDArray[float]
    """The current positions of the walkers in the parameter space."""

    random_state: Any
    """The state of random number generator."""


class EnsembleSampler(metaclass=ABCMeta):
    def __init__(
        self,
        numpyro_model: Callable,
        init_params: dict[str, float] | None = None,
        ignore_nan: bool = False,
        seed: int = 42,
        model_args: tuple = (),
        model_kwargs: dict | None = None,
    ):
        """The base class for ensemble samplers.

        Parameters
        ----------
        numpyro_model : callable
            The numpyro model to sample from.
        init_params : dict, optional
            The initial parameters for the model.
        ignore_nan : bool, optional
            Whether to transform log probability of NaN to -1e300.
            The default is False.
        seed : int, optional
            The random seed for reproducibility.
        model_args : tuple, optional
            The arguments to pass to `numpyro_model`.
        model_kwargs : dict, optional
            The keyword arguments to pass to `numpyro_model`.
        """
        if ignore_nan:
            warnings.warn(
                'setting `ignore_nan` to True may fail to spot potential '
                'issues of the model',
                Warning,
            )

        if init_params is None:
            init_params = {}

        info = get_model_info(
            model=numpyro_model,
            init_strategy=init_to_value(values=init_params),
            model_args=model_args,
            model_kwargs=model_kwargs,
            validate_grad=False,
            rng_seed=seed,
        )

        ndim = info.ndim
        blobs_dtype = info.params_dtype + info.deterministic_dtype
        blobs_names = [dt[0] for dt in blobs_dtype]
        init = info.init_ravel
        unravel = info.unravel
        log_prob_fn = info.log_prob_fn
        postprocess_fn = info.postprocess_fn

        @jax.vmap
        @jax.jit
        def _log_prob_with_blobs(z):
            z = unravel(z)
            log_prob = log_prob_fn(z)
            blobs = postprocess_fn(z)
            return log_prob, [blobs[i] for i in blobs_names]

        @jax.jit
        def log_prob_with_blobs(z):
            log_prob, blobs = _log_prob_with_blobs(z)
            if ignore_nan:
                log_prob = jnp.nan_to_num(log_prob, nan=-np.inf)
            return [
                (log_prob[i], *(b[i] for b in blobs))
                for i in range(len(log_prob))
            ]

        self._ndim = ndim
        self._init = init
        self._log_prob_with_blobs = log_prob_with_blobs
        self._blobs_dtype = blobs_dtype
        self._seed = seed

    def run(
        self,
        warmup: int = 5000,
        steps: int = 5000,
        chains: int | None = None,
        thinning: int = 1,
        n_parallel: int = 4,
        tune: bool | None = None,
        progress: bool = True,
        states: Sequence[EnsembleSamplerState] | None = None,
        warmup_kwargs: dict | None = None,
        sampling_kwargs: dict | None = None,
    ) -> tuple[dict[str, NDArray[float]], tuple[EnsembleSamplerState, ...]]:
        """Run the sampler.

        Parameters
        ----------
        warmup : int, optional
            The warmup (burn-in) steps. The default is 5000.
        steps : int, optional
            The sampling steps. The default is 5000.
        chains : int, optional
            The number of walkers. The default is 4 * ndim.
        thinning : int
            Stores every `thinning` samples in the chain. When this is set,
            `steps` * `thinning` proposals will be made. The default is 1.
        n_parallel : int, optional
            Number of parallel samplers. The default is 4.
        tune : bool, optional
            Whether to tune the parameters of moves.
            Defaults to the corresponding sampler's default.
        progress : bool, optional
            Whether to display a progress bar. The default is True.
        states : sequence, optional
            The initial states of the samplers.
        warmup_kwargs: dict, optional
            Extra parameters passed to sampler constructor for warm-up phase.
        sampling_kwargs: dict | None = None,
            Extra parameters passed to sampler constructor for sampling phase.

        Returns
        -------
        samples : dict
            The posterior samples.
        states : tuple
            The states of the samplers.
        """
        ndim = self._ndim
        if chains is None:
            chains = 4 * ndim
        if states is None:
            seed0 = np.random.default_rng(self._seed).integers(2**32)
            seed_init, *seeds_mcmc = seed0 + np.arange(n_parallel + 1)
            rng_init = np.random.default_rng(seed_init)
            jitter = rng_init.uniform(0.9, 1.1, (n_parallel, chains, ndim))
            init = jitter * np.full((chains, ndim), self._init)
            states = [
                EnsembleSamplerState(i, self.get_random_state(j))
                for i, j in zip(init, seeds_mcmc, strict=True)
            ]
        else:
            states = list(states)
            if len(states) != n_parallel:
                raise ValueError('states number should match n_parallel')
            if not all(isinstance(s, EnsembleSamplerState) for s in states):
                raise ValueError('states must be EnsembleSamplerState')
            for s in states:
                if s.coords.shape != (chains, ndim):
                    raise ValueError(
                        f"states' coords must have shape ({chains}, {ndim})"
                    )
            warmup = 0

        if warmup_kwargs is None:
            warmup_kwargs = {}
        else:
            warmup_kwargs = dict(warmup_kwargs)

        if sampling_kwargs is None:
            sampling_kwargs = {}
        else:
            sampling_kwargs = dict(sampling_kwargs)

        old_method = mp.get_start_method()
        if old_method != 'spawn':
            mp.set_start_method('spawn', force=True)
        else:
            old_method = ''

        pbars = {
            i: tqdm(
                total=warmup + steps,
                desc='Initializing... ',
                position=i,
                disable=not progress,
            )
            for i in range(1, n_parallel + 1)
        }

        queue = mp.Manager().Queue()
        listener_thread = threading.Thread(
            target=self._progress_listener,
            args=(queue, pbars, n_parallel),
        )
        listener_thread.start()

        sampling_fn = self.get_sampling_fn(
            chains=chains,
            warmup=warmup,
            steps=steps,
            thinning=thinning,
            tune=tune,
            warmup_kwargs=warmup_kwargs,
            sampling_kwargs=sampling_kwargs,
        )

        with mp.Pool(processes=n_parallel) as pool:
            results = []
            for i in range(n_parallel):
                r = pool.apply_async(
                    sampling_fn,
                    args=(i + 1, states[i], queue),
                )
                results.append(r)
            results = [r.get() for r in results]
            pool.close()
            pool.join()

        listener_thread.join()

        if old_method:
            mp.set_start_method(old_method, force=True)

        # get samples
        samples = [r[0] for r in results]
        # reshape samples from (n_step, n_walker) to (n_walker, n_step)
        samples = list(map(np.transpose, samples))
        # merge chains from the same sampler into a single one
        samples = list(map(np.concatenate, samples))
        # stack samples of different samplers
        samples = np.vstack(samples)
        # make samples a dict
        samples = {i: samples[i] for i in samples.dtype.names}
        # get states
        states = tuple(r[1] for r in results)
        return samples, states

    @staticmethod
    def _progress_listener(
        queue: Queue,
        pbars: dict[int, tqdm],
        num_samplers: int,
    ):
        finished = 0
        while finished < num_samplers:
            sampler_id, msg = queue.get()
            if msg == 'update':
                pbars[sampler_id].update(1)
            elif msg == 'warmup':
                pbars[sampler_id].set_description(
                    f'Warm-up sampler {sampler_id}'
                )
            elif msg == 'sample':
                pbars[sampler_id].set_description(
                    f'Running sampler {sampler_id}',
                )
            elif msg == 'finish':
                pbars[sampler_id].set_description(
                    f'Finished sampler {sampler_id}',
                )
                pbars[sampler_id].close()
                finished += 1

    @abstractmethod
    def get_sampling_fn(
        self,
        chains: int,
        warmup: int,
        steps: int,
        thinning: int,
        tune: bool | None,
        warmup_kwargs: dict,
        sampling_kwargs: dict,
    ) -> Callable[
        [int, EnsembleSamplerState, Queue],
        tuple[NDArray, EnsembleSamplerState],
    ]:
        """Generate the sampling function."""
        pass

    @abstractmethod
    def get_random_state(self, seed: int) -> Any:
        """Get the random state for the sampler."""
        pass
