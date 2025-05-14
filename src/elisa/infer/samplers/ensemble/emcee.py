from __future__ import annotations

import threading
import warnings
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import multiprocess as mp
import numpy as np
from emcee import EnsembleSampler, State
from numpyro.infer.initialization import init_to_value
from tqdm.auto import tqdm

from elisa.infer.samplers.util import get_model_info

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


class EmceeSampler:
    def __init__(
        self,
        numpyro_model: Callable,
        init_params: dict[str, float] | None = None,
        ignore_nan: bool = False,
        seed: int = 42,
        model_args: tuple = (),
        model_kwargs: dict | None = None,
    ):
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
        tune: bool = False,
        progress: bool = True,
        **kwargs: dict,
    ) -> dict[str, NDArray[float]]:
        pbars = {
            i: tqdm(
                total=warmup + steps,
                desc='Initializing... ',
                position=i,
                disable=not progress,
            )
            for i in range(1, n_parallel + 1)
        }

        ndim = self._ndim
        if chains is None:
            chains = 4 * ndim

        ss = np.random.SeedSequence(self._seed)
        seeds = ss.spawn(n_parallel)
        mcmc_rngs = list(map(np.random.default_rng, seeds))

        init = np.full((chains, ndim), self._init)

        log_prob_fn = self._log_prob_with_blobs
        blobs_dtype = self._blobs_dtype

        def run_sampler(sampler_id, init, rng, queue):
            jitter = rng.uniform(0.99, 1.01, size=(chains, ndim))
            init = init * jitter
            sampler = EnsembleSampler(
                chains,
                ndim,
                log_prob_fn,
                vectorize=True,
                blobs_dtype=blobs_dtype,
                **kwargs,
            )
            state = State(init, random_state=rng)
            queue.put((sampler_id, 'warmup'))
            for s in sampler.sample(
                state,
                iterations=warmup,
                tune=tune,
                store=False,
                progress=False,
            ):
                state = s
                queue.put((sampler_id, 'update'))
            queue.put((sampler_id, 'sample'))
            for _ in sampler.sample(
                state,
                iterations=steps,
                tune=tune,
                thin_by=thinning,
                store=True,
                progress=False,
            ):
                queue.put((sampler_id, 'update'))
            queue.put((sampler_id, 'finish'))
            return sampler.get_blobs()

        def progress_listener(queue, pbars, num_samplers):
            finished = 0
            while finished < num_samplers:
                sampler_id, msg = queue.get()
                if msg == 'update':
                    pbars[sampler_id].update(1)
                elif msg == 'warmup':
                    pbars[sampler_id].set_description(
                        f'Warming up sampler {sampler_id}'
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

        old_method = mp.get_start_method()
        if old_method != 'spawn':
            mp.set_start_method('spawn', force=True)
        else:
            old_method = ''

        queue = mp.Manager().Queue()
        listener_thread = threading.Thread(
            target=progress_listener, args=(queue, pbars, n_parallel)
        )
        listener_thread.start()

        with mp.Pool(processes=n_parallel) as pool:
            results = []
            for i in range(n_parallel):
                r = pool.apply_async(
                    run_sampler,
                    args=(i + 1, init, mcmc_rngs[i], queue),
                )
                results.append(r)
            results = [r.get() for r in results]

        listener_thread.join()

        if old_method:
            mp.set_start_method(old_method, force=True)

        # reshape results from (n_step, n_walker) to (n_walker, n_step)
        results = list(map(np.transpose, results))
        # merge chains from the same sampler into a single one
        results = list(map(np.concatenate, results))
        # stack results of different samplers
        results = np.vstack(results)
        return {i: results[i] for i in results.dtype.names}
