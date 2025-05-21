from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from emcee import EnsembleSampler as Sampler, State as EmceeState

from elisa.infer.samplers.ensemble.base import (
    EnsembleSampler,
    EnsembleSamplerState,
)

if TYPE_CHECKING:
    from multiprocessing import Queue

    from numpy.random import Generator


class EmceeSampler(EnsembleSampler):
    def get_sampling_fn(
        self,
        chains: int,
        warmup: int,
        steps: int,
        thinning: int,
        tune: bool | None,
        warmup_kwargs: dict,
        sampling_kwargs: dict,
    ):
        ndim = self._ndim
        if chains is None:
            chains = 4 * ndim
        if tune is None:
            tune = False
        log_prob_fn = self._log_prob_with_blobs
        blobs_dtype = self._blobs_dtype

        def sampling_fn(
            sampler_id: int, state: EnsembleSamplerState, queue: Queue
        ):
            emcee_state = EmceeState(
                coords=state.coords,
                random_state=state.random_state,
            )
            sampler1 = Sampler(
                chains,
                ndim,
                log_prob_fn,
                pool=None,
                args=None,
                kwargs=None,
                vectorize=True,
                blobs_dtype=blobs_dtype,
                parameter_names=None,
                **warmup_kwargs,
            )
            sampler2 = Sampler(
                chains,
                ndim,
                log_prob_fn,
                pool=None,
                args=None,
                kwargs=None,
                vectorize=True,
                blobs_dtype=blobs_dtype,
                parameter_names=None,
                **sampling_kwargs,
            )
            queue.put((sampler_id, 'warmup'))
            for s in sampler1.sample(
                emcee_state,
                iterations=warmup,
                tune=tune,
                store=False,
                progress=False,
            ):
                emcee_state = s
                queue.put((sampler_id, 'update'))
            queue.put((sampler_id, 'sample'))
            for s in sampler2.sample(
                emcee_state,
                iterations=steps,
                tune=tune,
                thin_by=thinning,
                store=True,
                progress=False,
            ):
                emcee_state = s
                queue.put((sampler_id, 'update'))
            queue.put((sampler_id, 'finish'))
            samples = sampler2.get_blobs()
            state = EnsembleSamplerState(
                coords=emcee_state.coords,
                random_state=emcee_state.random_state,
            )
            return samples, state

        return sampling_fn

    def get_random_state(self, seed: int) -> Generator:
        return np.random.default_rng(seed)
