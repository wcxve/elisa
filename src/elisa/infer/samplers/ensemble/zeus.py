from __future__ import annotations

from itertools import permutations
from typing import TYPE_CHECKING

import numpy as np
import zeus
import zeus.ensemble as zeus_ensemble
import zeus.moves as zeus_moves

from elisa.infer.samplers.ensemble.base import (
    EnsembleSampler,
    EnsembleSamplerState,
)

if TYPE_CHECKING:
    from multiprocessing import Queue

    from numpy.typing import NDArray


class ZeusSampler(EnsembleSampler):
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
        if tune is None:
            tune = True
        warmup_kwargs.setdefault('verbose', False)
        sampling_kwargs.setdefault('verbose', False)
        log_prob_fn = self._log_prob_with_blobs
        blobs_dtype = self._blobs_dtype

        def sampling_fn(
            sampler_id: int,
            state: EnsembleSamplerState,
            queue: Queue,
        ):
            np.random.set_state(state.random_state)
            start = state.coords
            sampler1 = zeus.EnsembleSampler(
                chains,
                ndim,
                log_prob_fn,
                args=None,
                kwargs=None,
                tune=tune,
                pool=None,
                vectorize=True,
                blobs_dtype=blobs_dtype,
                **warmup_kwargs,
            )
            sampler2 = zeus.EnsembleSampler(
                chains,
                ndim,
                log_prob_fn,
                args=None,
                kwargs=None,
                tune=tune,
                pool=None,
                vectorize=True,
                blobs_dtype=blobs_dtype,
                **sampling_kwargs,
            )
            queue.put((sampler_id, 'warmup'))
            for _ in sampler1.sample(
                start=start,
                iterations=warmup,
                progress=False,
            ):
                queue.put((sampler_id, 'update'))
            if warmup:
                start = sampler1.get_last_sample()
            sampler1.reset()
            queue.put((sampler_id, 'sample'))
            for _ in sampler2.sample(
                start=start,
                iterations=steps,
                thin_by=thinning,
                progress=False,
            ):
                queue.put((sampler_id, 'update'))
            queue.put((sampler_id, 'finish'))
            samples = sampler2.get_blobs()
            state = EnsembleSamplerState(
                coords=sampler2.get_last_sample(),
                random_state=np.random.get_state(legacy=True),
            )
            return samples, state

        return sampling_fn

    def get_random_state(self, seed: int) -> tuple:
        return np.random.RandomState(seed).get_state(legacy=True)


class DifferentialMove:
    """Improved DifferentialMove of zeus for reproducibility."""

    def __init__(self, tune=True, mu0=1.0):
        self.tune: bool = tune
        self.mu0: float = mu0
        self._nsamples: int | None = None
        self._perms: NDArray[int] | None = None
        self._nperms: int | None = None

    def get_direction(self, X: NDArray[float], mu: float):
        nsamples = X.shape[0]
        if nsamples != self._nsamples:
            self._nsamples = nsamples
            self._perms = np.array(list(permutations(np.arange(nsamples), 2)))
            self._nperms = len(self._perms)

        idx = np.random.choice(self._nperms, self._nsamples, replace=False)
        pairs = self._perms[idx].T

        if not self.tune:
            mu = self.mu0

        return 2.0 * mu * (X[pairs[0]] - X[pairs[1]]), self.tune


# Monkey patching the DifferentialMove for reproducibility
zeus_ensemble.DifferentialMove = DifferentialMove
zeus_moves.DifferentialMove = DifferentialMove
