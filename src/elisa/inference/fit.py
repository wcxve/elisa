"""Handle model fitting."""
from __future__ import annotations

from abc import ABC, abstractmethod

# [model_num^model]


class FitContext(ABC):
    """Base fitting context."""
    def __init__(self, *args, **kwargs):
        ...

    @abstractmethod
    def fit(self, *args, **kwargs):
        ...


class MaxLikeFit(FitContext):
    """Maximum likelihood fitting context."""

    def fit(self):
        ...


class BayesianFit(FitContext):
    """Bayesian fitting context."""

    def fit(self):
        ...


if __name__ == '__main__':
    bayes = BayesianFit(model=[], data=[], stat=[])
    ml = MaxLikeFit(model=[], data=[], stat=[])

