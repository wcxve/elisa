"""Handle model fitting."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..data.ogip import Data
from ..model.base import Model
from .likelihood import *

# [model_num^model]


class BaseFit(ABC):
    """Fitting context.

    Parameters
    ----------
    data : Data or list of Data
        The observation data.
    model : Model or list of Model
        The model used  to fit the data.
    stat : str or list of str
        The likelihood option for the data and model.
    seed : int

    """

    _stat_option: set[str] = {'chi2', 'cstat', 'pstat', 'pgstat', 'wstat'}

    def __init__(
        self,
        data: Data | list[Data],
        model: Model | list[Model],
        stat: str | list[str],
        seed: int = 42
    ):
        data, model, stat = self._sanity_check(data, model, stat)
        # self._data, self._model, self._stat
        self._seed = int(seed)

    @abstractmethod
    def fit(self, *args, **kwargs):
        ...

    @classmethod
    def stat_option(cls) -> set[str]:
        """List of available likelihood options."""
        return cls._stat_option

    def _sanity_check(self, data, model, stat):
        """Check if data, model, and stat are correct and return lists."""
        def get_list(inputs, name, itype, tname):
            """Check the model/data/stat, and return a list."""
            if isinstance(inputs, itype):
                input_list = [inputs]
            elif isinstance(inputs, (list, tuple)):
                if not inputs:
                    raise ValueError(f'{name} list is empty')
                if not all(map(lambda i: isinstance(i, itype), inputs)):
                    raise ValueError(f'all {name} must be {tname} instance')
                input_list = list(inputs)
            else:
                raise ValueError(f'got wrong type {type(inputs)} for {name}')
            return input_list

        data_list = get_list(data, 'data', Data, 'Data')
        model_list = get_list(model, 'model', Model, 'Model')
        stat_list = get_list(stat, 'stat', str, 'str')

        # check stat option
        flag = list(map(lambda i: (i in self._stat_option), stat_list))
        if not all(flag):
            unexpect = (j for i, j in enumerate(stat_list) if not flag[i])
            unexpect = ', '.join(f"'{i}'" for i in unexpect)
            supported = ', '.join(f"'{i}'" for i in self._stat_option)
            msg = f'got unexpected stat: {unexpect}; supported are {supported}'
            raise ValueError(msg)

        nd = len(data_list)
        nm = len(model_list)
        ns = len(stat_list)

        # check model number
        if nm == 1:
            model_list *= nd
        elif nm != nd:
            msg = f'number of model ({nm}) and data ({nd}) are not matched'
            raise ValueError(msg)

        # check stat number
        if ns == 1:
            stat_list *= nd
        elif ns != nd:
            msg = f'number of data ({nd}) and stat ({ns}) are not matched'
            raise ValueError(msg)

        return data_list, model_list, stat_list


class MaxLikeFit(BaseFit):
    def fit(self):
        ...


class BayesianFit(BaseFit):
    def fit(self):
        ...


if __name__ == '__main__':
    bayes = BayesianFit(model=[], data=[], stat=[])
    ml = MaxLikeFit(model=[], data=[], stat=[])

