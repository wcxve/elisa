"""Handle model fitting."""
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import reduce
from typing import Callable

import jax
import numpyro

from ..data.ogip import Data
from ..model.base import Model
from .likelihood import chi2, cstat, pstat, pgstat, wstat

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

    _stat_options = {
        'chi2': chi2,
        'cstat': cstat,
        'pstat': pstat,
        'pgstat': pgstat,
        'wstat': wstat
    }

    def __init__(
        self,
        data: Data | list[Data],
        model: Model | list[Model],
        stat: str | list[str],
        seed: int = 42
    ):
        data, model, stat = self._sanity_check(data, model, stat)

        self._data = {d.name: d for d in data}
        self._model = {d.name: m for d, m in zip(data, model)}
        self._stat = {d.name: s for d, s in zip(data, stat)}
        self._seed = int(seed)

    def _generate_sample(self) -> Callable:
        samples = (
            m._model_info['site']['sample']
            for m in self._model.values()
        )

        samples = reduce(lambda i, j: {**i, **j}, samples)

        def sample_func():
            """Return numpyro sample site dict."""
            site = {
                name: numpyro.sample(name, dist)
                for name, dist in samples.items()
            }
            return site

        return sample_func

    def _generate_deterministic(self) -> Callable:
        deterministic = (
            m._model_info['site']['deterministic']
            for m in self._model.values()
        )

        deterministic = reduce(
            lambda i, j: {**i, **j},
            deterministic
        )

        def deterministic_func(sample_dict):
            """Return numpyro deterministic site dict."""
            site = {}
            remains = list(deterministic.items())
            while remains:
                i = remains.pop(0)
                determ, (arg_names, func) = i
                all_site = {**sample_dict, **site}
                if all(arg_name in all_site for arg_name in arg_names):
                    args = (all_site[arg_name] for arg_name in arg_names)
                    site[determ] = func(*args)
                else:
                    remains.append(i)

            return site

        return deterministic_func

    @abstractmethod
    def run(self, *args, **kwargs):
        ...

    @classmethod
    def stat_options(cls) -> list:
        """List of available likelihood options."""
        return list(cls._stat_options.keys())

    def _sanity_check(
        self,
        data: Data | list[Data],
        model: Model | list[Model],
        stat: str | list[str]
    ):
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
        flag = list(map(lambda i: (i in self._stat_options), stat_list))
        if not all(flag):
            unexpect = (j for i, j in enumerate(stat_list) if not flag[i])
            unexpect = ', '.join(f"'{i}'" for i in unexpect)
            supported = ', '.join(f"'{i}'" for i in self._stat_options)
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


class LikelihoodFit(BaseFit):
    def run(self):
        ...


class BayesianFit(BaseFit):
    def run(self):
        ...
