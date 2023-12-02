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


        params = ...
        params_aux = ...

    def _generate_sample(self) -> Callable:
        samples = (
            m._model_info['site']['sample']
            for m in self._model.values()
        )

        samples = reduce(lambda i, j: i | j, samples)

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
            lambda i, j: i | j,
            deterministic
        )

        def deterministic_func(sample_dict):
            """Return numpyro deterministic site dict."""
            site = {}
            remains = list(deterministic.items())
            while remains:
                i = remains.pop(0)
                determ, (arg_names, func) = i
                all_site = sample_dict | site
                if all(arg_name in all_site for arg_name in arg_names):
                    args = (all_site[arg_name] for arg_name in arg_names)
                    site[determ] = numpyro.deterministic(determ, func(*args))
                else:
                    remains.append(i)

            return site

        return deterministic_func

    def _generate_stat_func_args(self) -> dict:
        args = {}
        for name, data in self._data.items():
            stat = self._stat[name]
            if stat == 'chi2':
                args[name] = (data.net_counts, data.net_error)
            elif stat == 'cstat':
                args[name] = (data.spec_counts,)
            elif stat == 'pstat':
                ratio = data.spec_effexpo / data.back_effexpo
                args[name] = (data.spec_counts, data.back_counts, ratio)
            elif stat == 'pgstat':
                ratio = data.spec_effexpo / data.back_effexpo
                args[name] = (
                    data.spec_counts, data.back_counts, data.back_error, ratio
                )
            elif stat == 'wstat':
                ratio = data.spec_effexpo / data.back_effexpo
                args[name] = (data.spec_counts, data.back_counts, ratio)

        return args

    def _generate_numpyro_model(self) -> Callable:
        sample_func = self._generate_sample()
        deterministic_func = self._generate_deterministic()

        names = list(self._data.keys())
        data = {}
        model_func = {}
        model_info = {}
        stat_func = {}
        for name in names:
            data[name] = self._data[name]
            model_func[name] = self._model[name]._wrapped_func
            model_info[name] = self._model[name]._model_info
            stat_func[name] = self._stat_options[self._stat[name]]
        func_args = self._generate_stat_func_args()

        def numpyro_model():
            """The numpyro model."""
            sample = sample_func()
            determ = deterministic_func(sample)
            params = sample | determ

            for name in names:
                d = data[name]
                ph_egird = d.ph_egrid
                resp_matrix = d.resp_matrix
                expo = d.spec_effexpo
                mfunc = model_func[name]
                loglike = stat_func[name]
                loglike_args = func_args[name]

                minfo = model_info[name]
                m_mapping = minfo['mapping']['name']
                p_mapping = minfo['params']
                params_i = {
                    m_mapping[m_id]: jax.tree_map(lambda k: params[k], p_dict)
                    for m_id, p_dict in p_mapping.items()
                }

                model_counts = mfunc(ph_egird, params_i) @ resp_matrix * expo
                loglike(name, model_counts, *loglike_args)

        return numpyro_model

    # @abstractmethod
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
                if not all(isinstance(i, itype) for i in inputs):
                    raise ValueError(f'all {name} must be {tname} instance')
                input_list = list(inputs)
            else:
                raise ValueError(f'got wrong type {type(inputs)} for {name}')
            return input_list

        data_list = get_list(data, 'data', Data, 'Data')
        model_list = get_list(model, 'model', Model, 'Model')
        stat_list = get_list(stat, 'stat', str, 'str')

        # check if model type is additive
        flag = list(i.type == 'add' for i in model_list)
        if not all(flag):
            err = (j for i, j in enumerate(model_list) if not flag[i])
            err = ', '.join(f"'{i}'" for i in err)
            msg = f'got models which are not additive type: {err}'
            raise ValueError(msg)

        # check stat option
        flag = list(i in self._stat_options for i in stat_list)
        if not all(flag):
            err = (j for i, j in enumerate(stat_list) if not flag[i])
            err = ', '.join(f"'{i}'" for i in err)
            supported = ', '.join(f"'{i}'" for i in self._stat_options)
            msg = f'got unexpected stat: {err}; supported are {supported}'
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

        # check if correctly using stat
        def check_data_stat(data, stat):
            """Check if data type and likelihood are matched."""
            name = data.name
            if stat != 'chi2' and not data.spec_poisson:
                stat_h = stat[:stat.index('stat')].upper()
                msg = f'Poisson data is required for using {stat_h}-statistics'
                raise ValueError(msg)

            if stat == 'cstat' and data.has_back:
                back = 'Poisson' if data.back_poisson else 'Gaussian'
                stat1 = 'W' if data.back_poisson else 'PG'
                stat2 = 'w' if data.back_poisson else 'pg'
                msg = 'C-statistics is not valid for Poisson data with '
                msg += f'{back} background, use {stat1}-statistics'
                msg += f'({stat2}stat) for {name} instead'
                raise ValueError(msg)

            elif stat == 'pstat' and not data.has_back:
                msg = 'Background is required for P-statistics'
                raise ValueError(msg)

            elif stat == 'pgstat' and not data.has_back:
                msg = 'Background is required for PG-statistics'
                raise ValueError(msg)

            elif stat == 'wstat':
                if not data.spec_poisson:
                    msg = 'Poisson data is required for W-statistics'
                    raise ValueError(msg)
                if not (data.has_back and data.back_poisson):
                    msg = 'Poisson background is required for W-statistics'
                    raise ValueError(msg)

        map(check_data_stat, zip(data_list, stat_list))

        return data_list, model_list, stat_list


class LikelihoodFit(BaseFit):
    def run(self):
        ...


class BayesianFit(BaseFit):
    def run(self):
        ...
