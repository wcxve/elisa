import os
import json
from json import JSONEncoder

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import scienceplots
import xarray as xr
from .data import Data
from .likelihood import chi, cstat, pstat, pgstat, wstat, fpstat
from .model.base import SpectralModel
from .plot import (
    _plot_data, _plot_ldata,
    _plot_ufspec, _plot_eufspec, _plot_eeufspec,
    _plot_icounts, _plot_icounts_residual,
    _plot_residuals, _plot_delchi, _plot_ratio,
    _plot_deviance, _plot_sign_deviance, _plot_sign_deviance2
)
from .plot import plot_corner, plot_ppc
from .sampling import sample_numpyro_nuts, sample_posterior_predictive
from iminuit import Minuit
from jacobi import propagate
from numdifftools import Gradient, Jacobian, Hessian
from pytensor import function
from scipy import stats
from scipy.optimize import minimize
from tqdm import tqdm


class Infer:
    stat_options = ('chi', 'cstat', 'pstat', 'pgstat', 'wstat', 'fpstat')
    _stat_func = {
        'chi': chi,
        'cstat': cstat,
        'pstat': pstat,
        'pgstat': pgstat,
        'wstat': wstat,
        'fpstat': fpstat
    }
    def __init__(self, data, model, stat, seed=42):
        data_list, model_list, stat_list = self._check_input(data, model, stat)

        self.seed = seed

        self._mle_result = None
        self._idata_boot = None
        self._boot_result = None
        self._idata = None
        self._ppc_result = None

        self.__values_to_root = None
        self.__values_to_rv = None
        self.__root_to_values = None
        self.__root_to_rv = None
        self.__rv_to_params = None
        self.__deviance = None  # i.e. -2 * lnL
        self.__deviance_group = None  # deviance for each spectral data
        self.__deviance_point = None  # deviance for all channel
        self.__lnL = None
        self.__values_covar = None
        self.__values_robust_covar = None

        self._pymc_model = pm.Model()

        self._data_name = tuple(data.name for data in data_list)
        self._data = {}
        for data in data_list:
            if data.name in self._data:
                raise ValueError(f'data name "{data.name}" is duplicate')
            else:
                self._data[data.name] = data

        self._model = {}
        self._stat = {}
        for name, model, stat in zip(self._data_name, model_list, stat_list):
            stat_func = self._stat_func[stat]
            stat_func(self._data[name], model, self._pymc_model)
            self._model[name] = model
            self._stat[name] = stat

        self._root = self._get_root()
        for root, name, default in zip(self._root['root'],
                                       self._root['name'],
                                       self._root['default']):
            self._pymc_model.register_rv(root, name, initval=default)

        self._rv = self._get_rv()
        for rv, name in zip(self._rv['rv'], self._rv['name']):
            pm.Deterministic(name, rv, self._pymc_model)

        self._values = self._get_values()

        self._params = {}
        for name in self._data_name:
            params = self._model[name].params
            self._params[name] = {
                'name': params['name'],
                'default': np.array(params['default']),
                'frozen': np.array(params['frozen']),
                'rv_mask': np.isin(self._rv['name'], params['name'])
            }

        dof = 0
        for d in self._data.values():
            dof += d.channel.size
        dof -= len(self._root['root'])
        self._dof = dof

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        if self._idata:
            az.to_netcdf(self._idata, f'{path}/idata.nc')

        if self._idata_boot:
            az.to_netcdf(self._idata, f'{path}/idata_boot.nc')

        if self._ppc_result:
            dic = {}
            for k, v in self._ppc_result.items():
                if isinstance(v, xr.Dataset):
                    v.to_netcdf(f'{path}/ppc_{k}.nc')
                else:
                    dic.update({k: v})

            with open(f'{path}/ppc.json', 'w') as f:
                json.dump(dic, f, indent=4, cls=NumpyArrayEncoder)

        if self._boot_result:
            dic = {}
            for k, v in self._boot_result.items():
                if isinstance(v, xr.Dataset):
                    v.to_netcdf(f'{path}/boot_{k}.nc', )
                else:
                    dic.update({k: v})

            with open(f'{path}/boot.json', 'w') as f:
                json.dump(dic, f, indent=4, cls=NumpyArrayEncoder)

    def restore(self, path):
        idata_path = f'{path}/idata.nc'
        if os.path.exists(idata_path):
            self._idata = az.from_netcdf(idata_path)

        idata_boot_path = f'{path}/idata_boot.nc'
        if os.path.exists(idata_boot_path):
            self._idata_boot = az.from_netcdf(idata_boot_path)

        ppc_path = f'{path}/ppc.json'
        if os.path.exists(ppc_path):
            with open(ppc_path) as file:
                self._ppc_result = json.load(file)

            self._ppc_result['data'] = xr.open_dataset(f'{path}/ppc_data.nc')
            self._ppc_result['fitted_rv'] = xr.open_dataset(f'{path}/ppc_fitted_rv.nc')
            self._ppc_result['init_rv'] = xr.open_dataset(f'{path}/ppc_init_rv.nc')

        boot_path = f'{path}/boot.json'
        if os.path.exists(boot_path):
            with open(boot_path) as file:
                self._boot_result = json.load(file)

            self._boot_result['data'] = xr.open_dataset(f'{path}/boot_data.nc')
            self._boot_result['init_rv'] = xr.open_dataset(f'{path}/boot_init_rv.nc')
            self._boot_result['fitted_rv'] = xr.open_dataset(f'{path}/boot_fitted_rv.nc')

    def mle(self, init=None):
        if init is None:
            init = self._root_to_values(self._root['default'])
        else:
            raise NotImplementedError(
                'setting initial values is not supported yet'
            )

        res = minimize(self._deviance, init, method='L-BFGS-B')
        if res.success and res.fun/self._dof <= 1.5:
            init = res.x

        m = Minuit(self._deviance,
                   init,
                   grad=Gradient(self._deviance),
                   name=[v.name for v in self._values])
        m.strategy = 0
        m.migrad()
        max_it = 10
        nit = 0
        while (not m.fmin.is_valid) and nit < max_it:
            m.hesse()
            m.migrad()
            nit += 1

        # m.minos()
        self._mle_result = {
            'minuit': m,
            'values': np.asarray(m.values),
            'root': self._values_to_root(m.values),
            'rv': self._values_to_rv(m.values),
            'deviance': {
                'total': m.fmin.fval,
                'group': self._deviance_group(m.values),
                'point': self._deviance_point(m.values)
            }
        }

        # net_counts = []
        # model_counts = []
        # for name in self._data_name:
        #     net_counts.extend(self._data[name].net_counts)
        #
        #     pars = self._rv_to_params(self._values_to_rv(self._mle_result['values']), name)
        #     data = self._data[name]
        #     model_counts.extend(self._model[name].counts(
        #         pars, data.ph_ebins, data.ch_emin, data.ch_emax,
        #         data.resp_matrix, data.spec_exposure
        #     ))
        # edf = EDFstat(np.array(net_counts), np.array(model_counts))

        k = m.npar
        n = self._dof + m.npar

        mle = self._mle_result['rv']
        try:
            se = self.se()
        except:
            se = {}
            for name in self._get_pars_name(pars=None):
                for i in range(len(mle)):
                    if name == self._rv['name'][i]:
                        se[name] = (mle[i], np.nan)

        mle_info = {
            'pars': se,
            'stat': m.fmin.fval,
            'dof': self._dof,
            # 'gof': stats.chi2.sf(m.fmin.fval, self._dof),
            # 'edf': edf,
            'aic': m.fmin.fval + 2 * k + 2 * k * (k + 1) / (n - k - 1),
            'bic': m.fmin.fval + k * np.log(n),
            'valid': m.fmin.is_valid,
            # 'edm': m.fmin.edm
        }

        self._mle_result.update(mle_info)

        return mle_info

    def se(self, pars=None, robust=False):
        if self._mle_result is None:
            raise RuntimeError(
                'MLE must be performed before calculating standard error'
            )

        mle_ = self._mle_result['values']
        if robust:
            cov_ = self._values_robust_covar(mle_)
        else:
            cov_ = self._values_covar(mle_)

        mle, cov = propagate(self._values_to_rv, mle_, cov_, method=0)
        err = np.sqrt(np.diag(cov))
        se_dict = {}
        for name in self._get_pars_name(pars):
            for i in range(len(mle)):
                if name == self._rv['name'][i]:
                    se_dict[name] = (mle[i], err[i])

        return se_dict

    # def parametric_bootstrap(self, n=1000):
    #     if self._mle_result is None:
    #         self.mle()
    #
    #     self._stat_obs = np.hstack([
    #         self._deviance(self._mle_result['values']),
    #         self._deviance_group(self._mle_result['values'])
    #     ])
    #     self._stat_p_obs = self._deviance_point(self._mle_result['values'])
    #
    #     mle_ = self._values_to_root(self._mle_result['values'])
    #     mle_dataset = xr.Dataset(
    #         data_vars={
    #             name: (('chain', 'draw'), np.full((1, n), mle_[i]))
    #             for i, name in enumerate(self._root['name'])
    #         },
    #         coords={
    #             'chain': ('chain', [0]),
    #             'draw': ('draw', np.arange(n))
    #         }
    #     )
    #     idata = az.InferenceData(posterior=mle_dataset)
    #     data_boot = pm.sample_posterior_predictive(
    #         trace=idata,
    #         model=self._pymc_model,
    #         random_seed=self.seed,
    #         progressbar=False,
    #         extend_inferencedata=False
    #     )
    #     bdata = {
    #         k.replace('_Non', '_spec_counts')
    #         .replace('_Noff', '_back_counts'): v.values[0]
    #         # [0] is because there is only one chain in idata
    #         for k, v in data_boot['posterior_predictive'].items()
    #     }
    #
    #     # extract net counts from bootstrap samples
    #     net_boot = []
    #     for name, data in self._data.items():
    #         counts = data_boot['posterior_predictive'][name + '_Non']
    #         if data.has_back and self._stat[name] != 'chi':
    #             back_counts = data_boot['posterior_predictive'][name + '_Noff']
    #             factor = data.spec_exposure / data.back_exposure
    #             counts = counts - factor * back_counts
    #         data_boot['posterior_predictive'][name + '_Net'] = counts
    #         net_boot.append(counts)
    #
    #     data_boot['posterior_predictive']['all_channel'] = (
    #         ('chain', 'draw', 'channel'),
    #         np.concatenate(net_boot, axis=-1)
    #     )
    #
    #     # net_counts = []
    #     # model_counts = []
    #     # for name in self._data_name:
    #     #     net_counts.extend(self._data[name].net_counts)
    #     #
    #     #     pars = self._rv_to_params(self._values_to_rv(self._mle_result['values']), name)
    #     #     data = self._data[name]
    #     #     model_counts.extend(self._model[name].counts(
    #     #         pars, data.ph_ebins, data.ch_emin, data.ch_emax,
    #     #         data.resp_matrix, data.spec_exposure
    #     #     ))
    #     # edf = EDFstat(np.array(net_counts), np.array(model_counts))
    #
    #     values_boot = np.empty((n, len(self._mle_result['values'])))
    #     stat_a_boot = np.empty(n)
    #     stat_g_boot = np.empty((n, len(self._data)))
    #     stat_p_boot = []
    #     flag = np.full(n, True)
    #     try:
    #         for i in tqdm(range(n), desc='Bootstrap'):
    #             pm.set_data({d: bdata[d][i] for d in bdata}, self._pymc_model)
    #             opt_res = minimize(fun=self._deviance,
    #                                x0=self._mle_result['values'],
    #                                method='L-BFGS-B',
    #                                jac='2-point')
    #
    #             values_boot[i] = opt_res.x
    #             stat_a_boot[i] = opt_res.fun
    #             stat_g_boot[i] = self._deviance_group(opt_res.x)
    #             stat_p_boot.append(self._deviance_point(opt_res.x))
    #             if not opt_res.success:
    #                 flag[i] = False
    #     except Exception as e:
    #         raise e
    #     finally:
    #         observed = {
    #             k.replace('_Non', '_spec_counts')
    #             .replace('_Noff', '_back_counts'): v.values
    #             for k, v in data_boot['observed_data'].items()
    #         }
    #         pm.set_data(observed, self._pymc_model)
    #
    #     pars_boot = self._values_to_rv(values_boot[flag])
    #
    #     # net_boot = np.column_stack([
    #     #     data_boot['posterior_predictive'][i + '_Net'].values[0]
    #     #     for i in self._data_name
    #     # ])[flag]
    #     #
    #     # CE_boot = np.column_stack([
    #     #     self._model[i].counts(self._rv_to_params(pars_boot, i),
    #     #                           self._data[i].ph_ebins,
    #     #                           self._data[i].ch_emin,
    #     #                           self._data[i].ch_emax,
    #     #                           self._data[i].resp_matrix,
    #     #                           self._data[i].spec_exposure)
    #     #     for i in self._data_name
    #     # ])
    #     # edf_boot = EDFstat(net_boot, CE_boot)
    #     # edf_pvalue = np.sum(edf_boot >= mle_res['edf'], axis=0) / nboot_valid
    #
    #     stat_boot = np.column_stack([stat_a_boot, stat_g_boot])[flag]
    #     stat_p_boot = [
    #         np.array([boot[n] for boot in stat_p_boot])[flag]
    #         for n in range(len(self._data))
    #     ]
    #
    #     nboot_valid = flag.sum()
    #     pvalue_boot = np.sum(stat_boot >= self._stat_obs, axis=0) / nboot_valid
    #     pvalue_p_boot = [
    #         np.sum(boot >= obs, axis=0) / nboot_valid
    #         for obs, boot in zip(self._stat_p_obs, stat_p_boot)
    #     ]
    #
    #     pars_boot = xr.Dataset(
    #         data_vars={
    #             name: (
    #                 ('chain', 'draw'), np.expand_dims(pars_boot[:, i], axis=0))
    #             for i, name in enumerate(self._rv['name'])
    #         },
    #         coords={
    #             'chain': ('chain', [0]),
    #             'draw': ('draw', np.arange(flag.sum()))
    #         }
    #     )
    #
    #     draws = np.flatnonzero(flag)
    #     self._data_boot = data_boot['posterior_predictive'].where(
    #         data_boot['posterior_predictive']['draw'].isin(draws),
    #         drop=True
    #     )
    #     self._pars_boot = pars_boot
    #     self._stat_boot = stat_boot
    #     self._pvalue_boot = pvalue_boot
    #     self._stat_p_boot = stat_p_boot
    #     self._pvalue_p_boot = pvalue_p_boot
    #     self._nboot = n

    def bootstrap(self, n=1000):
        if self._mle_result is None:
            self.mle()

        if self._boot_result is None or self._boot_result['n'] != n:
            mle_root = self._mle_result['root']
            root = {
                name: (('chain', 'draw'), np.full((1, n), mle_root[i]))
                for i, name in enumerate(self._root['name'])
            }
            mle_rv = self._mle_result['rv']
            rv = {
                name: (('chain', 'draw'), np.full((1, n), mle_rv[i]))
                for i, name in enumerate(self._rv['name'])
            }

            mle_dataset = xr.Dataset(
                data_vars=dict(**root, **rv),
                coords={
                    'chain': ('chain', np.arange(1)),
                    'draw': ('draw', np.arange(n))
                }
            )

            self._idata_boot = az.InferenceData(posterior=mle_dataset)

            self._boot_result = self._simulate_and_fit(self._idata_boot, n, True)

    def ppc(self, n=1000):
        if self._idata is None:
            raise ValueError('run MCMC before ppc')

        if self._ppc_result is None or self._ppc_result['n'] != n:
            self._ppc_result = self._simulate_and_fit(self._idata, n, False)

    def _simulate_and_fit(self, idata, n, is_bootstrap):
        if is_bootstrap:  # parametric bootstrap
            pm.sample_posterior_predictive(
                trace=idata,
                model=self._pymc_model,
                random_seed=self.seed,
                progressbar=False,
                extend_inferencedata=True
            )

            # calculate net counts
            sim_net = []
            channel = []
            pred = idata['posterior_predictive']
            for name, data in self._data.items():
                counts = pred[f'{name}_Non']

                if f'{name}_Noff' in pred:
                    alpha = data.spec_exposure / data.back_exposure
                    counts = counts - alpha * pred[f'{name}_Noff']

                idata['posterior_predictive'][f'{name}_Net'] = counts
                sim_net.append(counts)
                channel.extend(data.channel)

            idata['posterior_predictive']['all_channel'] = (
                ('chain', 'draw', 'channel'),
                np.concatenate(sim_net, axis=2)
            )

            idata = idata.assign_coords({'channel': channel})

            init_values = np.tile(self._mle_result['values'], (n, 1))

            pred = idata['posterior_predictive']
            data = {}
            for i in self._data_name:
                data[f'{i}_spec_counts'] = pred[f'{i}_Non'].values[0]
                if f'{i}_Noff' in pred.keys():
                    data[f'{i}_back_counts'] = pred[f'{i}_Noff'].values[0]

            chain_idx = np.zeros(n, dtype=np.int64)
            draw_idx = np.arange(n, dtype=np.int64)

        else:  # posterior prediction
            # pymc has performance issue on posterior prediction,
            # so a custom implementation is used here
            if 'posterior_predictive' not in idata.groups():
                sample_posterior_predictive(idata, self._stat, self.seed)

            post = idata['posterior']
            pred = idata['posterior_predictive']
            n_chain = pred.chain.size
            n_draw = pred.draw.size

            rng = np.random.default_rng(self.seed)
            chain_idx = rng.integers(0, n_chain, n)
            draw_idx = rng.integers(0, n_draw, n)

            init_root = [post[r].values[chain_idx, draw_idx] for r in self._root['name']]
            init_root = np.ascontiguousarray(np.transpose(init_root))
            init_values = self._root_to_values(init_root)

            data = {}
            for i in self._data_name:
                data[f'{i}_spec_counts'] = pred[f'{i}_Non'].values[chain_idx, draw_idx]
                if f'{i}_Noff' in pred.keys():
                    data[f'{i}_back_counts'] = pred[f'{i}_Noff'].values[chain_idx, draw_idx]

        fitted_values = np.empty_like(init_values)
        deviance_total = np.empty(n)
        deviance_group = np.empty((n, len(self._data_name)))
        deviance_point = []
        flag = np.full(n, True)

        try:
            for i in tqdm(range(n), desc='Fit'):
                pm.set_data({k: v[i] for k, v in data.items()}, self._pymc_model)

                fit_result = minimize(fun=self._deviance,
                                      x0=init_values[i],
                                      method='L-BFGS-B',
                                      jac='2-point')

                if fit_result.success:
                    fitted_values[i] = fit_result.x
                    deviance_total[i] = fit_result.fun
                    deviance_group[i] = self._deviance_group(fit_result.x)
                    deviance_point.append(self._deviance_point(fit_result.x))
                else:
                    flag[i] = False

        except Exception as e:
            raise e

        finally:
            data_obs = {}
            observed = idata['observed_data']
            for i in self._data_name:
                data_obs[f'{i}_spec_counts'] = observed[f'{i}_Non'].values
                if f'{i}_Noff' in observed:
                    data_obs[f'{i}_back_counts'] = observed[f'{i}_Noff'].values

            pm.set_data(data_obs, self._pymc_model)

        init_rv = self._values_to_rv(init_values[flag])
        fitted_rv = self._values_to_rv(fitted_values[flag])
        fitted_root = self._values_to_root(fitted_values[flag])
        deviance_total = deviance_total[flag]
        deviance_group = deviance_group[flag]
        deviance_point = [
            np.array([boot[i] for boot in deviance_point])
            for i in range(len(self._data_name))
        ]

        deviance_total_obs = self._mle_result['deviance']['total']
        deviance_group_obs = self._mle_result['deviance']['group']
        deviance_point_obs = self._mle_result['deviance']['point']

        n_valid = int(flag.sum())
        p_value_total = np.sum(deviance_total >= deviance_total_obs) / n_valid
        p_value_group = np.sum(deviance_group >= deviance_group_obs, axis=0) / n_valid
        p_value_point = [
            np.sum(sim >= obs, axis=0) / n_valid
            for obs, sim in zip(deviance_point_obs, deviance_point)
        ]

        valid_idx = np.flatnonzero(flag)
        coords = {
            'chain': ('chain', [0]),
            'draw': ('draw', valid_idx)
        }

        init_rv = np.expand_dims(init_rv, axis=0)
        init_rv = xr.Dataset(
            data_vars={
                name: (('chain', 'draw'), init_rv[..., i])
                for i, name in enumerate(self._rv['name'])
            },
            coords=coords
        )

        fitted_rv = np.expand_dims(fitted_rv, axis=0)
        fitted_root = np.expand_dims(fitted_root, axis=0)
        data_vars = {
            name: (('chain', 'draw'), fitted_rv[..., i])
            for i, name in enumerate(self._rv['name'])
        }
        data_vars.update({
            name: (('chain', 'draw'), fitted_root[..., i])
            for i, name in enumerate(self._root['name'])
        })
        fitted_rv = xr.Dataset(data_vars=data_vars, coords=coords)

        coords.update({
            k: v
            for k, v in idata['posterior_predictive'].coords.items()
            if k not in ('chain', 'draw')
        })
        sim_data = xr.Dataset(
            data_vars={
                k: (v.dims, v.values[None, chain_idx[flag], draw_idx[flag]])
                for k, v in idata['posterior_predictive'].items()
            },
            coords=coords
        )

        result = {
            'data': sim_data,
            'init_rv': init_rv,
            'fitted_rv': fitted_rv,
            'deviance': {
                'total': deviance_total,
                'group': deviance_group,
                'point': deviance_point
            },
            'p_value': {
                'total': p_value_total,
                'group': p_value_group,
                'point': p_value_point
            },
            'n': n,
            'n_valid': n_valid
        }

        return result

    # def gof(self, nsim=1000, cl=0.95):
    #     if self._idata is None:
    #         raise ValueError('run MCMC before ppc')
    #
    #     idata = self._idata
    #
    #     if 'posterior_predictive' not in self._idata.groups():
    #         sample_posterior_predictive(idata, self._stat, self.seed)
    #         # pm.sample_posterior_predictive(
    #         #     trace=idata,
    #         #     model=self._pymc_model,
    #         #     random_seed=self.seed,
    #         #     progressbar=True,
    #         #     extend_inferencedata=True
    #         # )
    #         # net_rep = []
    #         # for name, data in self._data.items():
    #         #     counts = idata['posterior_predictive'][name+'_Non']
    #         #     if data.has_back and self._stat[name] != 'chi':
    #         #         back_counts = idata['posterior_predictive'][name+'_Noff']
    #         #         factor = data.spec_exposure / data.back_exposure
    #         #         counts = counts - factor * back_counts
    #         #     idata['posterior_predictive'][name+'_Net'] = counts
    #         #     net_rep.append(counts)
    #         #
    #         # idata['posterior_predictive']['all_channel'] = (
    #         #     ('chain', 'draw', 'channel'),
    #         #     np.concatenate(net_rep, axis=-1)
    #         # )
    #
    #     posterior = idata['posterior']
    #     predictive = idata['posterior_predictive']
    #
    #     rng = np.random.default_rng(self.seed)
    #     idx = rng.integers(low=[[0], [0]],
    #                        high=[[posterior.chain.size],
    #                              [posterior.draw.size]],
    #                        size=(2, nsim))
    #     i, j = idx
    #
    #     pdata = {
    #         f'{name}_{d2}_counts': predictive[f'{name}_N{d1}'].values[i, j]
    #         for name in self._data_name
    #         for d1, d2 in zip(['on', 'off'], ['spec', 'back'])
    #         if f'{name}_N{d1}' in predictive.keys()
    #     }
    #     pred_net = predictive['all_channel'].values[i, j]
    #
    #     post = [posterior[r].values[i, j] for r in self._root['name']]
    #     post = np.ascontiguousarray(np.transpose(post))
    #
    #     D = -2.0 * idata.log_likelihood['total'].values[i, j]
    #     D_rep = np.empty(nsim)
    #     D_min = np.empty(nsim)
    #
    #     # net_counts = np.hstack([self._data[i].net_counts for i in self._data_name])
    #     # EDF = np.empty((nsim, 4))
    #     # EDF_rep = np.empty((nsim, 4))
    #     # EDF_min = np.empty((nsim, 4))
    #
    #     flag = np.full(nsim, True)
    #     y_hat = {
    #         name: np.empty((nsim, data.channel.size))
    #         for name, data in self._data.items()
    #     }
    #     try:
    #         for i in tqdm(range(nsim), desc='Fitting'):
    #             pm.set_data({d: pdata[d][i] for d in pdata}, self._pymc_model)
    #             D_rep[i] = self._deviance(self._root_to_values(post[i]))
    #             opt_res = minimize(fun=self._deviance,
    #                                x0=self._root_to_values(post[i]),
    #                                method='L-BFGS-B',
    #                                jac='2-point')
    #             print(self._deviance(self._root_to_values(post[i])), opt_res.fun,
    #                   opt_res.nfev)
    #             D_min[i] = opt_res.fun
    #
    #             # model_counts = []
    #             # model_fit_counts = []
    #             # for name in self._data_name:
    #             #     pars = self._rv_to_params(self._root_to_rv(post[i]), name)
    #             #     pars_fit = self._rv_to_params(self._values_to_rv(opt_res.x), name)
    #             #     data = self._data[name]
    #             #     model_counts_i = self._model[name].counts(
    #             #         pars, data.ph_ebins, data.ch_emin, data.ch_emax,
    #             #         data.resp_matrix, data.spec_exposure
    #             #     )
    #             #     y_hat[name][i] = model_counts_i
    #             #     model_fit_counts_i = self._model[name].counts(
    #             #         pars_fit, data.ph_ebins, data.ch_emin, data.ch_emax,
    #             #         data.resp_matrix, data.spec_exposure
    #             #     )
    #             #     model_counts.extend(model_counts_i)
    #             #     model_fit_counts.extend(model_fit_counts_i)
    #             #
    #             # model_counts = np.array(model_counts)
    #             # model_fit_counts = np.array(model_fit_counts)
    #             # rep_counts = pred_net[i]
    #             #
    #             # EDF[i] = EDFstat(net_counts, model_counts)
    #             # EDF_rep[i] = EDFstat(rep_counts, model_counts)
    #             # EDF_min[i] = EDFstat(rep_counts, model_fit_counts)
    #
    #             if not opt_res.success:
    #                 flag[i] = False
    #
    #         D_min = D_min[flag]
    #         # EDF_min = EDF_min[flag]
    #
    #     except Exception as e:
    #         raise e
    #     finally:
    #         observed = {
    #             f'{i}_{k}_counts': idata['observed_data'][f'{i}_N{j}'].values
    #             for i in self._data_name
    #             for j, k in zip(['on', 'off'], ['spec', 'back'])
    #             if f'{i}_N{j}' in idata['observed_data'].keys()
    #         }
    #         pm.set_data(observed, self._pymc_model)
    #
    #     if self._mle_result is None:
    #         self.mle()
    #
    #     D_best = self._mle_result['stat']
    #     # EDF_best = mle_res['edf']
    #
    #     x = []
    #     y = []
    #     y_rep = []
    #     y_hat_list = []
    #     i, j = idx
    #     for name, data in self._data.items():
    #         x.append(np.column_stack([data.ch_emin, data.ch_emax]))
    #         y.append(data.net_counts)
    #         y_rep.append(predictive[f'{name}_Net'].values[i, j])
    #         y_hat_list.append(y_hat[name])
    #
    #     res = plot_ppc(
    #         x, y, y_hat_list, D_best, self._dof, y_rep, D, D_rep, D_min, '$D$',
    #         cl=cl, xlabel='Energy [keV]', colors=None
    #     )
    #
    #     # for i, stat in enumerate(['ks', 'cvm', 'ad', 'cusum']):
    #     #     res_ = plot_ppc(
    #     #         x, y, EDF_best[i], self._dof, y_rep, EDF[:, i], EDF_rep[:, i], EDF_min[:, i], stat,
    #     #         cl=cl, xlabel='Energy [keV]', colors=None
    #     #     )
    #
    #     az.plot_loo_pit(idata, 'all_channel', ecdf=True, hdi_prob=cl)
    #
    #     return res

    def ci(self, method='quantile', cl=1.0, pars=None, nboot=1000):
        pars_name = self._get_pars_name(pars)

        if self._mle_result is None:
            self.mle()

        if cl >= 1.0:
            cl = 1.0 - stats.norm.sf(cl) * 2.0

        m = self._mle_result['minuit']
        mle_ = self._mle_result['values']
        mle = self._values_to_rv(mle_)

        if method == 'quantile':
            if self._idata is None:
                raise ValueError(
                    'run MCMC before calculating credible interval'
                )
            ci_ = self._idata['posterior'].quantile(
                q=[0.5 - cl / 2.0, 0.5, 0.5 + cl / 2.0],
                dim=['chain', 'draw']
            )
            lower_ = ci_.sel(quantile=0.5 - cl / 2.0)
            lower = [lower_[i].values for i in self._rv['name']]
            upper_ = ci_.sel(quantile=0.5 + cl / 2.0)
            upper = [upper_[i].values for i in self._rv['name']]
            mle_ = ci_.sel(quantile=0.5)
            mle = [float(mle_[i].values) for i in self._rv['name']]

        elif method == 'hdi':
            if self._idata is None:
                raise ValueError(
                    'run MCMC before calculating credible interval'
                )

            hdi = az.hdi(self._idata, cl)
            lower_ = hdi.sel(hdi='lower')
            lower = [lower_[i].values for i in self._rv['name']]
            upper_ = hdi.sel(hdi='higher')
            upper = [upper_[i].values for i in self._rv['name']]

        elif method == 'se' or method == 'rse':
            sigma = stats.norm.isf(0.5 - cl / 2.0)

            if method == 'rse':
                cov_ = self._values_robust_covar(mle_)
            else:
                cov_ = self._values_covar(mle_)
            err_ = np.sqrt(np.diagonal(cov_))

            lower_ = []
            upper_ = []
            for i in range(len(mle_)):
                lower_.append(mle_[i] - sigma*err_[i])
                upper_.append(mle_[i] + sigma*err_[i])

            lower = self._values_to_rv(lower_)
            upper = self._values_to_rv(upper_)

        elif method == 'profile':
            m.minos(cl=cl)
            self._mle_result['values'] = np.array(m.values)

            mle_ = self._mle_result['values']
            lower_ = []
            upper_ = []
            for i in range(len(mle_)):
                merror = m.merrors[i]
                if merror.lower_valid:
                    lower_.append(mle_[i] + merror.lower)
                else:
                    lower_.append(np.nan)

                if merror.upper_valid:
                    upper_.append(mle_[i] + merror.upper)
                else:
                    upper_.append(np.nan)

            mle = self._values_to_rv(mle_)
            lower = self._values_to_rv(lower_)
            upper = self._values_to_rv(upper_)

        elif method == 'boot':
            if self._boot_result is None:
                self.bootstrap(nboot)

            # bci = az.hdi(self._pars_boot, cl)
            # lower_ = bci.sel(hdi='lower')
            # lower_ = [lower_[i].values for i in self._rv['name']]
            # upper_ = bci.sel(hdi='higher')
            # upper_ = [upper_[i].values for i in self._rv['name']]
            # lower = lower_
            # upper = upper_

            ci_ = self._boot_result['fitted_rv'].quantile(
                q=[0.5 - cl / 2.0, 0.5 + cl / 2.0],
                dim=['chain', 'draw']
            )
            lower_ = ci_.sel(quantile=0.5 - cl / 2.0)
            lower = [lower_[i].values for i in self._rv['name']]
            upper_ = ci_.sel(quantile=0.5 + cl / 2.0)
            upper = [upper_[i].values for i in self._rv['name']]

            # lower_, upper_ = np.quantile(
            #     self._pars_boot.to_array().values,
            #     q=[0.5 - cl / 2.0, 0.5 + cl / 2.0],
            #     axis=(1,2)
            # )
            # pivot CI
            # lower = 2 * mle - upper_
            # upper = 2 * mle - lower_

            # percentile CI, better than pivot when underlying parameters are
            # normal distributed
            # lower = lower_
            # upper = upper_

        else:
            raise ValueError(
                f'method {method} not supported, the availables are '
                '"hdi", "se", "rse", "profile", "boot"'
            )

        ci = {}
        for name in pars_name:
            for i in range(len(mle)):
                if name == self._rv['name'][i]:
                    if self._rv['super'][i] and method not in ['hdi', 'boot']:
                        ci[name] = self.ci('boot', cl, name)[name]
                    else:
                        ci[name] = (
                            mle[i], lower[i] - mle[i], upper[i] - mle[i]
                        )
        return ci

    def mcmc_nest(self):
        ...

    def mcmc_nuts(
        self,
        draws=20000,
        tune=2000,
        init_mle=True,
        jitter=False,
        chains=4,
        target_accept=0.8,
        **kwargs
    ):
        if init_mle:
            if self._mle_result is None:
                self.mle()

            mle_root = self._mle_result['root']
            initvals = {p: v for p, v in zip(self._root['name'], mle_root)}
            initvals = [initvals] * 4
        else:
            initvals = None

        idata = sample_numpyro_nuts(draws=draws,
                                    tune=tune,
                                    chains=chains,
                                    target_accept=target_accept,
                                    random_seed=self.seed,
                                    initvals=initvals,
                                    jitter=jitter,
                                    idata_kwargs={'log_likelihood': True},
                                    model=self._pymc_model,
                                    **kwargs)

        log_likelihood = idata['log_likelihood']

        dims_to_sum = [
            d for d in log_likelihood.dims.keys()
            if d not in ['chain', 'draw']
        ]
        lnL_sum = log_likelihood.sum(dims_to_sum).to_array().sum('variable')

        lnL = []
        channel = []
        net = []
        for name in self._data_name:
            lnL_i = log_likelihood[name + '_Non'].values
            if f'{name}_Noff' in log_likelihood.data_vars:
                lnL_i = lnL_i + log_likelihood[f'{name}_Noff'].values
            lnL.append(lnL_i)
            channel.extend(log_likelihood[f'{name}_channel'].values)
            idata['log_likelihood'][f'{name}_Net'] = (
                ('chain', 'draw', f'{name}_channel'), lnL_i
            )

            data = self._data[name]
            counts = data.spec_counts
            if data.has_back:
                back_counts = data.back_counts
                factor = data.spec_exposure / data.back_exposure
                counts = counts - factor * back_counts
            idata['observed_data'][f'{name}_Net'] = (
                (f'{name}_channel',), counts
            )
            net.append(counts)

        idata['log_likelihood']['all_channel'] = (
            ('chain', 'draw', 'channel'),
            np.concatenate(lnL, axis=-1)
        )
        idata['log_likelihood']['total'] = lnL_sum

        idata['observed_data']['all_channel'] = (
            ('channel',),
            np.concatenate(net, axis=-1)
        )

        self._idata = idata.assign_coords({'channel': channel})

    # def plot_ppc(self, nsim=1000, cl=0.95):
    #     if self._idata is None:
    #         raise ValueError('run MCMC before ppc')
    #
    #     idata = draw_posterior_samples(self._idata, nsim, self.seed)
    #     lnL = idata['log_likelihood']['total'].values[0]
    #
    #     pm.sample_posterior_predictive(
    #         trace=idata,
    #         model=self._pymc_model,
    #         random_seed=self.seed,
    #         progressbar=False,
    #         extend_inferencedata=True
    #     )
    #
    #     pdata = {
    #         k.replace('_Non', '_spec_counts')
    #         .replace('_Noff', '_back_counts'): v.values[0]
    #         # [0] is because there is only one chain in idata
    #         for k, v in idata['posterior_predictive'].items()
    #     }
    #
    #     post = [idata['posterior'][i].values[0] for i in self._root['name']]
    #     post = np.ascontiguousarray(np.transpose(post))
    #
    #     Drep = np.zeros(nsim)
    #     Dmin = np.empty(nsim)
    #     flag = np.full(nsim, True)
    #     try:
    #         for i in tqdm(range(nsim), desc='Fitting'):
    #             pm.set_data({d: pdata[d][i] for d in pdata}, self._pymc_model)
    #             Drep[i] = self._deviance(self._root_to_values(post[i]))
    #             opt_res = minimize(fun=self._deviance,
    #                                x0=self._root_to_values(post[i]),
    #                                method='L-BFGS-B',
    #                                jac='2-point')
    #             Dmin[i] = opt_res.fun
    #             if not opt_res.success:
    #                 flag[i] = False
    #         Dmin = Dmin[flag]
    #     except Exception as e:
    #         raise e
    #     finally:
    #         observed = {
    #             k.replace('_Non', '_spec_counts')
    #             .replace('_Noff', '_back_counts'): v.values
    #             for k, v in idata['observed_data'].items()
    #         }
    #         pm.set_data(observed, self._pymc_model)
    #
    #     plt.style.use(['nature', 'science', 'no-latex'])
    #
    #     fig, axes = plt.subplots(1, 2, figsize=(4, 2.2), dpi=150)
    #     fig.subplots_adjust(
    #         left=0.125, bottom=0.11, right=0.875, top=0.912, wspace=0.05
    #     )
    #     axes[0].set_box_aspect(1)
    #     axes[1].set_box_aspect(1)
    #
    #     D = -2 * lnL
    #     _min = min(D.min(), Drep.min()) * 0.9
    #     _max = max(D.max(), Drep.max()) * 1.1
    #     axes[0].plot([_min, _max], [_min, _max], ls=':', color='gray')
    #     axes[0].set_xlim(_min, _max)
    #     axes[0].set_ylim(_min, _max)
    #     ppp1 = (Drep > D).sum() / D.size
    #     axes[0].set_title(f'$p$-value$=${ppp1:.3f}')
    #     axes[0].scatter(D, Drep, s=1, marker='.', alpha=0.5)
    #     axes[0].set_aspect('equal')
    #     axes[0].set_xlabel('$D$')
    #     axes[0].set_ylabel(r'$D^{\rm rep}$')
    #
    #     if self._mle_result is None:
    #         self.mle()
    #     Dmin_obs = self._mle_result['minuit'].fmin.fval
    #     ppp2 = (Dmin > Dmin_obs).sum() / Dmin.size
    #     axes[1].set_title(f'$p$-value$=${ppp2:.3f}')
    #     # axes[1].hist(Dmin, bins='auto', density=True, alpha=0.4)
    #     grid, pdf = az.kde(Dmin)
    #     axes[1].fill_between(grid, pdf, alpha=0.4)
    #     axes[1].plot(grid, pdf)
    #     axes[1].set_xlim(grid.min(), grid.max())
    #     axes[1].set_ylim(bottom=0.0)
    #     axes[1].axvline(self._dof, ls=':', color='gray')
    #     axes[1].axvline(Dmin_obs, c='r', ls='--')
    #     axes[1].set_xlabel(r'$D_{\rm min}$')
    #     # axes[1].set_ylabel('$N$ simulation')
    #     axes[1].set_ylabel('PDF')
    #     axes[1].yaxis.set_label_position("right")
    #     axes[1].yaxis.tick_right()
    #     axes[1].yaxis.set_ticks_position('both')
    #
    #     fig, axes = plt.subplots(2, 1, sharex=True, dpi=150)
    #     fig.subplots_adjust(hspace=0, wspace=0)
    #     fig.align_ylabels(axes)
    #
    #     if cl >= 1.0:
    #         cl = 1.0 - stats.norm.sf(cl) * 2.0
    #
    #     colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #     for i, color in zip(self._data_name, colors):
    #         data = self._data[i]
    #         delta = data.ch_emax - data.ch_emin
    #         src_factor = 1.0 / (delta * data.spec_exposure)
    #         bkg_factor = 1.0 / (delta * data.back_exposure)
    #
    #         d = data.net_spec
    #         Qd = d.cumsum()
    #         Qd /= Qd[-1]
    #
    #         # CE = getattr(getattr(model, f'{i}_model'), 'CE')
    #         # CE_pars = [p for p in pars_name if p in CE.finder]
    #         # CE_kwargs = {
    #         #     j: getattr(data, j) for j in CE.finder
    #         #     if type(j) == str and j not in pars_name
    #         # }
    #
    #         # m = np.empty((nsim, data.channel.size))
    #         # for n in range(nsim):
    #         #     m[n] = CE(**{p: post_pars[p][n] for p in CE_pars}, **CE_kwargs)
    #         # m *= src_factor
    #         # Qm = m.cumsum(-1)
    #         # Qm /= Qm[:, -1:]
    #
    #         dsim = pdata[f'{i}_spec_counts'] * src_factor
    #         if f'{i}_back_counts' in pdata:
    #             dsim -= pdata[f'{i}_back_counts'] * bkg_factor
    #         Qdsim = dsim.cumsum(-1)
    #         Qdsim /= Qdsim[:, -1:]
    #
    #         # Qm_hdi = az.hdi(np.expand_dims(Qm, axis=0), hdi_prob=q).T
    #         Qdsim_hdi = az.hdi(np.expand_dims(Qdsim, axis=0), hdi_prob=cl).T
    #         # Qdm_hdi = az.hdi(np.expand_dims(Qd - Qm, axis=0), hdi_prob=q).T
    #         # Qdsimm = Qdsim - Qm
    #         # Qdsimm_hdi = az.hdi(np.expand_dims(Qdsimm, axis=0), hdi_prob=q).T
    #         Qddsim = az.hdi(np.expand_dims(Qd - Qdsim, axis=0), hdi_prob=cl).T
    #
    #         d_err = Qdsim.std(axis=0, ddof=1)
    #         d_err[d_err == 0.0] = 1
    #         Qddsim /= d_err
    #
    #         # dm_err = Qdsimm.std(axis=0, ddof=1)
    #         # dm_err[dm_err==0.0] = 1.0
    #         # Qdm_hdi /= dm_err
    #         # Qdsimm_hdi /= dm_err
    #
    #         mask = data.ch_emin[1:] != data.ch_emax[:-1]
    #         idx = [0, *(np.flatnonzero(mask) + 1), len(data.channel)]
    #         for j in range(len(idx) - 1):
    #             slice_j = slice(idx[j], idx[j + 1])
    #             ebins = np.append(data.ch_emin[slice_j],
    #                               data.ch_emax[slice_j][-1])
    #
    #             axes[0].step(
    #                 ebins, np.append(Qd[slice_j], Qd[slice_j][-1]),
    #                 lw=0.6, where='post', color=color
    #             )
    #             # axes[0].fill_between(
    #             #     ebins,
    #             #     np.append(Qm_hdi[0][slice_j], Qm_hdi[0][slice_j][-1]),
    #             #     np.append(Qm_hdi[1][slice_j], Qm_hdi[1][slice_j][-1]),
    #             #     lw=0.2, step='post', alpha=0.6, color=color
    #             # )
    #             axes[0].fill_between(
    #                 ebins,
    #                 np.append(Qdsim_hdi[0][slice_j],
    #                           Qdsim_hdi[0][slice_j][-1]),
    #                 np.append(Qdsim_hdi[1][slice_j],
    #                           Qdsim_hdi[1][slice_j][-1]),
    #                 lw=0, step='post', alpha=0.4, color='gray'
    #             )
    #
    #             # axes[1].fill_between(
    #             #     ebins,
    #             #     np.append(Qdm_hdi[0][slice_j], Qdm_hdi[0][slice_j][-1]),
    #             #     np.append(Qdm_hdi[1][slice_j], Qdm_hdi[1][slice_j][-1]),
    #             #     lw=0.2, step='post', alpha=0.6, color=color
    #             # )
    #             axes[1].fill_between(
    #                 ebins,
    #                 np.append(Qddsim[0][slice_j], Qddsim[0][slice_j][-1]),
    #                 np.append(Qddsim[1][slice_j], Qddsim[1][slice_j][-1]),
    #                 lw=0, step='post', alpha=0.4, color='gray'
    #             )
    #
    #     axes[1].axhline(0, ls=':', c='gray', zorder=0)
    #     axes[0].set_xscale('log')
    #     axes[0].set_ylabel('EDF')
    #     axes[1].set_ylabel('EDF residual')
    #     axes[1].set_xlabel('Energy [keV]')
    #
    #     plt.style.use('default')
    #
    #     return ppp1, ppp2

    def plot_data(
        self,
        format='ldata sdev icnt',
        plot_comps=True,
        sim=None,
        n=1000,
        show_pars=None,
        fig_path=None,
        c=1
    ):
        supported_plots = {
            'data': _plot_data,
            'ldata': _plot_ldata,
            'uf': _plot_ufspec,
            'euf': _plot_eufspec,
            'eeuf': _plot_eeufspec,
            'icnt': _plot_icounts,
            'icntresd': _plot_icounts_residual,
            'resd': _plot_residuals,
            'ratio': _plot_ratio,
            'delchi': _plot_delchi,
            'dev': _plot_deviance,
            'sdev': _plot_sign_deviance,
            'sdev2': _plot_sign_deviance2,
        }

        plots = format.split(' ')
        for i in plots:
            if i not in supported_plots:
                raise ValueError(
                    f'{i} plot not supported, availables are '
                    f'{list(supported_plots.keys())}'
                )
        plot_funcs = [supported_plots[i] for i in plots]

        if self._mle_result is None:
            self.mle()

        if sim is not None:
            if sim == 'boot':
                if self._boot_result is None:
                    self.bootstrap(n)
                data_sim = self._boot_result['data']
                rv_dist = self._boot_result['fitted_rv']
                rv_fit = self._boot_result['fitted_rv']
                deviance_dist = self._boot_result['deviance']['point']
                p_value = self._boot_result['p_value']['total']

            elif sim == 'ppc':
                if self._ppc_result is None:
                    self.ppc()
                data_sim = self._ppc_result['data']
                rv_dist = self._ppc_result['init_rv']
                rv_fit = self._ppc_result['fitted_rv']
                deviance_dist = self._ppc_result['deviance']['point']
                p_value = self._ppc_result['p_value']['total']

            else:
                raise ValueError('`sim_type` must be "boot" or "ppc"')

        else:
            data_sim = None
            rv_dist = None
            rv_fit = None
            deviance_dist = None
            p_value = None

        if data_sim is not None:
            use_data_sim = False
            plot_use_data_sim = [
                'icnt', 'icntresd', 'resd', 'ratio', 'delchi', 'sdev2'
            ]
            for i in plots:
                if i in plot_use_data_sim:
                    use_data_sim = True
                    break
            if use_data_sim:
                data_sim = [
                    data_sim[f'{i}_Net'].values[0] for i in self._data_name
                ]
            else:
                data_sim = None

        if rv_dist is not None:
            use_rv_dist = False
            plot_use_rv_dist = [
                'data', 'ldata', 'uf', 'euf', 'eeuf'
            ]
            for i in plots:
                if i in plot_use_rv_dist:
                    use_rv_dist = True
                    break
            if use_rv_dist:
                rv_dist = [
                    rv_dist[i].values[0] for i in self._rv['name']
                ]
                rv_dist = np.column_stack(rv_dist)
            else:
                rv_dist = None

        if rv_fit is not None:
            use_rv_fit = False
            plot_use_rv_fit = [
                'icntresd', 'resd', 'ratio', 'delchi', 'sdev2'
            ]
            for i in plots:
                if i in plot_use_rv_fit:
                    use_rv_fit = True
                    break
            if use_rv_fit:
                rv_fit = [
                    rv_fit[i].values[0] for i in self._rv['name']
                ]
                rv_fit = np.column_stack(rv_fit)
            else:
                rv_fit = None

        if deviance_dist is not None:
            use_deviance_dist = False
            plot_use_deviance_dist = [
                'dev', 'sdev', 'sdev2'
            ]
            for i in plots:
                if i in plot_use_deviance_dist:
                    use_deviance_dist = True
                    break
            if not use_deviance_dist:
                deviance_dist = None

        colors = {
            1: (
                '#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97',
                '#474747', '#9e9e9e'
            ),
            2: (
                "#0d49fb", "#e6091c", "#26eb47", "#8936df", "#fec32d",
                "#25d7fd"
            ),
            3: (
                '#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE',
                '#AA3377', '#BBBBBB'
            )
        }[c]

        markers = ['s', 'o', 'D', '^', 'd', 'p', 'h', 'H', 'D']

        with plt.style.context(['nature', 'science']):
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['font.serif'] = 'Times New Roman'
            plt.rcParams['text.latex.preamble'] += r' \usepackage{mathptmx}'
            hr = {
                'data': 1.6,
                'ldata': 1.6,
                'uf': 1.6,
                'euf': 1.6,
                'eeuf': 1.6,
                'icnt': 1,
                'icntresd': 1,
                'resd': 1,
                'ratio': 1,
                'delchi': 1,
                'dev': 1,
                'sdev': 1,
                'sdev2': 1,
            }
            height_ratios = [hr[i] for i in plots]
            width = 1.25*sum(height_ratios)
            fig, axes = plt.subplots(
                len(plots), 1,
                sharex=True,
                gridspec_kw={'height_ratios': height_ratios},
                figsize=[width/3*4, width],
                dpi=200
            )
            fig.subplots_adjust(hspace=0, wspace=0)
            fig.align_ylabels(axes)
            mle = self._mle_result

            if show_pars is not None:
                if show_pars is True:
                    pars_name = self._get_pars_name(None)
                else:
                    pars_name = self._get_pars_name(show_pars)

                pars_info = []
                for pname in pars_name:
                    best, err = mle['pars'][pname]
                    pars_info.append(
                        rf'{pname}: {best:.2f}$\pm${err:.2f}'
                    )
                axes[0].annotate('\n'.join(pars_info),
                                 xy=(0.04, 0.05), xycoords='axes fraction',
                                 ha='left', va='bottom')

            rv_to_pars = self._rv_to_params
            alpha = 0.6
            # if sim == 'boot':
            #     hdi = False
            # else:
            #     hdi = True
            hdi = False

            for i, data_name in enumerate(self._data_name):
                color = colors[i]
                marker = markers[i]
                data = self._data[data_name]
                model = self._model[data_name]
                pars = rv_to_pars(self._mle_result['rv'], data_name)

                if rv_dist is not None:
                    pars_dist = rv_to_pars(rv_dist, data_name)
                else:
                    pars_dist = None

                if rv_fit is not None:
                    pars_fit = rv_to_pars(rv_fit, data_name)
                else:
                    pars_fit = None

                if data_sim is not None:
                    data_sim_ = data_sim[i]
                else:
                    data_sim_ = None

                if deviance_dist is not None:
                    deviance_dist_ = deviance_dist[i]
                else:
                    deviance_dist_ = None

                deviance_obs = self._mle_result['deviance']['point'][i]

                args = [
                    data, model, pars,
                    marker, color, alpha, plot_comps,
                    pars_dist, pars_fit, data_sim_,
                    deviance_obs, deviance_dist_,
                    hdi
                ]

                for ax, f in zip(axes, plot_funcs):
                    f(ax, *args)

            xlim_left = np.min([d.ch_emin.min() for d in self._data.values()])
            xlim_right = np.max([d.ch_emax.max() for d in self._data.values()])
            axes[0].set_xlim(xlim_left*0.93, xlim_right*1.07)
            axes[0].set_xscale('log')
            axes[0].legend(frameon=True, loc='upper right')
            title = f'{model.expression}'
            stat = mle['stat']
            dof = mle['dof']
            title += f', stat/dof={stat/dof:.2f} ({stat:.2f}/{dof})'
            if p_value is not None:
                if sim == 'ppc':
                    sim_str = 'Posterior Predictive'
                else:
                    sim_str = 'Bootstrap'
                title += f', {sim_str} $p$-value={p_value:.3f}'
            axes[0].set_title(title)
            axes[-1].set_xlabel('Energy [keV]')

            for i, plot in enumerate(plots):
                if plot == 'data' or plot == 'ldata':
                    axes[i].set_ylabel('$C_E$ [s$^{-1}$ keV$^{-1}$]')
                elif plot == 'uf':
                    axes[i].set_ylabel('$N_E$ [cm$^{-2}$ s$^{-1}$ keV$^{-1}$]')
                elif plot == 'euf':
                    axes[i].set_ylabel('$EN_E$ [erg cm$^{-2}$ s$^{-1}$ keV$^{-1}$]')
                elif plot == 'eeuf':
                    axes[i].set_ylabel('$E^2 N_E$ [erg cm$^{-2}$ s$^{-1}$]')
                elif plot == 'icnt':
                    axes[i].set_ylabel('Integrated Counts')
                elif plot == 'icntresd':
                    axes[i].set_ylabel('Icounts $-$ Imodel')
                    axes[i].axhline(0, ls='--', c='gray', zorder=0, lw=1)
                elif plot == 'resd':
                    axes[i].set_ylabel('Data $-$ Model')
                    axes[i].axhline(0, ls='--', c='gray', zorder=0, lw=1)
                elif plot == 'ratio':
                    axes[i].set_ylabel('Data/Model')
                    axes[i].axhline(1, ls='--', c='gray', zorder=0, lw=1)
                elif plot == 'delchi':
                    axes[i].set_ylabel('(Data$-$Model)/Error')
                    axes[i].axhline(0, ls='--', c='gray', zorder=0, lw=1)
                elif plot == 'dev':
                    axes[i].set_ylabel('Deviance')
                elif plot == 'sdev' or plot == 'sdev2':
                    axes[i].set_ylabel('sign * Deviance$^{1/2}$')
                    axes[i].axhline(0, ls='--', c='gray', zorder=0, lw=1)

                if plot in ['ldata', 'uf'] and axes[i].get_ylim()[0] <= 1e-8:
                    axes[i].set_ylim(bottom=1e-8)
                elif plot in ['euf', 'eeuf'] and axes[i].get_ylim()[0] <= 1e-17:
                    axes[i].set_ylim(bottom=1e-17)

            if fig_path is not None:
                fig.savefig(fig_path)

            return fig, axes

    def plot_trace(self, root=False, back=False, fig_path=None):
        if self._idata is None:
            raise ValueError('run MCMC before plotting trace')

        var_names = list(self._rv['name'])
        if root:
            var_names.extend(self._root['name'])
        if back:
            var_names.extend(
                [p for p in self._idata.posterior.data_vars if '_BKG' in p]
            )
        with az.style.context(['arviz-darkgrid']):
            az.plot_trace(self._idata, var_names=var_names, compact=False)
            if fig_path:
                plt.gcf().savefig(fig_path)

    def plot_corner(
            self, samples='mcmc', profile=True, root=False, level_idx=3,
            smooth=0.0, fig_path=None, **kwargs
    ):
        if samples == 'mcmc':
            if self._idata is None:
                raise ValueError('run MCMC before plotting corner')
            idata = self._idata

        elif samples == 'boot':
            if self._boot_result is None:
                raise ValueError('run bootstrap before plotting corner')
            idata = az.InferenceData(posterior=self._boot_result['fitted_rv'])
        else:
            raise ValueError('`samples` should be "mcmc" or "boot"')

        var_names = list(self._rv['name'])
        if root:
            var_names.extend(self._root['name'])

        plot_corner(
            idata, var_names, profile, level_idx, smooth, fig_path,
            **kwargs
        )

    def loo(self, scale='deviance'):
        if self._idata is None:
            raise ValueError('run MCMC before calculating PSIS-LOO-CV')

        loo = az.loo(self._idata, var_name='all_channel', scale=scale)

        return loo.elpd_loo, loo.se, loo.p_loo, loo.warning

    def waic(self, scale='deviance'):
        if self._idata is None:
            raise ValueError('run MCMC before calculating WAIC')

        waic = az.waic(self._idata, var_name='all_channel', scale=scale)

        return waic.elpd_waic, waic.se, waic.p_waic, waic.warning

    def _check_input(self, data, model, stat):
        # check data
        data_list = list(np.atleast_1d(data))
        n_data = len(data_list)
        if n_data == 0:
            raise ValueError('data is empty')

        for d in data_list:
            if not isinstance(d, Data):
                raise TypeError('data must be given in "Data" type')

        # check model
        model_list = list(np.atleast_1d(model))
        for m in model_list:
            if not isinstance(m, SpectralModel):
                raise TypeError('model must be given in "SpectralModel" type')
            else:
                if m.mtype != 'add':
                    raise TypeError(
                        f'photon flux is undefined for "{m}", additive model '
                        'is required'
                    )

        n_model = len(model_list)
        if n_model == 0:
            raise ValueError('model is empty')
        elif n_model == 1:
            model_list *= n_data
        else:
            if n_model != n_data:
                raise ValueError(
                    f'data number ({n_data}) and model number ({n_model}) are '
                    'not matched'
                )

        # check stat
        stat_list = list(np.atleast_1d(stat))
        for s in stat_list:
            if s not in self.stat_options:
                raise ValueError(
                    f'stat must be one of {self.stat_options}, but got {s}'
                )

        n_stat = len(stat_list)
        if n_data == 0:
            raise ValueError('stat is empty')
        elif n_stat == 1:
            stat_list *= n_data
        else:
            if n_stat != n_data:
                raise ValueError(
                    f'data number ({n_data}) and stat number ({n_stat}) are '
                    'not matched'
                )

        return data_list, model_list, stat_list


    def _get_rv(self):
        rv = []
        name = []
        default = []
        super_ = []

        for m in self._model.values():
            params = m.params
            for i in range(len(params['rv'])):
                if params['rv'][i] not in rv and not params['frozen'][i]:
                    rv.append(params['rv'][i])
                    name.append(params['name'][i])
                    default.append(params['default'][i])
                    super_.append(params['super'][i])

        return {
            'rv': tuple(rv),
            'name': tuple(name),
            'default': tuple(default),
            'super': tuple(super_)
        }

    def _get_root(self):
        root = []
        name = []
        default = []

        for m in self._model.values():
            r = m.root
            for i in range(len(r['root'])):
                if r['root'][i] not in root:
                    root.append(r['root'][i])
                    name.append(r['name'][i])
                    default.append(r['default'][i])

        return {
            'root': tuple(root),
            'name': tuple(name),
            'default': tuple(default)
        }

    def _get_values(self):
        return [self._pymc_model.rvs_to_values[r] for r in self._root['root']]

    def _get_pars_name(self, pars):
        if pars is None:
            pars = self._rv['name']
        else:
            if type(pars) == str:
                pars = [pars]
            elif not type(pars) in [list, tuple, set]:
                raise TypeError('`pars` should be a list, tuple, or set')

            for p in pars:
                if p not in self._rv['name']:
                    raise ValueError(
                        f'no parameters {p} in the model {self._rv["name"]})'
                    )

        return list(pars)

    def _deviance(self, values):
        if self.__deviance is None:
            self.__deviance = function(self._values, -2 * self._pymc_model.observedlogp)

            # f = function(self._values, -2 * self._pymc_model.observedlogp)
            # self.__deviance = np.vectorize(
            #     lambda x: f(*x),
            #     otypes=[np.float64],
            #     signature='(i)->()'
            # )

        return self.__deviance(*values)

    def _deviance_group(self, values):
        if self.__deviance_group is None:
            logp = {
                i.name: i
                for i in self._pymc_model.logp(self._pymc_model.observed_RVs, sum=False)
            }

            data_logp = []
            for i in self._data_name:
                logp_i = logp[f'{i}_spec_counts_logprob'].sum()

                logp_i_back = logp.get(f'{i}_back_counts_logprob')
                if logp_i_back is not None:
                    logp_i = logp_i + logp_i_back.sum()

                data_logp.append(-2 * logp_i)

            self.__deviance_group = function(self._values, pt.stack(data_logp))

            # f = function(self._values, pt.stack(data_logp))
            # self.__deviance_group = np.vectorize(
            #     lambda x: f(*x),
            #     otypes=[np.float64],
            #     signature='(i)->(j)'
            # )

        return self.__deviance_group(*values)

    def _deviance_point(self, values):
        if self.__deviance_point is None:
            logp = {
                i.name: i
                for i in self._pymc_model.logp(self._pymc_model.observed_RVs, sum=False)
            }

            point_logp = []
            for i in self._data_name:
                logp_i = logp[f'{i}_spec_counts_logprob']

                logp_i_back = logp.get(f'{i}_back_counts_logprob')
                if logp_i_back is not None:
                    logp_i = logp_i + logp_i_back

                point_logp.append(-2 * logp_i)

            self.__deviance_point = function(self._values, point_logp)
            # f = function(self._values, point_logp)
            # out = ','.join([f'(o{d+1})' for d in range(len(self._data_name))])
            # self.__deviance_point = np.vectorize(
            #     lambda x: tuple(f(*x)),
            #     otypes=[np.float64]*len(self._data_name),
            #     signature=f'(i)->{out}'
            # )

        return self.__deviance_point(*values)

    def _lnL(self, values):
        if self.__lnL is None:
            compiled = function(self._values, self._pymc_model.observedlogp)
            self.__lnL = compiled

        return self.__lnL(*values)

    def _values_covar(self, values):
        if self.__values_covar is None:
            hess = Hessian(self._lnL)
            self.__values_covar = lambda x: np.linalg.inv(-hess(x))

        return self.__values_covar(values)

    # pytensor version
    # def _values_robust_covar(self, values):
    #     ctx = self._pymc_model
    #
    #     point_wise_score = [
    #         pt.stack(pt.jacobian(lnL_i, self._values), axis=0)
    #         for lnL_i in ctx.logp(ctx.observed_RVs, False, False)
    #     ]  # score, shape=(n_par, n_point) for each data
    #
    #     score_outer = pt.sum(
    #         input=[
    #             pt.sum(
    #                 i.dimshuffle(0, 'x', 1) * i.dimshuffle('x', 0, 1),
    #                 axis=2
    #             )
    #             for i in point_wise_score
    #         ],
    #         axis=0
    #     )
    #     score_outer_compiled = function(self._values, score_outer)
    #     cov = self._values_covar(values)
    #
    #     cov = np.array(cov @ score_outer_compiled(*values) @ cov)
    #     return cov

    def _values_robust_covar(self, values):
        if self.__values_robust_covar is None:
            ctx = self._pymc_model

            logp = ctx.logp(ctx.observed_RVs, jacobian=False, sum=False)

            compiled = [function(self._values, i) for i in logp]
            compiled = list(map(lambda f: (lambda x: f(*x)), compiled))

            score = list(map(lambda f: Jacobian(f), compiled))

            outer = lambda s: (s[:, :, None] * s[:, None, :]).sum(axis=0)
            outer_funcs = list(map(lambda f: lambda x: outer(f(x)), score))
            score_outer = lambda x: np.sum([i(x) for i in outer_funcs], axis=0)

            def robust_covar(x):
                covar = self._values_covar(x)
                return covar @ score_outer(x) @ covar

            self.__values_robust_covar = robust_covar

        return self.__values_robust_covar(values)

    def _values_to_root(self, values):
        if self.__values_to_root is None:
            transforms = self._pymc_model.rvs_to_transforms
            root = self._root

            scalar_inputs = [pt.scalar() for _ in root['root']]
            scalar_res = [
                transforms[r].backward(scalar, *r.owner.inputs)
                for r, scalar in zip(root['root'], scalar_inputs)
            ]
            f = function(scalar_inputs, pt.stack(scalar_res))
            self.__values_to_root = np.vectorize(
                lambda x: f(*x),
                otypes=[np.float64],
                signature='(i)->(j)'
            )

        return self.__values_to_root(values)

    def _values_to_rv(self, values):
        return self._root_to_rv(self._values_to_root(values))

    def _root_to_rv(self, root_value):
        if self.__root_to_rv is None:
            f = function(self._root['root'], pt.stack(self._rv['rv']))
            self.__root_to_rv = np.vectorize(
                lambda x: f(*x),
                otypes=[np.float64],
                signature='(i)->(j)'
            )

        return self.__root_to_rv(root_value)

    def _root_to_values(self, root_value):
        if self.__root_to_values is None:
            transforms = self._pymc_model.rvs_to_transforms
            root = self._root

            scalar_inputs = [pt.scalar() for _ in root['root']]
            scalar_res = [
                transforms[r].forward(scalar, *r.owner.inputs)
                for r, scalar in zip(root['root'], scalar_inputs)
            ]
            f = function(scalar_inputs, pt.stack(scalar_res))
            self.__root_to_values = np.vectorize(
                lambda x: f(*x),
                otypes=[np.float64],
                signature='(i)->(j)'
            )

        return self.__root_to_values(root_value)

    def _rv_to_params(self, rv_values, data_name):
        if self.__rv_to_params is None:
            def rv_to_params(rv, data_name):
                params = self._params[data_name]
                default = params['default']
                frozen = params['frozen']
                rv_mask = params['rv_mask']
                params_values = np.asarray(default)
                params_values[~frozen] = np.asarray(rv[rv_mask])
                return params_values

            self.__rv_to_params = np.vectorize(
                rv_to_params,
                otypes=[np.float64],
                excluded={'data_name'},
                signature='(i)->(j)'
            )

        return self.__rv_to_params(rv_values, data_name=data_name)


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def EDFstat(data, model):
    data = np.atleast_2d(data)
    model = np.atleast_2d(model)

    Y = data.cumsum(axis=1)
    M = model.cumsum(axis=1)

    Y = Y / Y[:, -1:]
    M = M / M[:, -1:]

    diff = Y - M
    diff2 = diff * diff
    dF = np.diff(np.column_stack([np.zeros(len(M)), M]), axis=1)
    diff_dF = diff2 * dF

    ks = np.max(np.abs(diff), axis=1)
    log_ks = np.log(ks)

    cvm = np.sum(diff_dF, axis=1)
    log_cvm = np.log(cvm)

    _weight = M * np.abs(1 - M)
    mask = _weight != 0.0
    with np.errstate(divide='ignore'):
        weight = np.where(mask, 1.0/_weight, 0.0)
    ad = np.sum(diff_dF * weight, axis=1)
    log_ad = np.log(ad)

    cusum = np.max(diff, axis=1) - np.min(diff, axis=1)
    log_cusum = np.log(cusum)

    edf = np.column_stack([log_ks, log_cvm, log_ad, log_cusum])
    if len(edf) == 1:
        edf = edf[0]

    return edf


def runs(resd_list):
    R = 0
    Np = 0
    Nn = 0

    for resd in resd_list:
        R += 1 + np.sum(resd[:-1] * resd[1:] <= 0.0)
        positive = resd > 0.0
        Np += np.sum(positive)
        Nn += np.sum(~positive)

    N = Np + Nn
    mu = 2 * Np * Nn / N + 1
    runs = (R - mu) / np.sqrt((mu - 1) * (mu - 2) / (N - 1))

    return runs


def draw_posterior_samples(idata, n, seed):
    posterior = idata.posterior
    rng = np.random.default_rng(seed)
    i = rng.integers(0, posterior.chain.size, n)
    j = rng.integers(0, posterior.draw.size, n)
    coords = {
        'chain': ('chain', [0]),
        'draw': ('draw', np.arange(n))
    }
    coords.update({
        k: v.values
        for k, v in posterior.coords.items()
        if k not in ['chain', 'draw']
    })
    posterior_dataset = xr.Dataset(
        data_vars={
            k: (v.coords.dims, np.expand_dims(v.values[i, j], axis=0))
            for k, v in posterior.data_vars.items()
        },
        coords={
            'chain': ('chain', [0]),
            'draw': ('draw', np.arange(n))
        }
    )
    idata2 = az.InferenceData(posterior=posterior_dataset)

    if 'log_likelihood' in idata.groups():
        log_likelihood = idata.log_likelihood
        coords = {
            'chain': ('chain', [0]),
            'draw': ('draw', np.arange(n))
        }
        coords.update({
            k: v.values
            for k, v in log_likelihood.coords.items()
            if k not in ['chain', 'draw']
        })
        log_likelihood_dataset = xr.Dataset(
            data_vars={
                k: (v.coords.dims, np.expand_dims(v.values[i, j], axis=0))
                for k, v in log_likelihood.data_vars.items()
            },
            coords=coords
        )
        idata2.add_groups({'log_likelihood': log_likelihood_dataset})

    return idata2


if __name__ == '__main__':
    # NaI_data = Data(
    #     [28, 250],
    #     '/Users/xuewc/BurstData/EXO2030+375/upper_limit/NaI_TOTAL.fits{2}',
    #     '/Users/xuewc/BurstData/EXO2030+375/upper_limit/NaI_BKG.fits{2}',
    #     '/Users/xuewc/BurstData/EXO2030+375/upper_limit/HE_rsp.rsp',
    #     name='NaI'
    # )
    # CsI_data = Data(
    #     [200, 600],
    #     '/Users/xuewc/BurstData/EXO2030+375/upper_limit/CsI_TOTAL.fits{2}',
    #     '/Users/xuewc/BurstData/EXO2030+375/upper_limit/CsI_BKG.fits{2}',
    #     '/Users/xuewc/BurstData/EXO2030+375/upper_limit/DA.rsp',
    #     name='CsI'
    # )
    #
    # from elisa import xs, CutoffPowerlaw, UniformParameter
    # src = CutoffPowerlaw() + xs.posm()
    # # src.CPL.norm.log = True
    # src.CPL.norm = src.posm.norm * UniformParameter('f', 10, 1e-5, 1e5, log=1)
    # src.CPL.Ec.log = True
    # i = Infer([NaI_data, CsI_data], src, ['wstat', 'pgstat'])
    # print(i.mle())
    # i.plot_spec()
    # i.mcmc_nuts(1000, 1000)

    path = '/Users/xuewc/BurstData/FRB221014/HXMT/'
    LE = Data([2, 10],
              f'{path}/LE_optbmin5.fits',
              f'{path}/LE_phabkg20s_g0_0-94.pha',
              f'{path}/LE_rsp.rsp',
              group_type='bmin',
              group_scale=25)

    ME = Data([10, 35],
              f'{path}/ME_optbmin5.fits',
              f'{path}/ME_phabkg20s_g0_0-53.pha',
              f'{path}/ME_rsp.rsp',
              group_type='bmin',
              group_scale=25)

    HE = Data([28, 250],
              f'{path}/HE_optbmin5.fits',
              f'{path}/HE_phabkg20s_g0_0-12.pha',
              f'{path}/HE_rsp.rsp',
              group_type='bmin',
              group_scale=25)

    from elisa import BlackBodyRad, CutoffPowerlaw, OTTB, Powerlaw, xs, EnergyFlux, UniformParameter
    wabs = xs.wabs(2.79)
    src = [
        xs.powerlaw(),
        Powerlaw(),
        EnergyFlux(1.5, 250) * Powerlaw(norm=1),
        EnergyFlux(1.5, 250) * Powerlaw(norm=1) + EnergyFlux(1.5, 250) * BlackBodyRad(norm=1),
        CutoffPowerlaw()
    ][1]
    # src.CPL.Ec.max = 300
    # src.CPL.Ec.log = True
    # src.CPL.norm.log = True
    # src.PL.norm = src.PL.PhoIndex + src.PL.PhoIndex
    src.PL.norm.log = 1
    # src = BlackBodyRad()
    # src.BBrad.norm.log = 1
    # src.BBrad.kT.log = 1
    # src = OOTB()
    # src.OOTB.kT.log = 1
    # src.OOTB.norm.log = 1
    # src = EnergyFlux(1.5, 250)*BlackBodyRad(norm=1) + EnergyFlux(1.5, 250)*BlackBodyRad(norm=1)
    # src.BBrad_2.kT = src.BBrad.kT * UniformParameter('factor', 0.5, 0.001, 0.999, log=1)
    infer = Infer([LE, ME, HE], wabs*src, 'wstat')
    # infer.bootstrap()
    infer.mcmc_nuts()
    infer.plot_corner()
    infer.ppc()
    infer.plot_data('ldata sdev icnt',
                    sim='ppc',
                    show_pars=infer._rv['name'])

    # test for GRB 230307A
    # path = '/Users/xuewc/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/188ccc99a439bc9cc579a929ee149d49/Message/MessageTemp/f73fb0aee92be46a90a93165d0b6ae4c/OpenData/11668/39e6f9ad32176b0685ae7020e561c41b'
    # data = []
    # for i in [4,5,1,3,12]:
    #     data.extend([
    #         Data([40, 300],
    #              f'{path}/full_spec_bg{i:02d}H_v20230704.FITS{{1}}',
    #              f'{path}/bg_spec_bg{i:02d}H_v20230704.FITS{{1}}',
    #              f'{path}/gbg_{i:02d}H_x_evt_v00.rsp',
    #              name=f'GBG{i:02d}H', is_spec_poisson=True, is_back_poisson=False),
    #         Data([700, 8000] if i != 1 else [700, 7400],
    #              f'{path}/full_spec_bg{i:02d}L_v20230704.FITS{{1}}',
    #              f'{path}/bg_spec_bg{i:02d}L_v20230704.FITS{{1}}',
    #              f'{path}/gbg_{i:02d}L_x_evt_v00.rsp',
    #              name=f'GBG{i:02d}L', is_spec_poisson=True, is_back_poisson=False)
    #     ])
    #
    # data.extend([
    #     Data([[15,35],[40,100]],
    #          f'{path}/GCG01H_195Phases_TOTAL.fits{{56}}',
    #          f'{path}/GCG01H_195Phases_BKG.fits{{56}}',
    #          f'{path}/gcg_01H_x_evt.rsp',
    #          name='GCG01H', is_spec_poisson=True, is_back_poisson=False),
    #     Data([1000,6000],
    #          f'{path}/full_spec_cg01L_v20230704.FITS{{1}}',
    #          f'{path}/bg_spec_cg01L_v20230704.FITS{{1}}',
    #          f'{path}/gcg_01L_x_evt.rsp',
    #          name='GCG01L', is_spec_poisson=True, is_back_poisson=False),
    # ])
    #
    # src = xs.cutoffpl()
    # src.cutoffpl.HighECut.max = 3000
    # # src.CPL.norm.log=1
    # # src.CPL.Ec.log=1
    # from elisa import Constant
    # f1 = Constant()*src
    # f1.constant.name='B05'
    # f2 = Constant()*src
    # f2.constant.name='B01'
    # f3 = Constant()*src
    # f3.constant.name='B03'
    # f4 = Constant()*src
    # f4.constant.name='B12'
    # f5 = Constant()*src
    # f5.constant.name='C01'
    # infer = Infer(data, [src, src,
    #                             f1,f1,f2,f2,f3,f3,f4,f4,f5,f5], 'pgstat')
    # infer.restore('/Users/xuewc/test')
    # plt.close('all')
