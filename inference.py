import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from bayespec.data import Data
from bayespec.likelihood import chi, cstat, pstat, pgstat, wstat
from bayespec.model.base import SpectralModel
from bayespec.plot import (
    _plot_ldata, _plot_icounts, _plot_sign_deviance
)
from bayespec.plot import plot_corner, plot_ppc
from bayespec.sampling import sample_numpyro_nuts
from iminuit import Minuit
from jacobi import propagate
from numdifftools import Gradient, Jacobian, Hessian
from pytensor import function
from scipy import stats
from scipy.optimize import minimize
from tqdm import tqdm


class Infer:
    stat_options = ('chi', 'cstat', 'pstat', 'pgstat', 'wstat')
    _stat_func = {
        'chi': chi,
        'cstat': cstat,
        'pstat': pstat,
        'pgstat': pgstat,
        'wstat': wstat,
    }
    def __init__(self, data, model, stat, seed=42):
        data_list, model_list, stat_list = self._check_input(data, model, stat)

        self.seed = seed
        self._idata = None
        self._nboot = 0
        self._data_boot = None
        self._pars_boot = None
        self._stat_boot = None
        self._stat_obs = None
        self._stat_p_boot = None
        self._stat_p_obs = None
        self._pvalue_boot = None
        self._pvalue_p_boot = None
        self._minuit = None
        self._mle_values = None
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

        self._ctx = pm.Model()

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
            stat_func(self._data[name], model, self._ctx)
            self._model[name] = model
            self._stat[name] = stat

        self._root = self._get_root()
        for root, name, default in zip(self._root['root'],
                                       self._root['name'],
                                       self._root['default']):
            self._ctx.register_rv(root, name, initval=default)

        self._rv = self._get_rv()
        for rv, name in zip(self._rv['rv'], self._rv['name']):
            pm.Deterministic(name, rv, self._ctx)

        self._values = self._get_values()

        self._params = {}
        for name in self._data_name:
            params = self._model[name].params
            rv_mask = np.array([i in params['name'] for i in self._rv['name']])
            self._params[name] = {
                'name': params['name'],
                'default': np.array(params['default']),
                'frozen': np.array(params['frozen']),
                'rv_mask': rv_mask
            }

        dof = 0
        for d in self._data.values():
            dof += d.channel.size
        dof -= len(self._root['root'])
        self._dof = dof

    def mle(self, init=None):
        if init is None:
            init = self._root_to_values(self._root['default'])
        else:
            raise NotImplementedError(
                'setting initial values is not supported yet'
            )

        res = minimize(self._deviance, init, method='L-BFGS-B', jac='3-point')
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
        self._minuit = m
        self._mle_values = np.array(m.values)
        self._stat_obs = np.hstack([
            self._deviance(self._mle_values),
            self._deviance_group(self._mle_values)
        ])
        self._stat_p_obs = self._deviance_point(self._mle_values)

        net_counts = []
        model_counts = []
        for name in self._data_name:
            net_counts.extend(self._data[name].net_counts)

            pars = self._rv_to_params(self._values_to_rv(self._mle_values), name)
            data = self._data[name]
            model_counts.extend(self._model[name].counts(
                pars, data.ph_ebins, data.ch_emin, data.ch_emax,
                data.resp_matrix, data.spec_exposure
            ))
        edf = EDFstat(np.array(net_counts), np.array(model_counts))

        k = m.npar
        n = self._dof + m.npar
        mle_info = {
            'pars': self.se(),
            'stat': m.fmin.fval,
            'dof': self._dof,
            'gof': stats.chi2.sf(m.fmin.fval, self._dof),
            'edf': edf,
            'aic': m.fmin.fval + 2 * k + 2 * k * (k + 1) / (n - k - 1),
            'bic': m.fmin.fval + k * np.log(n),
            'valid': m.fmin.is_valid,
            'edm': m.fmin.edm
        }

        return mle_info

    def se(self, pars=None, robust=False):
        if self._mle_values is None:
            self.mle()

        mle_ = self._mle_values
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

    def parametric_bootstrap(self, nboot=1000):
        if self._nboot == nboot:
            return

        self.mle()

        mle_ = self._values_to_root(self._mle_values)
        mle_dataset = xr.Dataset(
            data_vars={
                name: (('chain', 'draw'), np.full((1, nboot), mle_[i]))
                for i, name in enumerate(self._root['name'])
            },
            coords={
                'chain': ('chain', [0]),
                'draw': ('draw', np.arange(nboot))
            }
        )
        idata = az.InferenceData(posterior=mle_dataset)
        data_boot = pm.sample_posterior_predictive(
            trace=idata,
            model=self._ctx,
            random_seed=self.seed,
            progressbar=False,
            extend_inferencedata=False
        )
        bdata = {
            k.replace('_Non', '_spec_counts')
            .replace('_Noff', '_back_counts'): v.values[0]
            # [0] is because there is only one chain in idata
            for k, v in data_boot['posterior_predictive'].items()
        }

        # extract net counts from bootstrap samples
        # data_boot = []
        for name, data in self._data.items():
            counts = data_boot['posterior_predictive'][name + '_Non']
            if data.has_back and self._stat[name] != 'chi':
                back_counts = data_boot['posterior_predictive'][name + '_Noff']
                factor = data.spec_exposure / data.back_exposure
                counts = counts - factor * back_counts
            data_boot['posterior_predictive'][name + '_Net'] = counts
        #     data_boot.append(counts)
        #
        # idata['posterior_predictive']['all_channel'] = (
        #     ('chain', 'draw', 'channel'),
        #     np.concatenate(data_boot, axis=-1)
        # )

        # net_counts = []
        # model_counts = []
        # for name in self._data_name:
        #     net_counts.extend(self._data[name].net_counts)
        #
        #     pars = self._rv_to_params(self._values_to_rv(self._mle_values), name)
        #     data = self._data[name]
        #     model_counts.extend(self._model[name].counts(
        #         pars, data.ph_ebins, data.ch_emin, data.ch_emax,
        #         data.resp_matrix, data.spec_exposure
        #     ))
        # edf = EDFstat(np.array(net_counts), np.array(model_counts))

        values_boot = np.empty((nboot, len(self._mle_values)))
        stat_a_boot = np.empty(nboot)
        stat_g_boot = np.empty((nboot, len(self._data)))
        stat_p_boot = []
        flag = np.full(nboot, True)
        try:
            for i in tqdm(range(nboot), desc='Bootstrap'):
                pm.set_data({d: bdata[d][i] for d in bdata}, self._ctx)
                opt_res = minimize(fun=self._deviance,
                                   x0=self._mle_values,
                                   method='L-BFGS-B',
                                   jac='2-point')
                values_boot[i] = opt_res.x
                stat_a_boot[i] = opt_res.fun
                stat_g_boot[i] = self._deviance_group(opt_res.x)
                stat_p_boot.append(self._deviance_point(opt_res.x))
                if not opt_res.success:
                    flag[i] = False
        except Exception as e:
            raise e
        finally:
            observed = {
                k.replace('_Non', '_spec_counts')
                .replace('_Noff', '_back_counts'): v.values
                for k, v in data_boot['observed_data'].items()
            }
            pm.set_data(observed, self._ctx)

        pars_boot = self._values_to_rv(values_boot[flag])

        # net_boot = np.column_stack([
        #     data_boot['posterior_predictive'][i + '_Net'].values[0]
        #     for i in self._data_name
        # ])[flag]
        #
        # CE_boot = np.column_stack([
        #     self._model[i].counts(self._rv_to_params(pars_boot, i),
        #                           self._data[i].ph_ebins,
        #                           self._data[i].ch_emin,
        #                           self._data[i].ch_emax,
        #                           self._data[i].resp_matrix,
        #                           self._data[i].spec_exposure)
        #     for i in self._data_name
        # ])
        # edf_boot = EDFstat(net_boot, CE_boot)
        # edf_pvalue = np.sum(edf_boot >= mle_res['edf'], axis=0) / nboot_valid

        stat_boot = np.column_stack([stat_a_boot, stat_g_boot])[flag]
        stat_p_boot = [
            np.array([boot[n] for boot in stat_p_boot])[flag]
            for n in range(len(self._data))
        ]

        nboot_valid = flag.sum()
        pvalue_boot = np.sum(stat_boot >= self._stat_obs, axis=0) / nboot_valid
        pvalue_p_boot = [
            np.sum(boot >= obs, axis=0) / nboot_valid
            for obs, boot in zip(self._stat_p_obs, stat_p_boot)
        ]

        pars_boot = xr.Dataset(
            data_vars={
                name: (
                    ('chain', 'draw'), np.expand_dims(pars_boot[:, i], axis=0))
                for i, name in enumerate(self._rv['name'])
            },
            coords={
                'chain': ('chain', [0]),
                'draw': ('draw', np.arange(flag.sum()))
            }
        )

        draws = np.flatnonzero(flag)
        self._data_boot = data_boot['posterior_predictive'].where(
            data_boot['posterior_predictive']['draw'].isin(draws),
            drop=True
        )
        self._pars_boot = pars_boot
        self._stat_boot = stat_boot
        self._pvalue_boot = pvalue_boot
        self._stat_p_boot = stat_p_boot
        self._pvalue_p_boot = pvalue_p_boot
        self._nboot = nboot

    def ci(self, method='hdi', cl=1.0, pars=None, nboot=1000):
        pars_name = self._get_pars_name(pars)

        if self._minuit is None:
            self.mle()

        if cl >= 1.0:
            cl = 1.0 - stats.norm.sf(cl) * 2.0

        m = self._minuit
        mle_ = self._mle_values
        mle = self._values_to_rv(mle_)

        if method == 'hdi':
            if self._idata is None:
                raise ValueError(
                    'run MCMC before calculating credible interval'
                )

            # hdi = az.hdi(self._idata, cl)
            # lower_ = hdi.sel(hdi='lower')
            # lower = [lower_[i].values for i in self._rv['name']]
            # upper_ = hdi.sel(hdi='higher')
            # upper = [upper_[i].values for i in self._rv['name']]

            ci_ = self._idata['posterior'].quantile(
                q=[0.5 - cl / 2.0, 0.5 + cl / 2.0],
                dim=['chain', 'draw']
            )
            lower_ = ci_.sel(quantile=0.5 - cl / 2.0)
            lower = [lower_[i].values for i in self._rv['name']]
            upper_ = ci_.sel(quantile=0.5 + cl / 2.0)
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
            self._mle_values = np.array(m.values)

            mle_ = self._mle_values
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
            if self._pars_boot is None:
                self.parametric_bootstrap(nboot)

            # bci = az.hdi(self._pars_boot, cl)
            # lower_ = bci.sel(hdi='lower')
            # lower_ = [lower_[i].values for i in self._rv['name']]
            # upper_ = bci.sel(hdi='higher')
            # upper_ = [upper_[i].values for i in self._rv['name']]
            # lower = lower_
            # upper = upper_

            ci_ = self._pars_boot.quantile(
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

    def mcmc_nuts(
        self,
        draws=20000,
        tune=2000,
        init_mle=True,
        jitter=False,
        chains=4,
        target_accept=0.8
    ):
        if init_mle:
            if self._mle_values is None:
                self.mle()

            mle_root = i._values_to_root(i._mle_values)
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
                                    model=self._ctx)

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

    def gof(self, nsim=1000, cl=0.95):
        if self._idata is None:
            raise ValueError('run MCMC before ppc')

        idata = self._idata

        if 'posterior_predictive' not in self._idata.groups():
            pm.sample_posterior_predictive(
                trace=idata,
                model=self._ctx,
                random_seed=self.seed,
                progressbar=True,
                extend_inferencedata=True
            )
            net_rep = []
            for name, data in self._data.items():
                counts = idata['posterior_predictive'][name+'_Non']
                if data.has_back and self._stat[name] != 'chi':
                    back_counts = idata['posterior_predictive'][name+'_Noff']
                    factor = data.spec_exposure / data.back_exposure
                    counts = counts - factor * back_counts
                idata['posterior_predictive'][name+'_Net'] = counts
                net_rep.append(counts)

            idata['posterior_predictive']['all_channel'] = (
                ('chain', 'draw', 'channel'),
                np.concatenate(net_rep, axis=-1)
            )

        posterior = idata['posterior']
        predictive = idata['posterior_predictive']

        rng = np.random.default_rng(self.seed)
        idx = rng.integers(low=[[0], [0]],
                           high=[[posterior.chain.size],
                                 [posterior.draw.size]],
                           size=(2, nsim))
        i, j = idx

        pdata = {
            f'{name}_{d2}_counts': predictive[f'{name}_N{d1}'].values[i, j]
            for name in self._data_name
            for d1, d2 in zip(['on', 'off'], ['spec', 'back'])
            if f'{name}_N{d1}' in predictive.keys()
        }
        pred_net = predictive['all_channel'].values[i, j]

        post = [posterior[r].values[i, j] for r in self._root['name']]
        post = np.ascontiguousarray(np.transpose(post))

        D = -2.0 * idata.log_likelihood['total'].values[i, j]
        D_rep = np.empty(nsim)
        D_min = np.empty(nsim)

        net_counts = np.hstack([self._data[i].net_counts for i in self._data_name])
        EDF = np.empty((nsim, 4))
        EDF_rep = np.empty((nsim, 4))
        EDF_min = np.empty((nsim, 4))

        flag = np.full(nsim, True)
        y_hat = {
            name: np.empty((nsim, data.channel.size))
            for name, data in self._data.items()
        }
        try:
            for i in tqdm(range(nsim), desc='Fitting'):
                pm.set_data({d: pdata[d][i] for d in pdata}, self._ctx)
                D_rep[i] = self._deviance(self._root_to_values(post[i]))
                opt_res = minimize(fun=self._deviance,
                                   x0=self._root_to_values(post[i]),
                                   method='L-BFGS-B',
                                   jac='2-point')
                D_min[i] = opt_res.fun

                model_counts = []
                model_fit_counts = []
                for name in self._data_name:
                    pars = self._rv_to_params(self._root_to_rv(post[i]), name)
                    pars_fit = self._rv_to_params(self._values_to_rv(opt_res.x), name)
                    data = self._data[name]
                    model_counts_i = self._model[name].counts(
                        pars, data.ph_ebins, data.ch_emin, data.ch_emax,
                        data.resp_matrix, data.spec_exposure
                    )
                    y_hat[name][i] = model_counts_i
                    model_fit_counts_i = self._model[name].counts(
                        pars_fit, data.ph_ebins, data.ch_emin, data.ch_emax,
                        data.resp_matrix, data.spec_exposure
                    )
                    model_counts.extend(model_counts_i)
                    model_fit_counts.extend(model_fit_counts_i)

                model_counts = np.array(model_counts)
                model_fit_counts = np.array(model_fit_counts)
                rep_counts = pred_net[i]

                EDF[i] = EDFstat(net_counts, model_counts)
                EDF_rep[i] = EDFstat(rep_counts, model_counts)
                EDF_min[i] = EDFstat(rep_counts, model_fit_counts)

                if not opt_res.success:
                    flag[i] = False

            D_min = D_min[flag]
            EDF_min = EDF_min[flag]

        except Exception as e:
            raise e
        finally:
            observed = {
                f'{i}_{k}_counts': idata['observed_data'][f'{i}_N{j}'].values
                for i in self._data_name
                for j, k in zip(['on', 'off'], ['spec', 'back'])
                if f'{i}_N{j}' in idata['observed_data'].keys()
            }
            pm.set_data(observed, self._ctx)

        mle_res = self.mle()
        D_best = mle_res['stat']
        EDF_best = mle_res['edf']

        x = []
        y = []
        y_rep = []
        y_hat_list = []
        i, j = idx
        for name, data in self._data.items():
            x.append(np.column_stack([data.ch_emin, data.ch_emax]))
            y.append(data.net_counts)
            y_rep.append(predictive[f'{name}_Net'].values[i, j])
            y_hat_list.append(y_hat[name])

        res = plot_ppc(
            x, y, y_hat_list, D_best, self._dof, y_rep, D, D_rep, D_min, '$D$',
            cl=cl, xlabel='Energy [keV]', colors=None
        )

        # for i, stat in enumerate(['ks', 'cvm', 'ad', 'cusum']):
        #     res_ = plot_ppc(
        #         x, y, EDF_best[i], self._dof, y_rep, EDF[:, i], EDF_rep[:, i], EDF_min[:, i], stat,
        #         cl=cl, xlabel='Energy [keV]', colors=None
        #     )

        az.plot_loo_pit(idata, 'all_channel', ecdf=True, hdi_prob=cl)

        return res

    # def plot_ppc(self, nsim=1000, cl=0.95):
    #     if self._idata is None:
    #         raise ValueError('run MCMC before ppc')
    #
    #     idata = draw_posterior_samples(self._idata, nsim, self.seed)
    #     lnL = idata['log_likelihood']['total'].values[0]
    #
    #     pm.sample_posterior_predictive(
    #         trace=idata,
    #         model=self._ctx,
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
    #             pm.set_data({d: pdata[d][i] for d in pdata}, self._ctx)
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
    #         pm.set_data(observed, self._ctx)
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
    #     if self._minuit is None:
    #         self.mle()
    #     Dmin_obs = self._minuit.fmin.fval
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
        self, fmt='ldata delchi icounts',
        comps=True, show_pars=None, fig_path=None, c=1
    ):
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
        with (plt.style.context(['nature', 'science'])):
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['font.serif'] = 'Times New Roman'
            plt.rcParams['text.latex.preamble'] += r' \usepackage{mathptmx}'
            fig, axes = plt.subplots(
                3, 1,
                sharex=True,
                gridspec_kw={'height_ratios': [1.6, 1, 1]},
                figsize=[4 * 1.5, 3 * 1.5],
                dpi=200
            )
            fig.subplots_adjust(hspace=0, wspace=0)
            fig.align_ylabels(axes)
            mle = self.mle()

            if show_pars is not None:
                pars_info = []
                for pname in self._get_pars_name(show_pars):
                    best, err = mle['pars'][pname]
                    pars_info.append(
                        rf'{pname}: {best:.2f}$\pm${err:.2f}'
                    )
                axes[0].annotate('\n'.join(pars_info),
                                 xy=(0.04, 0.05), xycoords='axes fraction',
                                 ha='left', va='bottom')

            for i, data_name in enumerate(self._data_name):
                if self._data_boot is not None:
                    data_boot = self._data_boot[f'{data_name}_Net'].values[0]
                    pars_boot = self._pars_boot.to_array().values
                    pars_boot = np.transpose(pars_boot, axes=(1, 2, 0))[0]
                    pars_boot = np.ascontiguousarray(pars_boot)
                    pars_boot = self._rv_to_params(pars_boot, data_name)
                    deviance_boot = self._stat_p_boot[i]
                else:
                    data_boot = None
                    pars_boot = None
                    deviance_boot = None

                color = colors[i]
                marker = markers[i]
                data = self._data[data_name]
                model = self._model[data_name]
                mle_rv = self._values_to_rv(self._mle_values)
                pars = self._rv_to_params(mle_rv, data_name)

                _plot_ldata(
                    axes[0], data, model, pars, marker, color, 0.6, True,
                    sim_pars=pars_boot
                )
                # _plot_residuals(axes[1], data, model, pars, marker, color,0.6, data_boot, pars_boot)
                _plot_sign_deviance(axes[1], data, model, pars, marker, color, 0.6,
                                    self._stat_p_obs[i], data_boot, pars_boot, deviance_boot)
                # _plot_ratio(axes[1], data, model, pars, marker, color, 0.6, data_boot, pars_boot)
                # _plot_icounts_residual(axes[2], data, model, pars, marker, color, 0.6, data_boot, pars_boot)
                _plot_icounts(axes[2], data, model, pars, marker, color, 0.6, data_boot, pars_boot)
                # _plot_delchi(axes[1], data, model, pars, marker, color,0.6, data_boot, pars_boot)
                # _plot_ratio(axes[2], data, model, pars, marker, color, 0.6, data_boot, pars_boot)

            xlim_left = np.min([d.ch_emin.min() for d in self._data.values()])
            xlim_right = np.max([d.ch_emax.max() for d in self._data.values()])
            axes[0].set_xlim(xlim_left*0.93, xlim_right*1.07)

            # axes[2].set_ylim(bottom=0.0)

            axes[1].axhline(0, ls='--', c='gray', zorder=0, lw=1)
            axes[2].axhline(0, ls='--', c='gray', zorder=0, lw=1)
            axes[0].set_ylabel('$C_E$ [s$^{-1}$ keV$^{-1}$]')
            # axes[1].set_ylabel('(data$-$model)$/$error')
            # axes[1].set_ylabel('data $-$ model')
            axes[1].set_ylabel('sign * Deviance$^{1/2}$')
            # axes[2].set_ylabel('icounts $-$ imodel')
            axes[2].set_ylabel('integrated counts')
            axes[-1].set_xlabel('Energy [keV]')
            axes[0].legend(frameon=True, loc='upper right')
            axes[0].set_xscale('log')

            title = f'{model.expression}'
            title += f', stat/dof={mle["stat"]:.2f}/{mle["dof"]}'
            if self._pvalue_boot is not None:
                title += f', Bootstrap $p$-value={self._pvalue_boot[0]:.3f}'
            axes[0].set_title(title)

            if fig_path is not None:
                fig.savefig(fig_path)

            return fig, axes

    # def plot_spec(self, show_pars=None, fig_path=None):
    #     # colors = ["#0d49fb", "#e6091c", "#26eb47", "#8936df", "#fec32d", "#25d7fd"]
    #     # colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377',
    #     #           '#BBBBBB']
    #     colors = [
    #         '#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97',
    #         '#474747', '#9e9e9e'
    #     ]
    #     markers = ['s', 'o', 'D', '^', 'd', 'p', 'h', 'H', 'D']
    #     plt.style.use(['nature', 'science', 'no-latex'])
    #     fig, axes = plt.subplots(
    #         3, 1,
    #         sharex=True,
    #         gridspec_kw={'height_ratios': [1.6, 1, 1]},
    #         figsize=[4 * 1.5, 3 * 1.5],
    #         dpi=100
    #     )
    #     fig.subplots_adjust(hspace=0, wspace=0)
    #     fig.align_ylabels(axes)
    #
    #     mle = self.mle()
    #
    #     if show_pars is not None:
    #         pars_info = []
    #         for pname in self._get_pars_name(show_pars):
    #             best, err = mle['pars'][pname]
    #             pars_info.append(
    #                 rf'{pname}: {best:.2f}$\pm${err:.2f}'
    #             )
    #         axes[0].annotate('\n'.join(pars_info),
    #                          xy=(0.04, 0.05), xycoords='axes fraction',
    #                          ha='left', va='bottom')
    #
    #     for i, data_name in enumerate(self._data_name):
    #         color = colors[i]
    #         marker = markers[i]
    #         d = self._data[data_name]
    #         m = self._model[data_name]
    #         mle_rv = self._values_to_rv(self._mle_values)
    #         pars = self._rv_to_params(mle_rv, data_name)
    #         other = [d.ph_ebins, d.ch_emin, d.ch_emax, d.resp_matrix]
    #         CE = m.CE(pars, *other)
    #         CE_comps = m.CE(pars, *other, comps=True)
    #
    #         if f'{data_name}_f' in mle:
    #             factor = mle[f'{data_name}_f']
    #             label = f'{data_name} ({factor:.2f})'
    #         else:
    #             label = f'{data_name}'
    #
    #         emid = d.ch_emid_geom
    #         eerr = np.abs([d.ch_emin, d.ch_emax] - d.ch_emid_geom)
    #         net = d.net_spec
    #         net_err = d.net_error
    #
    #         uplims = net < 0.0
    #         upper = np.zeros_like(net)
    #         upper[uplims] = net_err[uplims] * 3
    #
    #         alpha = 0.6
    #         color = color + hex(round(255 * alpha))[2:]
    #         mfc = '#FFFFFFCC'
    #         axes[0].errorbar(
    #             emid, net+upper, net_err, eerr,
    #             fmt=f'{marker} ', c=color, label=label, ms=2.5, mfc=mfc,
    #             uplims=uplims, mec=color, capsize=1
    #         )
    #
    #         mask = d.ch_emin[1:] != d.ch_emax[:-1]
    #         idx = [0, *(np.flatnonzero(mask) + 1), len(d.channel)]
    #         net_cumsum = np.cumsum(d.net_spec * d.ch_width * d.spec_exposure)
    #         CE_cumsum = np.cumsum(CE * d.ch_width * d.spec_exposure)
    #         plot_CE_cumsum = net_cumsum[-1] > 0.0
    #         CE_cumsum /= net_cumsum[-1]
    #         net_cumsum /= net_cumsum[-1]
    #         for k in range(len(idx) - 1):
    #             slice_k = slice(idx[k], idx[k + 1])
    #             ch_ebins_k = np.append(d.ch_emin[slice_k],
    #                                    d.ch_emax[slice_k][-1])
    #             CE_k = CE[slice_k]
    #             CE_k = np.append(CE_k, CE_k[-1])
    #             axes[0].step(ch_ebins_k, CE_k, where='post', c=color, alpha=alpha, lw=1.3)
    #             if len(CE_comps.keys()) > 1:
    #                 for CE_comp in CE_comps.values():
    #                     CE_comp_k = CE_comp[slice_k]
    #                     CE_comp_k = np.append(CE_comp_k, CE_comp_k[-1])
    #                     axes[0].step(ch_ebins_k, CE_comp_k, where='post', c=color,
    #                                  alpha=alpha, lw=1.3, ls='--')
    #             if plot_CE_cumsum:
    #                 CE_cumsum_k = CE_cumsum[slice_k]
    #                 CE_cumsum_k = np.append(CE_cumsum_k, CE_cumsum_k[-1])
    #                 axes[1].step(ch_ebins_k, CE_cumsum_k, where='post',
    #                              c=color, alpha=alpha, lw=1.3)
    #         axes[1].errorbar(
    #             emid, net_cumsum, xerr=eerr,
    #             fmt=f'{marker} ', c=color, mec=color, label=label, ms=2.5, mfc=mfc, capsize=1
    #         )
    #
    #         axes[2].errorbar(
    #             emid, (net - CE) / net_err, 1, eerr,
    #             fmt=f'{marker} ', c=color, label=label, ms=2.5, mfc=mfc, mec=color, capsize=1
    #         )
    #     axes[0].loglog()
    #     # ylim = axes[0].get_ylim()
    #     # if ylim[1]/ylim[0] > 1e7:
    #     #     axes[0].set_ylim(bottom=ylim[1]/1e7)
    #     axes[1].set_ylabel('EDF')
    #     axes[1].set_ylim(bottom=0.0)
    #     axes[2].axhline(0, ls='--', c='gray', zorder=0, lw=1)
    #     # axes[2].set_ylabel('(data$\,-\,$model)$\,$/$\,$error')
    #     axes[2].set_ylabel('$\chi$')
    #     axes[-1].set_xlabel('Energy [keV]')
    #     axes[0].set_ylabel('$C_E$ [s$^{-1}$ keV$^{-1}$]')
    #
    #     # axes[1].set_yticks([-3,-2,-1,0,1,2,3])
    #     # axes[1].set_yticklabels([-3,-2,-1,0,1,2,3], ha='center')
    #     axes[0].legend(
    #         loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=10,
    #         fancybox=True, shadow=True, ncol=6, frameon=True,
    #         columnspacing=1, handletextpad=0.3
    #     )
    #     for ax in axes:
    #         ax.tick_params(axis='both', which='both', direction='in',
    #                        top=True,
    #                        right=True)
    #     # axes[0].set_xlim(data.ch_ebins.min()*0.95, data.ch_ebins.max()*1.05)
    #
    #     # axes[1].set_ylim(-3.9, 3.9)
    #     # ylim = np.abs(axes[1].get_ylim()).max()
    #     # axes[1].set_ylim(-ylim, ylim)
    #
    #     if fig_path is not None:
    #         fig.savefig(fig_path)
    #
    #     plt.style.use('default')

    def plot_trace(self, root=False, back=False):
        if self._idata is None:
            raise ValueError('run MCMC before plotting trace')

        var_names = list(self._rv['name'])
        if root:
            var_names.extend(self._root['name'])
        if back:
            var_names.extend(
                [p for p in self._idata.posterior.data_vars if '_BKG' in p]
            )

        az.plot_trace(self._idata, var_names=var_names)

    def plot_corner(
            self, profile=True, root=False, level_idx=3,
            smooth=0.0, fig_path=None, **kwargs
    ):
        if self._idata is None:
            raise ValueError('run MCMC before plotting corner')

        var_names = list(self._rv['name'])
        if root:
            var_names.extend(self._root['name'])

        plot_corner(
            self._idata, var_names, profile, level_idx, smooth, fig_path,
            **kwargs
        )


    def loo(self, scale='deviance'):
        if self._idata is None:
            raise ValueError('run MCMC before calculating PSIS-LOO-CV')

        res = az.loo(self._idata, var_name='all_channel', scale=scale)[:3]
        return tuple(res)

    def waic(self, scale='deviance'):
        if self._idata is None:
            raise ValueError('run MCMC before calculating WAIC')

        res = az.waic(self._idata, var_name='all_channel', scale=scale)[:3]
        return tuple(res)

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
        return [self._ctx.rvs_to_values[r] for r in self._root['root']]

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
            self.__deviance = function(self._values, -2*self._ctx.observedlogp)

            # f = function(self._values, -2 * self._ctx.observedlogp)
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
                for i in self._ctx.logp(self._ctx.observed_RVs, sum=False)
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
                for i in self._ctx.logp(self._ctx.observed_RVs, sum=False)
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
            compiled = function(self._values, self._ctx.observedlogp)
            self.__lnL = compiled

        return self.__lnL(*values)

    def _values_covar(self, values):
        if self.__values_covar is None:
            hess = Hessian(self._lnL)
            self.__values_covar = lambda x: np.linalg.inv(-hess(x))

        return self.__values_covar(values)

    # pytensor version
    # def _values_robust_covar(self, values):
    #     ctx = self._ctx
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
            ctx = self._ctx

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
            transforms = self._ctx.rvs_to_transforms
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
            transforms = self._ctx.rvs_to_transforms
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
                rv = np.asarray(rv)
                params = self._params[data_name]
                frozen = params['frozen']
                params_values = np.array(params['default'])
                params_values[~frozen] = rv[params['rv_mask']]
                return params_values

            self.__rv_to_params = np.vectorize(
                rv_to_params,
                otypes=[np.float64],
                excluded={'data_name'},
                signature='(i)->(j)'
            )

        return self.__rv_to_params(rv_values, data_name=data_name)


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


def draw_posterior_samples(idata, nsample, seed):
    posterior = idata.posterior
    rng = np.random.default_rng(seed)
    i, j = rng.integers(low=[[0], [0]],
                        high=[[posterior.chain.size], [posterior.draw.size]],
                        size=(2, nsample))

    coords = {
        'chain': ('chain', [0]),
        'draw': ('draw', np.arange(nsample))
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
            'draw': ('draw', np.arange(nsample))
        }
    )
    idata2 = az.InferenceData(posterior=posterior_dataset)

    if 'log_likelihood' in idata.groups():
        log_likelihood = idata.log_likelihood
        coords = {
            'chain': ('chain', [0]),
            'draw': ('draw', np.arange(nsample))
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
    # from bayespec import xs, CutoffPowerlaw, UniformParameter
    # src = CutoffPowerlaw() + xs.posm()
    # # src.CPL.norm.log = True
    # src.CPL.norm = src.posm.norm * UniformParameter('f', 10, 1e-5, 1e5, log=1)
    # src.CPL.Ec.log = True
    # i = Infer([NaI_data, CsI_data], src, ['wstat', 'pgstat'])
    # print(i.mle())
    # i.plot_spec()
    # i.mcmc_nuts(1000, 1000)

    path = '/Users/xuewc/BurstData/FRB221014/HXMT/'
    LE = Data([1.5, 10],
              f'{path}/LE_optbmin5.fits',
              f'{path}/LE_phabkg20s_g0_0-94.pha',
              f'{path}/LE_rsp.rsp',
              group_type='bmin', group_scale=25)

    ME = Data([10, 35],
              f'{path}/ME_optbmin5.fits',
              f'{path}/ME_phabkg20s_g0_0-53.pha',
              f'{path}/ME_rsp.rsp',
              group_type='bmin', group_scale=25)

    HE = Data([28, 250],
              f'{path}/HE_optbmin5.fits',
              f'{path}/HE_phabkg20s_g0_0-12.pha',
              f'{path}/HE_rsp.rsp',
              group_type='bmin', group_scale=25)

    from bayespec import Powerlaw, xs

    wabs = xs.wabs(2.79)
    src = Powerlaw()
    src.PL.norm.log = 1
    # src = BlackBodyRad()
    # src.BBrad.norm.log = 1
    # src.BBrad.kT.log = 1
    # src = OOTB()
    # src.OOTB.kT.log = 1
    # src.OOTB.norm.log = 1
    # src = BlackBody() + BlackBody()
    # src.BB_2.kT = src.BB.kT * UniformParameter('factor', 0.5, 0.001, 0.999)
    i = Infer([LE, ME, HE], wabs*src, 'wstat')
    # i.parametric_bootstrap(1000)
    # i.plot_data(show_pars=i._rv['name'])
    i.mcmc_nuts()
