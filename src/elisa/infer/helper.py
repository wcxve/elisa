"""Helper for fitting and analysis."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import optimistix as optx
from jax import lax
from numpyro import handlers
from numpyro.infer.util import constrain_fn, unconstrain_fn

from elisa.infer.likelihood import (
    _STATISTIC_BACK_NORMAL,
    _STATISTIC_SPEC_NORMAL,
    _STATISTIC_WITH_BACK,
    chi2,
    cstat,
    pgstat,
    pstat,
    wstat,
)
from elisa.util.misc import (
    get_parallel_number,
    get_unit_latex,
    progress_bar_factory,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Callable, Literal

    from numpyro.distributions import Distribution

    from elisa.data.base import FixedData
    from elisa.infer.fit import Fit
    from elisa.infer.likelihood import Statistic
    from elisa.models.model import CompiledModel, ModelInfo, ParamSetup
    from elisa.util.typing import (
        JAXArray,
        JAXFloat,
        ParamID,
        ParamName,
        ParamNameValMapping,
    )


def check_params(
    params: str | Sequence[str] | None, helper: Helper
) -> list[str]:
    params_names = helper.params_names

    all_params = set(params_names['all']) | set(helper.params_setup)
    forwarded = {
        k: v[0]
        for k, v in helper.params_setup.items()
        if v[1].name == 'Forwarded'
    }
    fixed = [k for k, v in helper.params_setup.items() if v[1].name == 'Fixed']
    integrated = [
        k for k, v in helper.params_setup.items() if v[1].name == 'Integrated'
    ]

    if params is None:
        params = set(params_names['interest'])

    elif isinstance(params, str):
        # check if params exist
        if params not in all_params:
            raise ValueError(f'parameter {params} is not exist')

        params = {params}

    elif isinstance(params, Iterable):
        # check if params exist
        params = {str(i) for i in params}
        if not params.issubset(all_params):
            params_err = params - set(params_names['all'])
            raise ValueError(f'parameters: {params_err} are not exist')

    else:
        raise ValueError('params must be str, or sequence of str')

    if params_err := params.intersection(forwarded):
        forwarded = {i: forwarded[i] for i in params_err}
        info = ', '.join(f'{k} to {v}' for k, v in forwarded.items())
        raise RuntimeError(
            f"parameters are linked: {info}; corresponding parameters' "
            'name should be used'
        )

    if params_err := params.intersection(fixed):
        info = ', '.join(params_err)
        raise RuntimeError(f'parameters are fixed: {info}')

    if params_err := params.intersection(integrated):
        info = ', '.join(params_err)
        raise RuntimeError(f'parameters are integrated-out: {info}')

    return sorted(params, key=params_names['all'].index)


# def get_reparam(dist: Distribution) -> tuple[Reparam, Callable] | None:
#     """Get reparam for a distribution."""
#     # TODO: support more reparameterizers
#     if isinstance(dist, (Normal, StudentT, Cauchy)):
#         param = '_centered'
#         suffix = '_decentered'
#         reparam = LocScaleReparam()
#     elif isinstance(dist, TransformedDistribution):
#         suffix = '_base'
#         reparam = TransformReparam()
#         transforms = dist.transforms
#         def inv(x):
#             """Inverse transformation."""
#             for t in reversed(transforms):
#                 x = t.inv(x)
#             return x
#     elif isinstance(dist, ProjectedNormal):
#         suffix = '_normal'
#         reparam = ProjectedNormalReparam()
#     elif isinstance(dist, VonMises):
#         suffix = '_unwrapped'
#         reparam = CircularReparam()
#     else:
#         return None
#
#     return reparam, inv


def get_helper(fit: Fit) -> Helper:
    """Get helper functions for fitting."""
    model_info: ModelInfo = fit._model_info
    data: dict[str, FixedData] = fit._data
    model: dict[str, CompiledModel] = fit._model
    stat: dict[str, Statistic] = fit._stat
    seed0 = fit._seed
    rng_seed: dict[str, int] = {
        'mcmc': seed0,  # for MCMC
        'pred': seed0 + 1,  # for data simulation
        'resd': seed0 + 2,  # for random quantile residuals
    }

    # channel number of data
    ndata = {k: v.channel.size for k, v in data.items()}
    ndata['total'] = sum(ndata.values())

    # channel information
    channels = {f'{k}_channel': v.channel for k, v in data.items()}
    channels['channel'] = np.hstack(list(channels.values()))

    # number of free parameters
    nparam = len(model_info.sample)

    # degree of freedom
    dof = ndata['total'] - nparam

    # ======================== count data calculator ==========================
    on_names = [f'{i}_Non' for i in data]
    off_names = [f'{i}_Noff' for i in data if stat[i] in _STATISTIC_WITH_BACK]
    back_ratio = {
        k: v.back_ratio
        for k, v in data.items()
        if stat[k] in _STATISTIC_WITH_BACK
    }
    spec_unit = {
        k: 1.0 / (v.spec_exposure * v.channel_width) for k, v in data.items()
    }

    @jax.jit
    def get_counts_data(counts: dict[str, JAXArray]) -> dict[str, JAXArray]:
        """Get count data, including "on", "off" and net counts."""
        counts = {k: jnp.asarray(v, float) for k, v in counts.items()}
        assert set(counts) == set(on_names + off_names)

        # counts in the "on" measurement of each dataset
        counts_data = {i: counts[i] for i in on_names}

        # counts in the "off" measurement of each dataset
        counts_data |= {i: counts[i] for i in off_names}

        net_counts = {
            i: (
                counts[f'{i}_Non']
                - back_ratio.get(i, 0.0) * counts.get(f'{i}_Noff', 0.0)
            )
            for i in data
        }

        # net spectrum [count s^-1 keV^-1]
        counts_data |= {i: net_counts[i] * spec_unit[i] for i in data.keys()}

        # stack net spectrum of all channels of all datasets
        counts_data['channels'] = jnp.concatenate(
            [counts_data[i] for i in data.keys()],
            axis=-1,
        )

        # total net counts of datasets
        counts_data['total'] = jnp.sum(
            jnp.asarray([i.sum(axis=-1) for i in net_counts.values()]), axis=0
        )

        return counts_data

    # ======================== count data calculator ==========================

    obs_counts = {
        f'{k}_Non': (
            v.net_counts
            if stat[k] in _STATISTIC_SPEC_NORMAL
            else v.spec_counts
        )
        for k, v in data.items()
    }
    obs_counts |= {
        f'{k}_Noff': v.back_counts
        for k, v in data.items()
        if stat[k] in _STATISTIC_WITH_BACK
    }
    obs_data = get_counts_data(obs_counts)

    # ======================== count data simulator ===========================
    def simulator_factory(data_dist: Literal['norm', 'poisson'], *dist_args):
        """Factory to create data simulator."""

        def simulator(
            rng: np.random.Generator,
            model_values: np.ndarray,
            n: int,
        ):
            """Simulate data given random number generator and model values."""
            if n != 1:
                shape = (n,) + model_values.shape
            else:
                shape = model_values.shape

            if data_dist == 'norm':
                # TODO: fix the negative counts by setting them to zeros
                return rng.normal(model_values, *dist_args, shape)
            elif data_dist == 'poisson':
                return rng.poisson(model_values, shape)
            else:
                raise NotImplementedError(f'{data_dist = }')

        return simulator

    simulators = {}
    sampling_dist: dict[str, tuple[Literal['norm', 'poisson'], tuple]] = {}
    for k, s in stat.items():
        d = data[k]

        name = f'{k}_Non'
        if s in _STATISTIC_SPEC_NORMAL:
            simulators[name] = simulator_factory('norm', d.spec_errors)
            sampling_dist[name] = ('norm', (d.spec_errors,))
        else:
            simulators[name] = simulator_factory('poisson')
            sampling_dist[name] = ('poisson', ())

        if s in _STATISTIC_WITH_BACK:
            name = f'{k}_Noff'
            if s in _STATISTIC_BACK_NORMAL:
                simulators[name] = simulator_factory('norm', d.back_errors)
                sampling_dist[name] = ('norm', (d.spec_errors,))
            else:
                simulators[name] = simulator_factory('poisson')
                sampling_dist[name] = ('poisson', ())

    def simulate(
        rng_seed: int,
        model_values: dict[str, JAXArray],
        n: int = 1,
    ) -> dict[str, JAXArray]:
        """Simulate data given model values.
        Use numpy.random instead of numpyro.infer.Predictive for performance.
        """
        models = {i: model_values[f'{i}_model'] for i in simulators.keys()}
        rng = np.random.default_rng(int(rng_seed))
        sim = {k: v(rng, models[k], n) for k, v in simulators.items()}
        return get_counts_data(sim)

    # ======================== count data simulator ===========================

    # ======================== create numpyro model ===========================
    pname_to_latex: dict[ParamName, str] = {
        pname: model_info.latex[pid] for pid, pname in model_info.name.items()
    }
    pname_to_log: dict[ParamName, bool] = {
        pname: model_info.log[pid] for pid, pname in model_info.name.items()
    }

    pname_to_unit: dict[ParamName, str] = {
        pname: get_unit_latex(model_info.unit[pid], throw=False)
        for pid, pname in model_info.name.items()
    }
    pname_to_comp_latex: dict[ParamName, str] = {
        pname: model_info.pid_to_comp_latex[pid]
        for pid, pname in model_info.name.items()
    }

    # get model parameters priors
    pid_to_pname: dict[ParamID, ParamName] = model_info.name
    pname_to_pid: dict[ParamName, ParamID] = {
        v: k for k, v in pid_to_pname.items()
    }
    pid_to_prior: dict[ParamID, Distribution] = model_info.sample
    params_prior: dict[ParamName, Distribution] = {
        pid_to_pname[pid]: pid_to_prior[pid] for pid in pid_to_prior
    }

    # get deterministic value getter function
    deterministic: dict[ParamID, Callable] = model_info.deterministic

    # get the likelihood function for each dataset
    likelihood_wrapper = {
        'chi2': chi2,
        'cstat': cstat,
        'pstat': pstat,
        'wstat': wstat,
        'pgstat': pgstat,
    }
    likelihood: dict[str, Callable[[JAXArray], None]] = {
        k: likelihood_wrapper[stat[k]](v, model[k].eval)
        for k, v in data.items()
    }

    # get default re-parameterization of each parameter
    # reparams: dict[ParamName, tuple[Reparam, Callable]] = {
    #     name: reparam_and_inv
    #     for name, prior in params_prior.items  ()
    #     if (reparam_and_inv := get_reparam(prior)) is not None
    # }

    def numpyro_model(predictive: bool = False) -> None:
        """The numpyro model for spectral fitting."""
        # TODO:
        #  figure out how to handle reparam transformation, so we can
        #       * give initial parameter value in the original space and
        #         transformed to repameterized space,
        #       * find the classic confidence interval in the repameterized
        #         space and then transform back to the original space.
        #  This is not trivial because transformation is not always bijective!
        #  The trick to handle confidence interval of composite parameter may
        #  also be used here to solve the second problem.
        # with numpyro.handlers.reparam(config=reparams):
        #     params_name_values = {
        #         name: numpyro.sample(name, dist)
        #         for name, dist in params_prior.items()
        #     }

        # get parameter value from prior
        params_name_values = {
            name: numpyro.sample(name, dist)
            for name, dist in params_prior.items()
        }
        params_id_values = {
            pname_to_pid[name]: value
            for name, value in params_name_values.items()
        }

        # store composite parameters into chains
        for pid, fn in deterministic.items():
            numpyro.deterministic(pid_to_pname[pid], fn(params_id_values))

        # the likelihood between observation and model for each dataset
        jax.tree_map(
            lambda f: f(params_name_values, predictive=predictive),
            likelihood,
        )

    # ======================== create numpyro model ===========================

    # =================== functions used in optimization ======================
    params_names = [
        f'{i[1]}.{i[2]}' if i[1] else i[2] for i in model_info.info if i[0]
    ]
    interest_names = [f'{i[1]}.{i[2]}' for i in model_info.info if all(i[:2])]
    free_names = sorted(params_prior.keys(), key=params_names.index)
    deterministic_names = [pid_to_pname[i] for i in deterministic]
    deterministic_names = sorted(deterministic_names, key=params_names.index)

    # ensure if names are consistent
    if set(free_names + deterministic_names) != set(params_names):
        raise RuntimeError(
            f'{params_names = }, {free_names = }, {deterministic_names = }'
        )

    data_group = {
        k: (f'{k}_Non', f'{k}_Noff')
        if v in _STATISTIC_WITH_BACK
        else (f'{k}_Non',)
        for k, v in stat.items()
    }

    @jax.jit
    def arr_to_dic(arr: JAXArray) -> ParamNameValMapping:
        """Covert free parameters from an array to a dict."""
        assert len(arr) == len(free_names)
        return dict(zip(free_names, arr))

    @jax.jit
    def dic_to_arr(dic: ParamNameValMapping) -> JAXArray:
        """Covert free parameters from a dict to an array."""
        return jnp.array([dic[i] for i in free_names], float)

    @jax.jit
    def unconstr_arr_to_constr_dic(arr: JAXArray) -> ParamNameValMapping:
        """Covert free parameters array from unconstrained space to dict in
        constrained space.
        """
        return constrain_fn(
            model=numpyro_model,
            model_args=(),
            model_kwargs={},
            params=arr_to_dic(arr),
        )

    @jax.jit
    def constr_arr_to_unconstr_dic(arr: JAXArray) -> ParamNameValMapping:
        """Covert free parameters array from constrained space to dict in
        unconstrained space.
        """
        return unconstrain_fn(
            model=numpyro_model,
            model_args=(),
            model_kwargs={},
            params=arr_to_dic(arr),
        )

    @jax.jit
    def constr_arr_to_unconstr_arr(arr: JAXArray) -> JAXArray:
        """Covert free parameters array from constrained space into
        unconstrained space.
        """
        unconstr_dic = constr_arr_to_unconstr_dic(arr)
        return jnp.asarray([unconstr_dic[i] for i in free_names])

    @jax.jit
    def constr_dic_to_unconstr_arr(dic: ParamNameValMapping) -> JAXArray:
        """Covert free parameters dict from constrained space to array in
        unconstrained space.
        """
        constr_arr = dic_to_arr(dic)
        return constr_arr_to_unconstr_arr(constr_arr)

    # get default value of each parameter
    default_constr_dic = {
        pid_to_pname[k]: v for k, v in model_info.default.items()
    }
    default_constr_dic = {k: default_constr_dic[k] for k in free_names}
    default_constr_arr = jnp.array([default_constr_dic[i] for i in free_names])
    default_unconstr_arr = constr_arr_to_unconstr_arr(default_constr_arr)
    free_default: dict[str, dict[ParamName, JAXFloat] | JAXArray] = {
        'constr_dic': default_constr_dic,
        'constr_arr': default_constr_arr,
        'unconstr_arr': default_unconstr_arr,
    }

    def get_sites(
        unconstr_arr: JAXArray,
    ) -> dict[
        Literal['params', 'models', 'loglike'], dict[str, JAXFloat | JAXArray]
    ]:
        """Get parameters in constrained space, models values and log
        likelihood, given free parameters array in unconstrained space.
        """
        sites = constrain_fn(
            model=numpyro_model,
            model_args=(),
            model_kwargs={},
            params=arr_to_dic(unconstr_arr),
            return_deterministic=True,
        )
        params = get_params(sites)
        models = get_models(sites)
        loglike = get_loglike(sites)
        return {'params': params, 'models': models, 'loglike': loglike}

    def get_params(sites: Mapping) -> dict:
        """Get parameters' values in constrained space given numpyro model
        sites.
        """
        params = {i: sites[i] for i in params_names}
        return params

    def get_models(sites: Mapping) -> dict:
        """Get model values given numpyro model sites."""
        models = {k: sites[k] for k in data_group.keys()}
        models |= {
            f'{i}_model': sites[f'{i}_model']
            for v in data_group.values()
            for i in v
        }
        return models

    def get_loglike(sites: Mapping) -> dict:
        """Get log likelihood given numpyro model sites."""
        loglike_data = {
            i: sites[f'{i}_loglike'] for v in data_group.values() for i in v
        }
        loglike_point = {i: sites[f'{i}_loglike'] for i in data_group.keys()}
        loglike_group = {k: v.sum(axis=-1) for k, v in loglike_point.items()}
        loglike_channels = jnp.concatenate(
            [loglike_point[i] for i in data_group.keys()], axis=-1
        )
        loglike_total = loglike_channels.sum(axis=-1)
        loglike = {
            'data': loglike_data,
            'point': loglike_point,
            'group': loglike_group,
            'channels': loglike_channels,
            'total': loglike_total,
        }
        return loglike

    @jax.jit
    def unconstr_dic_to_params_dic(
        dic: ParamNameValMapping,
    ) -> ParamNameValMapping:
        """Get parameters dict in constrained space,
        given a free parameters dict in unconstrained space.
        """
        return jax.jit(get_sites)(dic_to_arr(dic))['params']

    @jax.jit
    def unconstr_arr_to_params_array(arr: JAXArray) -> JAXArray:
        """Get parameters dict in constrained space,
        given a free parameters array in unconstrained space.
        """
        unconstr_dic = arr_to_dic(arr)
        params_dic = unconstr_dic_to_params_dic(unconstr_dic)
        return jnp.array([params_dic[i] for i in params_names])

    @jax.jit
    def unconstr_covar(unconstr_arr: JAXArray) -> JAXArray:
        """Calculate covariance matrix of free parameters in unconstrained
        space, given a free parameters array in unconstrained space.
        """
        hess = jax.jit(jax.hessian(deviance_total))(unconstr_arr)
        covar = jnp.linalg.inv(hess)
        return 2.0 * covar

    @jax.jit
    def params_covar(
        unconstr_arr: JAXArray,
        unconstr_cov: JAXArray,
    ) -> jnp.ndarray:
        """Calculate covariance matrix of all parameters in constrained space,
        given values and covariance matrix of free parameters in unconstrained
        space.
        """
        jac = jax.jit(jax.jacobian(unconstr_arr_to_params_array))(unconstr_arr)
        return jac @ unconstr_cov @ jac.T

    @jax.jit
    def get_mle(unconstr_arr: JAXArray) -> tuple[JAXArray, JAXArray]:
        """Get the value and covariance matrix of all parameters in constrained
        space, given MLE of free parameters in unconstrained space.
        """
        params_arr = unconstr_arr_to_params_array(unconstr_arr)
        params_cov = params_covar(unconstr_arr, unconstr_covar(unconstr_arr))
        return params_arr, params_cov

    # NOTE:
    #   the following functions will be used in simulation procedure,
    #   so we do not jit it here, or data substitution will fail
    def loglike(unconstr_arr: JAXArray) -> dict[str, JAXArray]:
        """Calculate log-likelihood given free parameters dict in constrained
        space.
        """
        return get_sites(unconstr_arr)['loglike']

    def deviance(unconstr_arr: JAXArray) -> dict:
        """Calculate total/group/point deviance given free parameters array in
        unconstrained space.
        """
        loglike_dic = loglike(unconstr_arr)
        neg_double = jax.jit(lambda x: -2.0 * x)
        point = jax.tree_map(neg_double, loglike_dic['point'])
        group = jax.tree_map(neg_double, loglike_dic['group'])
        total = jax.tree_map(neg_double, loglike_dic['total'])
        return {'total': total, 'group': group, 'point': point}

    def deviance_total(unconstr_arr: JAXArray) -> JAXFloat:
        """Calculate total deviance given free parameters array in
        unconstrained space.
        """
        return deviance(unconstr_arr)['total']

    def residual(unconstr_arr: JAXArray) -> JAXArray:
        """Calculate deviance residual (i.e. sqrt deviance) given free
        parameters array in unconstrained space.
        """
        loglike_dic = loglike(unconstr_arr)
        loglike_arr = jnp.hstack(list(loglike_dic['point'].values()))
        return jnp.sqrt(-2.0 * loglike_arr)

    # =================== functions used in optimization ======================

    # =============== functions used in simulation procedure ==================
    lm_solver = optx.LevenbergMarquardt(rtol=0.0, atol=1e-6)

    @jax.jit
    def fit_once(i: int, args: tuple) -> tuple:
        """Loop core, fit simulation data once."""
        sim_data, result, init = args

        # substitute observation data with simulation data
        new_data = {
            f'{j}_data': sim_data[j][i] for v in data_group.values() for j in v
        }
        new_residual = jax.jit(handlers.substitute(fn=residual, data=new_data))
        new_deviance = jax.jit(handlers.substitute(fn=deviance, data=new_data))
        new_sites = jax.jit(handlers.substitute(fn=get_sites, data=new_data))

        # fit simulation data
        res = optx.least_squares(
            fn=lambda p, _: new_residual(p),
            solver=lm_solver,
            y0=init[i],
            max_steps=1024,
            throw=False,
        )
        fitted_params = res.value
        grad_norm = jnp.linalg.norm(res.state.f_info.grad)

        sites = new_sites(fitted_params)

        # update best fit params to result
        params = sites['params']
        result['params'] = jax.tree_map(
            lambda x, y: x.at[i].set(y),
            result['params'],
            params,
        )

        # update the best fit model to result
        models = sites['models']
        result['models'] = jax.tree_map(
            lambda x, y: x.at[i].set(y),
            result['models'],
            {k: models[k] for k in result['models']},
        )

        # update the deviance information to result
        dev = new_deviance(fitted_params)
        res_dev = result['deviance']
        res_dev['group'] = jax.tree_map(
            lambda x, y: x.at[i].set(y),
            res_dev['group'],
            dev['group'],
        )
        res_dev['point'] = jax.tree_map(
            lambda x, y: x.at[i].set(y),
            res_dev['point'],
            dev['point'],
        )
        res_dev['total'] = res_dev['total'].at[i].set(dev['total'])

        valid = jnp.bitwise_not(
            jnp.isnan(dev['total'])
            | jnp.isnan(grad_norm)
            | jnp.greater(grad_norm, 1e-3)
        )
        result['valid'] = result['valid'].at[i].set(valid)

        return sim_data, result, init

    def sim_sequence_fit(
        sim_data: dict[str, JAXArray],
        result: dict,
        init: JAXArray,
        run_str: str,
        progress: bool,
        update_rate: int,
    ):
        """Fit simulation data in sequence."""
        n = len(result['valid'])

        if progress:
            pbar_factory = progress_bar_factory(
                n, 1, run_str=run_str, update_rate=update_rate
            )
            fn = pbar_factory(fit_once)
        else:
            fn = fit_once

        fit_jit = jax.jit(lambda *args: lax.fori_loop(0, n, fn, args)[1])
        result = fit_jit(sim_data, result, init)
        return result

    def sim_parallel_fit(
        sim_data: dict[str, JAXArray],
        result: dict,
        init: JAXArray,
        run_str: str,
        progress: bool,
        update_rate: int,
        n_parallel: int,
    ) -> dict:
        """Fit simulation data in parallel."""
        n = len(result['valid'])
        n_parallel = int(n_parallel)
        batch = n // n_parallel

        if progress:
            pbar_factory = progress_bar_factory(
                n, n_parallel, run_str=run_str, update_rate=update_rate
            )
            fn = pbar_factory(fit_once)
        else:
            fn = fit_once

        fit_pmap = jax.pmap(lambda *args: lax.fori_loop(0, batch, fn, args)[1])
        reshape = lambda x: x.reshape((n_parallel, -1) + x.shape[1:])
        result = fit_pmap(
            jax.tree_map(reshape, sim_data),
            jax.tree_map(reshape, result),
            jax.tree_map(reshape, init),
        )

        return jax.tree_map(jnp.concatenate, result)

    def simulate_and_fit(
        seed: int,
        free_params: dict[ParamName, JAXArray],
        model_values: dict[str, JAXArray],
        n: int = 1,
        parallel: bool = True,
        n_parallel: int | None = None,
        progress: bool = True,
        update_rate: int = 50,
        run_str: str = 'Fitting',
    ) -> dict:
        """Simulate data and then fit the simulation data.

        Parameters
        ----------
        seed : int
            The random number generator seed used for data simulation.
        free_params : dict
            The free parameters values in unconstrained space.
        model_values : dict
            The model values corresponding to `free_params`.
        n : int, optional
            The number of simulations of each model value, by default 1.
        parallel : bool, optional
            Whether to fit in parallel, by default True.
        n_parallel : int, optional
            The number of parallel processes when `parallel` is ``True``.
            Defaults to ``jax.local_device_count()``.
        progress : bool, optional
            Whether to show progress bar, by default True.
        update_rate : int, optional
            The update rate of the progress bar, by default 50.
        run_str : str, optional
            The string to ahead progress bar during the run when `progress` is
            True. The default is 'Fitting'.

        Returns
        -------
        result : dict
            The simulation and fitting result.
        """
        seed = int(seed)
        free_params = jax.tree_map(jnp.array, free_params)
        model_values = {
            f'{k}_model': model_values[f'{k}_model'] for k in simulators
        }
        n = int(n)
        n_parallel = get_parallel_number(n_parallel)

        assert set(free_params) == set(free_names)
        assert n > 0

        # check if all params shapes are the same
        shapes = list(jax.tree_map(jnp.shape, free_params).values())
        assert all(i == shapes[0] for i in shapes)

        # TODO: support posterior prediction with n > 1
        assert not (shapes[0] != () and n > 1)

        # get free parameters arrays in unconstrained space,
        # used as initial value in optimization
        to_unconstr = constr_dic_to_unconstr_arr
        if shapes[0] != ():
            for _ in range(len(shapes[0])):
                to_unconstr = jax.jit(jax.vmap(to_unconstr, in_axes=0))
        init = to_unconstr(free_params)

        # simulate data
        sim_data = simulate(seed, model_values, n)

        if n > 1:
            init = jnp.full((n, len(init)), init)
            nsim = n
        else:
            nsim = len(init)

        # fit result container
        result = {
            'params': {k: jnp.empty(nsim) for k in params_names},
            'models': {
                i: jnp.empty((nsim, ndata[k]))
                for k, v in data_group.items()
                for i in [k, *map('{}_model'.format, v)]
            },
            'deviance': {
                'total': jnp.empty(nsim),
                'group': {k: jnp.empty(nsim) for k in data_group},
                'point': {k: jnp.empty((nsim, ndata[k])) for k in data_group},
            },
            'valid': jnp.full(nsim, True, bool),
        }

        # fit simulation data
        if parallel:
            result = sim_parallel_fit(
                sim_data,
                result,
                init,
                run_str,
                progress,
                update_rate,
                n_parallel,
            )
        else:
            result = sim_sequence_fit(
                sim_data, result, init, run_str, progress, update_rate
            )
        result['data'] = sim_data
        return result

    # =============== functions used in simulation procedure ==================

    return Helper(
        ndata=ndata,
        nparam=nparam,
        dof=dof,
        data_names=list(data.keys()),
        statistic=stat,
        channels=channels,
        obs_data=obs_data,
        data=dict(data),
        model=dict(model),
        seed=rng_seed,
        sampling_dist=sampling_dist,
        numpyro_model=numpyro_model,
        params_names={
            'free': free_names,
            'deterministic': deterministic_names,
            'interest': interest_names,
            'all': params_names,
        },
        params_setup=model_info.setup,
        params_latex=pname_to_latex,
        params_unit=pname_to_unit,
        params_log=pname_to_log,
        params_comp_latex=pname_to_comp_latex,
        free_default=free_default,
        get_sites=get_sites,
        get_params=get_params,
        get_models=get_models,
        get_loglike=get_loglike,
        get_mle=get_mle,
        params_covar=params_covar,
        deviance_total=deviance_total,
        deviance=deviance,
        residual=residual,
        constr_arr_to_unconstr_arr=constr_arr_to_unconstr_arr,
        constr_dic_to_unconstr_arr=constr_dic_to_unconstr_arr,
        unconstr_dic_to_params_dic=unconstr_dic_to_params_dic,
        simulate=simulate,
        simulate_and_fit=simulate_and_fit,
    )


class Helper(NamedTuple):
    """Helper for fitting and analysis."""

    ndata: dict[str, int]
    """The number of channels in each dataset and the total number of channels.
    """

    nparam: int
    """The number of free parameters in the model."""

    dof: int
    """The degree of freedom."""

    data_names: list[str]
    """Name of each data."""

    statistic: dict[str, Statistic]
    """The statistic used in each dataset."""

    channels: dict[str, np.ndarray]
    """Channel information of the datasets."""

    obs_data: dict[str, JAXArray]
    """The datasets of observations, including net counts, counts in the "on"
    and "off" measurements.
    """

    data: dict[str, FixedData]
    """FixedData instances."""

    model: dict[str, CompiledModel]
    """Compiled spectral models."""

    seed: dict[str, int]
    """Random number generator seed."""

    sampling_dist: dict[str, tuple[Literal['norm', 'poisson'], tuple]]
    """Sampling distribution of observation data, this is used for probability
    integral transform calculation.
    """

    numpyro_model: Callable[[bool], None]
    """The numpyro model for spectral fitting."""

    params_names: dict
    """The names of parameters in the model."""

    free_default: dict[str, dict[ParamName, JAXFloat] | JAXArray]
    """The default values of free parameters."""

    params_setup: dict[ParamName, tuple[ParamName, ParamSetup]]
    """The mapping from forwarded parameters names to parameters names."""

    params_latex: dict[ParamName, str]
    """The LaTeX representation of parameters."""

    params_unit: dict[ParamName, str]
    """The unit of parameters."""

    params_log: dict[ParamName, bool]
    """Whether the parameters are in log space."""

    params_comp_latex: dict[ParamName, str]
    """The LaTeX representation of parameter's component."""

    get_sites: Callable[
        [JAXArray],
        dict[
            Literal['params', 'models', 'loglike'],
            dict[str, JAXFloat | JAXArray],
        ],
    ]
    """Get parameters in constrained space, models values and log likelihood,
    given free parameters array in unconstrained space.
    """

    get_params: Callable[[Mapping], dict]
    """Get parameters' values in constrained space given numpyro model sites.
    """

    get_models: Callable[[Mapping], dict]
    """Get model values given numpyro model sites."""

    get_loglike: Callable[[Mapping], dict]
    """Get log likelihood given numpyro model sites."""

    get_mle: Callable[[JAXArray], tuple[JAXArray, JAXArray]]
    """Get the MLE and error of all parameters in constrained space,
    given MLE of free parameters in unconstrained space.
    """

    params_covar: Callable[[JAXArray, JAXArray], JAXArray]
    """Calculate covariance matrix of all parameters in constrained space,
    given values and covariance matrix of free parameters in unconstrained
    space.
    """

    deviance_total: Callable[[JAXArray], JAXFloat]
    """Calculate total deviance given free parameters array in unconstrained
    space.
    """

    deviance: Callable[[JAXArray], dict[str, JAXArray]]
    """Calculate total, group and point deviance given free parameters array in
    unconstrained space.
    """

    residual: Callable[[JAXArray], JAXArray]
    """Calculate deviance residual (i.e., sqrt deviance) given free parameters
    array in unconstrained space.
    """

    constr_arr_to_unconstr_arr: Callable[[JAXArray], JAXArray]
    """Covert free parameters array from constrained space into
    unconstrained space.
    """

    constr_dic_to_unconstr_arr: Callable[[ParamNameValMapping], JAXArray]
    """Covert free parameters dict from constrained space to array in
    unconstrained space.
    """

    unconstr_dic_to_params_dic: Callable[
        [ParamNameValMapping], ParamNameValMapping
    ]
    """Get parameters dict in constrained space, given a free parameters dict
    in unconstrained space.
    """

    simulate: Callable[[int, dict[str, JAXArray], int], dict[str, JAXArray]]
    """Function to simulate data."""

    simulate_and_fit: Callable[
        [int, dict, dict, int, bool, int, bool, int, str], dict
    ]
    """Function to simulate data and then fit the simulation data."""
