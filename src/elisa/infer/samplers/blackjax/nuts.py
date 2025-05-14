from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import blackjax.adaptation.window_adaptation as window_adaptation
import blackjax.mcmc.integrators as integrators
import jax
import jax.numpy as jnp
import jax.random as random
from blackjax.mcmc.hmc import HMCState
from blackjax.mcmc.nuts import build_kernel
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import init_to_uniform, initialize_model
from numpyro.util import identity, is_prng_key

if TYPE_CHECKING:
    from collections.abc import Callable

    from blackjax.mcmc.metrics import MetricTypes


class BlackJAXNUTSState(NamedTuple):
    """State of the BlackJAX NUTS."""

    i: int
    """The iteration number."""

    z: dict
    """Python collection representing values (unconstrained samples from
    the posterior) at latent sites."""

    z_grad: dict
    """Gradient of potential energy w.r.t. latent sample sites."""

    potential_energy: float
    """Potential energy computed at the given value of `z`."""

    energy: float
    """Sum of potential energy and kinetic energy of the current state."""

    r: dict
    """The current momentum variable. If this is None, a new momentum variable
    will be drawn at the beginning of each sampling step."""

    num_steps: int
    """Number of steps in the Hamiltonian trajectory (for diagnostics)."""

    tree_depth: int
    """Tree depth of the current trajectory."""

    accept_prob: float
    """Acceptance probability of the proposal. Note that `z` does not
    correspond to the proposal if it is rejected."""

    mean_accept_prob: float
    """Mean acceptance probability until current iteration during warmup
    adaptation or sampling (for diagnostics)."""

    diverging: bool
    """Whether the current trajectory is diverging."""

    step_size: float
    """Step size to be used by the integrator in the next iteration."""

    inverse_mass_matrix: MetricTypes
    """The inverse mass matrix to be used for the next iteration."""

    adapt_state: window_adaptation.WindowAdaptationState
    """The current window adaption state of the NUTS."""

    rng_key: jax.Array
    """Random number generator seed used for the iteration."""


class BlackJAXNUTS(MCMCKernel):
    """NUTS implementation of BlackJAX, with automatic window adaptation."""

    def __init__(
        self,
        model: Callable | None = None,
        potential_fn: Callable | None = None,
        init_strategy: Callable = init_to_uniform,
        dense_mass: bool = True,
        initial_step_size: float = 1.0,
        target_accept_prob: float = 0.8,
        max_tree_depth: int = 10,
        divergence_threshold: float = 1000.0,
        integrator: Callable = integrators.velocity_verlet,
    ):
        if not (model is None) ^ (potential_fn is None):
            raise ValueError(
                'Only one of `model` or `potential_fn` must be specified.'
            )

        self._model = model
        self._potential_fn = potential_fn
        self._init_strategy = init_strategy

        # Window adaption parameters
        self._dense_mass = dense_mass
        self._initial_step_size = initial_step_size
        self._target_accept_prob = target_accept_prob

        # NUTS kernel parameters
        self._max_tree_depth = max_tree_depth
        self._divergence_threshold = divergence_threshold
        self._integrator = integrator

        # Sampling related
        self._postprocess_gen = None
        self._mcmc_kernel = None

    def postprocess_fn(self, model_args, model_kwargs):
        if self._postprocess_gen is None:
            return identity
        return self._postprocess_gen(*model_args, **model_kwargs)

    def _init_state(self, rng_key, model_args, model_kwargs, init_params):
        if self._model is not None:
            model_info = initialize_model(
                rng_key,
                self._model,
                init_strategy=self._init_strategy,
                dynamic_args=True,
                model_args=model_args,
                model_kwargs=model_kwargs,
                validate_grad=True,
            )
            init_params = model_info.param_info.z
            potential_gen = model_info.potential_fn
            postprocess_gen = model_info.postprocess_fn
            model_kwargs = {} if model_kwargs is None else model_kwargs
            potential_fn = potential_gen(*model_args, **model_kwargs)
            self._potential_fn = potential_fn
            self._postprocess_gen = postprocess_gen
        return init_params

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        # TODO: support chain_method='vectorized'
        if not is_prng_key(rng_key):
            raise NotImplementedError(
                "BlackJAX's NUTS only supports chain_method='parallel' or "
                "chain_method='sequential'. Please put in a feature request "
                'if it would be useful to be used in vectorized mode.'
            )

        rng_key, rng_key_init_model, rng_key_wa = random.split(rng_key, 3)

        # Initial parameters
        init_params = self._init_state(
            rng_key_init_model, model_args, model_kwargs, init_params
        )
        if self._potential_fn and init_params is None:
            raise ValueError(
                '`init_params` must be provided with `potential_fn`.'
            )

        # log posterior density function for blackjax
        log_density = lambda z: -self._potential_fn(z)

        # Build NUTS kernel
        kernel = build_kernel(self._integrator, self._divergence_threshold)

        # Initialize window adaption, adapted from blackjax.window_adaptation
        adapt_init, adapt_step, adapt_final = window_adaptation.base(
            is_mass_matrix_diagonal=not self._dense_mass,
            target_acceptance_rate=self._target_accept_prob,
        )
        schedule = window_adaptation.build_schedule(num_warmup)

        def wa_update(state, new_hmc_state, info):
            return adapt_step(
                state.adapt_state,
                schedule[state.i],
                new_hmc_state.position,
                info.acceptance_rate,
            )

        def mcmc_kernel(state: BlackJAXNUTSState) -> BlackJAXNUTSState:
            i = state.i + 1
            (rng_key,) = random.split(state.rng_key, 1)
            hmc_state = HMCState(
                position=state.z,
                logdensity=-state.potential_energy,
                logdensity_grad=jax.tree.map(jnp.negative, state.z_grad),
            )

            step_size = jnp.where(
                i <= num_warmup, state.adapt_state.step_size, state.step_size
            )
            inverse_mass_matrix = jnp.where(
                i <= num_warmup,
                state.adapt_state.inverse_mass_matrix,
                state.inverse_mass_matrix,
            )
            new_state, info = kernel(
                rng_key,
                hmc_state,
                log_density,
                step_size,
                inverse_mass_matrix,
                max_num_doublings=self._max_tree_depth,
            )

            adapt_state = jax.lax.cond(
                i <= num_warmup,
                (state, new_state, info),
                lambda args: wa_update(*args),
                state.adapt_state,
                identity,
            )
            step_size, inverse_mass_matrix = adapt_final(adapt_state)
            n = jnp.where(i <= num_warmup, i, i - num_warmup)
            new_mean_acc_prob = (
                state.mean_accept_prob
                + (info.acceptance_rate - state.mean_accept_prob) / n
            )
            return BlackJAXNUTSState(
                i=i,
                z=new_state.position,
                z_grad=jax.tree.map(jnp.negative, new_state.logdensity_grad),
                potential_energy=-new_state.logdensity,
                energy=info.energy,
                r=info.momentum,
                num_steps=info.num_integration_steps,
                tree_depth=info.num_trajectory_expansions,
                accept_prob=info.acceptance_rate,
                mean_accept_prob=new_mean_acc_prob,
                diverging=info.is_divergent,
                step_size=step_size,
                inverse_mass_matrix=inverse_mass_matrix,
                adapt_state=adapt_state,
                rng_key=rng_key,
            )

        self._mcmc_kernel = mcmc_kernel

        init_adapt_state = adapt_init(init_params, self._initial_step_size)
        pe, z_grad = jax.value_and_grad(self._potential_fn)(init_params)

        return BlackJAXNUTSState(
            i=0,
            z=init_params,
            z_grad=z_grad,
            potential_energy=pe,
            energy=jnp.nan,
            r=dict.fromkeys(init_params, jnp.nan),
            num_steps=-1,
            tree_depth=-1,
            accept_prob=jnp.nan,
            mean_accept_prob=0.0,
            diverging=False,
            step_size=init_adapt_state.step_size,
            inverse_mass_matrix=init_adapt_state.inverse_mass_matrix,
            adapt_state=init_adapt_state,
            rng_key=rng_key,
        )

    def sample(self, state, model_args, model_kwargs):
        return self._mcmc_kernel(state)

    @property
    def sample_field(self):
        return 'z'

    @property
    def default_fields(self):
        return 'z', 'diverging'

    def get_diagnostics_str(self, state):
        return (
            f'{state.num_steps} steps of size {state.step_size:.2e}. '
            f'acc. prob={state.mean_accept_prob:.2f}'
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_mcmc_kernel'] = None
        return state
