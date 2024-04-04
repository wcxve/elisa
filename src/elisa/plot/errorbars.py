"""Error bars calculation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jax.scipy.special import expit, logit, xlogy
from scipy.stats import chi2, norm

from elisa.infer.likelihood import pgstat_background, wstat_background

if TYPE_CHECKING:
    from elisa.util.typing import JAXArray, NumPyArray as NDArray


@jax.jit
def wstat(s, n_on, n_off, a) -> JAXArray:
    b = wstat_background(s, n_on, n_off, a)
    m_on = s + a * b
    m_off = b
    return 2.0 * (m_on - xlogy(n_on, m_on) + m_off - xlogy(n_off, m_off))


@jax.jit
def pgstat(s, n, b_est, b_err, a) -> JAXArray:
    b = pgstat_background(s, n, b_est, b_err, a)
    m_on = s + a * b
    m_off = b
    chi = (m_off - b_est) / b_err
    return 2.0 * (m_on - xlogy(n, m_on)) + chi * chi


def get_sigma(cl: float) -> float:
    return norm.isf(0.5 - 0.5 * cl)


def get_delta(cl: float) -> float:
    return chi2.ppf(cl, 1.0)


@jax.jit
def _sigmoid(x, s):
    finfo = jnp.finfo(jnp.result_type(x))
    return jnp.clip(expit(x), a_min=finfo.tiny, a_max=1.0 - finfo.eps) * s


@jax.jit
def _inv_sigmoid(y, s):
    return logit(y / s)


def pgstat_errors(
    n: NDArray,
    b_est: NDArray,
    b_err: NDArray,
    a: NDArray,
    uplim: float = 0.95,
) -> NDArray: ...


def wstat_errors(
    n_on: NDArray,
    n_off: NDArray,
    a: float | NDArray,
    uplim: float = 0.95,
) -> tuple[NDArray, NDArray, NDArray]:
    a = np.full(n_on.shape, a)
    lower = np.empty(n_on.shape)

    s = np.clip(n_on - a * n_off, 0.0, None)
    error = np.sqrt(n_on + a * a * n_off)
    non_pos_mask = np.equal(s, 0.0)
    wstat0 = wstat(s, n_on, n_off, a)
    wstat_at_zero = wstat(jnp.zeros(n_on.shape), n_on, n_off, a)
    zero_lower_mask = np.less_equal(wstat_at_zero - wstat0, 1.0)
    lower[zero_lower_mask] = 0.0

    @jax.jit
    def stat_diff(s_trans, args):
        s0, stat0, delta, n_on, n_off, a = args
        return wstat(s0 + jnp.exp(s_trans), n_on, n_off, a) - (stat0 + delta)

    delta_up = get_delta(uplim)
    sigma_up = get_sigma(uplim)

    lm_solver = optx.LevenbergMarquardt(rtol=0.0, atol=1e-6)

    up_guess = np.log(error)
    up_guess[non_pos_mask] = np.log(sigma_up * error[non_pos_mask])
    delta = np.ones(n_on.shape)
    delta[non_pos_mask] = delta_up
    res = optx.root_find(
        stat_diff,
        lm_solver,
        up_guess,
        args=(s, wstat0, delta, n_on, n_off, a),
        max_steps=1024,
    )
    upper = s + np.exp(res.value)

    @jax.jit
    def stat_diff(s_trans, args):
        s0, stat0, delta, n_on, n_off, a = args
        return wstat(_sigmoid(s_trans, s0), n_on, n_off, a) - (stat0 + delta)

    lower_mask = ~zero_lower_mask
    lo_guess = np.clip(s[lower_mask] - error[lower_mask], 1e-3, None)
    lo_guess = _inv_sigmoid(lo_guess, s[lower_mask])
    delta = np.ones(n_on.shape)
    res = optx.root_find(
        stat_diff,
        lm_solver,
        lo_guess,
        args=(
            s[lower_mask],
            wstat0[lower_mask],
            delta[lower_mask],
            n_on[lower_mask],
            n_off[lower_mask],
            a[lower_mask],
        ),
    )
    lower[lower_mask] = _sigmoid(res.value, s[lower_mask])

    return lower, upper, non_pos_mask
