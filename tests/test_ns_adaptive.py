"""Regression tests for nested-sampling posterior exports and efficiency."""

from __future__ import annotations

import math

import numpy as np
import pytest

from elisa import BayesFit
from elisa.infer.samplers.ns import nautilus as _nautilus
from elisa.infer.samplers.ns.nautilus import (
    DEFAULT_ESS_MULTIPLIER,
    adaptive_equal_weight_boost,
    check_equal_weight_boost,
    check_ess_multiplier,
    resolve_ess_multiplier,
    weighted_bases,
)
from elisa.models import PowerLaw

SLOW = pytest.mark.slow


def test_weighted_bases_matches_nautilus_rule():
    rng = np.random.default_rng(0)
    log_w = rng.normal(size=500)
    expected = np.exp(log_w - np.max(log_w)).sum()
    assert weighted_bases(log_w) == pytest.approx(expected)


def test_check_ess_multiplier_accepts_ge_one():
    assert check_ess_multiplier(1.0) == 1.0
    assert check_ess_multiplier(2.0) == 2.0
    assert check_ess_multiplier(1) == 1.0


@pytest.mark.parametrize(
    'bad', [0.0, 0.5, -1.0, math.nan, math.inf, -math.inf]
)
def test_check_ess_multiplier_rejects_invalid(bad):
    with pytest.raises(ValueError):
        check_ess_multiplier(bad)


def test_check_equal_weight_boost_accepts_positive():
    assert check_equal_weight_boost(0.1) == 0.1
    assert check_equal_weight_boost(1.0) == 1.0


@pytest.mark.parametrize('bad', [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_check_equal_weight_boost_rejects_invalid(bad):
    with pytest.raises(ValueError):
        check_equal_weight_boost(bad)


def test_default_ess_multiplier(monkeypatch):
    assert DEFAULT_ESS_MULTIPLIER == 2.0
    assert resolve_ess_multiplier(None) == 2.0

    monkeypatch.setattr(_nautilus, 'DEFAULT_ESS_MULTIPLIER', 5.0)
    assert resolve_ess_multiplier(None) == 5.0

    monkeypatch.setattr(_nautilus, 'DEFAULT_ESS_MULTIPLIER', 0.5)
    with pytest.raises(ValueError):
        resolve_ess_multiplier(None)


def test_adaptive_boost_matches_target_draw_count():
    n_base = 101.0
    weighted_ess = 5100.5
    multiplier = 2.0
    boost = adaptive_equal_weight_boost(
        weighted_ess,
        n_base,
        multiplier,
    )
    assert boost * n_base == pytest.approx(multiplier * weighted_ess)


def test_adaptive_boost_grows_with_concentration():
    weighted_ess = 2000.0
    multiplier = 2.0
    concentrated = adaptive_equal_weight_boost(
        weighted_ess,
        1.0,
        multiplier,
    )
    broad = adaptive_equal_weight_boost(
        weighted_ess,
        500.0,
        multiplier,
    )
    assert concentrated > broad >= 1.0


def test_adaptive_boost_validates_multiplier():
    with pytest.raises(ValueError):
        adaptive_equal_weight_boost(100.0, 50.0, 0.5)


def _fit(simulation, method='nautilus', **kwargs):
    model = PowerLaw()
    model.PowerLaw.K.log = True
    return getattr(BayesFit(simulation, model, seed=100), method)(**kwargs)


@SLOW
def test_nautilus_default_boost_reff_roundtrip(simulation):
    result = _fit(
        simulation,
        ess=300,
        constructor_kwargs={'pool': 1},
        termination_kwargs={'verbose': False},
    )
    parameter = 'PowerLaw.K'
    n_draws = int(result.idata['posterior'][parameter].shape[1])
    ess = result.ess[parameter]
    assert n_draws >= ess
    assert 0.0 < result.reff <= 1.0
    assert result.reff == pytest.approx(min(1.0, ess / n_draws))


@SLOW
def test_nautilus_explicit_legacy_boost_clamps_reff(simulation):
    result = _fit(
        simulation,
        ess=300,
        equal_weight_boost=1.0,
        constructor_kwargs={'pool': 1},
        termination_kwargs={'verbose': False},
    )
    parameter = 'PowerLaw.K'
    n_draws = int(result.idata['posterior'][parameter].shape[1])
    ess = result.ess[parameter]
    assert 0.0 < result.reff <= 1.0
    assert result.reff == pytest.approx(min(1.0, ess / n_draws))


@SLOW
def test_nautilus_evidence_invariant_to_resampling(simulation):
    result_a = _fit(
        simulation,
        ess=300,
        ess_multiplier=1.0,
        termination_kwargs={'verbose': False},
        constructor_kwargs={'pool': 1},
    )
    result_b = _fit(
        simulation,
        ess=300,
        ess_multiplier=5.0,
        termination_kwargs={'verbose': False},
        constructor_kwargs={'pool': 1},
    )
    assert np.isclose(result_a.lnZ[0], result_b.lnZ[0], rtol=1e-6)


@SLOW
def test_ultranest_reff_roundtrip(simulation):
    result = _fit(
        simulation,
        method='ultranest',
        ess=250,
        print_result=False,
    )
    parameter = 'PowerLaw.K'
    n_draws = int(result.idata['posterior'][parameter].shape[1])
    ess = result.ess[parameter]
    assert 0.0 < result.reff <= 1.0
    assert result.reff == pytest.approx(min(1.0, ess / n_draws))
