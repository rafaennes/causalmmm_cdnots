"""Tests for CMIknn speedup changes in cdnots_discovery.py."""
import sys, os
import numpy as np
import pytest

CAUSALMMM_ROOT = os.path.join(os.path.dirname(__file__), "..")
PYMC_ROOT      = "/home/ennes/mestrado/pymc_meridian_comparison"
ARQUIVOS       = os.path.join(CAUSALMMM_ROOT, "arquivos_recentes")

for p in [PYMC_ROOT, CAUSALMMM_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import importlib.util, types

def _load(name, filepath):
    full = f"mmm_param_recovery.benchmarking.{name}"
    spec = importlib.util.spec_from_file_location(full, filepath)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod

disc = _load("cdnots_discovery", os.path.join(ARQUIVOS, "cdnots_discovery.py"))


# ── helpers ──────────────────────────────────────────────────────────────────

def _small_data(n_vars=6, T=80, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, n_vars)).astype(np.float64)


# ── Task 1 tests: _build_mmm_link_assumptions ────────────────────────────────

def test_y_receives_from_all_channels_and_controls():
    """y_idx must accept edges from every channel and control, at all lags."""
    n_ch, n_ctrl, n_vars, max_lag = 4, 1, 6, 2
    la = disc._build_mmm_link_assumptions(n_ch, n_ctrl, n_vars, max_lag)
    y_idx = n_vars - 1
    for i in range(n_vars - 1):
        for tau in range(1, max_lag + 1):
            assert (i, -tau) in la[y_idx], f"missing ({i}, -{tau}) in y targets"


def test_channel_receives_from_channels_and_controls():
    """Each channel must accept edges from all other channels and all controls."""
    n_ch, n_ctrl, n_vars, max_lag = 4, 1, 6, 2
    la = disc._build_mmm_link_assumptions(n_ch, n_ctrl, n_vars, max_lag)
    ch_indices   = list(range(n_ch))
    ctrl_indices = list(range(n_ch, n_ch + n_ctrl))
    for j in ch_indices:
        for i in ch_indices + ctrl_indices:
            for tau in range(1, max_lag + 1):
                assert (i, -tau) in la[j], f"channel {j}: missing ({i}, -{tau})"


def test_controls_have_no_incoming_edges():
    """Controls are exogenous — their link_assumptions dict must be empty."""
    n_ch, n_ctrl, n_vars, max_lag = 4, 2, 7, 2
    la = disc._build_mmm_link_assumptions(n_ch, n_ctrl, n_vars, max_lag)
    ctrl_indices = list(range(n_ch, n_ch + n_ctrl))
    for j in ctrl_indices:
        assert la[j] == {}, f"control {j} should have no incoming edges, got {la[j]}"


def test_y_is_never_a_source():
    """y must not appear as a source (i) in any target's link dict."""
    n_ch, n_ctrl, n_vars, max_lag = 4, 1, 6, 2
    la = disc._build_mmm_link_assumptions(n_ch, n_ctrl, n_vars, max_lag)
    y_idx = n_vars - 1
    for j, incoming in la.items():
        for (i, _tau), _ in incoming.items():
            assert i != y_idx, f"y ({y_idx}) appears as source for target {j}"


def test_channels_do_not_cause_controls():
    """Channels must not appear in the incoming-edge dicts of controls."""
    n_ch, n_ctrl, n_vars, max_lag = 4, 2, 7, 2
    la = disc._build_mmm_link_assumptions(n_ch, n_ctrl, n_vars, max_lag)
    ctrl_indices = list(range(n_ch, n_ch + n_ctrl))
    ch_indices   = list(range(n_ch))
    for j in ctrl_indices:
        for (i, _tau) in la[j]:
            assert i not in ch_indices, f"channel {i} → control {j} must be excluded"


def test_link_type_is_question_arrow():
    """Every allowed edge must use the '?->' (uncertain-directed) link type."""
    n_ch, n_ctrl, n_vars, max_lag = 3, 1, 5, 1
    la = disc._build_mmm_link_assumptions(n_ch, n_ctrl, n_vars, max_lag)
    for j, incoming in la.items():
        for key, ltype in incoming.items():
            assert ltype == "?->", f"unexpected link type {ltype!r} at target {j} key {key}"
