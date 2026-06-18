"""Unit tests for _build_mmm_link_assumptions in cdnots_discovery.py."""
import sys, os
import time
import numpy as np
import pytest

CAUSALMMM_ROOT = os.path.join(os.path.dirname(__file__), "..")
PYMC_ROOT = os.environ.get(
    "PYMC_MERIDIAN_ROOT",
    "/home/ennes/mestrado/pymc_meridian_comparison",
)
ARQUIVOS       = os.path.join(CAUSALMMM_ROOT, "cdnots")

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


@pytest.fixture(scope="module")
def disc():
    return _load("cdnots_discovery", os.path.join(ARQUIVOS, "cdnots_discovery.py"))


# ── Task 1 tests: _build_mmm_link_assumptions ────────────────────────────────

def test_y_receives_from_all_channels_and_controls(disc):
    """y_idx must accept edges from every channel and control, at all lags."""
    n_ch, n_ctrl, n_vars, max_lag = 4, 1, 6, 2
    la = disc._build_mmm_link_assumptions(n_ch, n_ctrl, n_vars, max_lag)
    y_idx = n_vars - 1
    for i in range(n_vars - 1):
        for tau in range(1, max_lag + 1):
            assert (i, -tau) in la[y_idx], f"missing ({i}, -{tau}) in y targets"
    assert len(la[y_idx]) == (n_vars - 1) * max_lag, \
        f"expected {(n_vars-1)*max_lag} edges to y, got {len(la[y_idx])}"


def test_channel_receives_from_channels_and_controls(disc):
    """Each channel must accept edges from all other channels and all controls."""
    n_ch, n_ctrl, n_vars, max_lag = 4, 1, 6, 2
    la = disc._build_mmm_link_assumptions(n_ch, n_ctrl, n_vars, max_lag)
    ch_indices   = list(range(n_ch))
    ctrl_indices = list(range(n_ch, n_ch + n_ctrl))
    for j in ch_indices:
        for i in ch_indices + ctrl_indices:
            for tau in range(1, max_lag + 1):
                assert (i, -tau) in la[j], f"channel {j}: missing ({i}, -{tau})"


def test_controls_have_no_incoming_edges(disc):
    """Controls are exogenous — their link_assumptions dict must be empty.
    An empty dict already implies channels (and all other variables) cannot cause controls."""
    n_ch, n_ctrl, n_vars, max_lag = 4, 2, 7, 2
    la = disc._build_mmm_link_assumptions(n_ch, n_ctrl, n_vars, max_lag)
    ctrl_indices = list(range(n_ch, n_ch + n_ctrl))
    for j in ctrl_indices:
        assert la[j] == {}, f"control {j} should have no incoming edges, got {la[j]}"


def test_y_is_never_a_source(disc):
    """y must not appear as a source (i) in any target's link dict."""
    n_ch, n_ctrl, n_vars, max_lag = 4, 1, 6, 2
    la = disc._build_mmm_link_assumptions(n_ch, n_ctrl, n_vars, max_lag)
    y_idx = n_vars - 1
    for j, incoming in la.items():
        for (i, _tau), _ in incoming.items():
            assert i != y_idx, f"y ({y_idx}) appears as source for target {j}"


def test_link_type_is_question_arrow(disc):
    """Every allowed edge must use the '?->' (uncertain-directed) link type."""
    n_ch, n_ctrl, n_vars, max_lag = 3, 1, 5, 1
    la = disc._build_mmm_link_assumptions(n_ch, n_ctrl, n_vars, max_lag)
    for j, incoming in la.items():
        for key, ltype in incoming.items():
            assert ltype == "?->", f"unexpected link type {ltype!r} at target {j} key {key}"


# ── helpers ───────────────────────────────────────────────────────────────────

def _small_data(n_vars=6, T=80, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, n_vars)).astype(np.float64)


# ── Integration tests ─────────────────────────────────────────────────────────

def test_pcmci_discovery_returns_correct_shapes(disc):
    """_pcmci_discovery must return (adj, pval, qval) of shape (n_vars, n_vars)."""
    n_vars, n_ch, n_ctrl = 6, 4, 1
    data = _small_data(n_vars=n_vars, T=80)
    adj, pval, qval = disc._pcmci_discovery(
        data, n_vars, alpha=0.1, max_lag=1,
        ci_test_name="parcorr",
        n_channels=n_ch, n_controls=n_ctrl,
    )
    assert adj.shape  == (n_vars, n_vars)
    assert pval.shape == (n_vars, n_vars)
    assert qval.shape == (n_vars, n_vars)


def test_pcmci_discovery_y_has_no_outgoing_edges(disc):
    """After discovery, y (last var) must never appear as a source in adj."""
    n_vars, n_ch, n_ctrl = 6, 4, 1
    data = _small_data(n_vars=n_vars, T=80)
    adj, _, _ = disc._pcmci_discovery(
        data, n_vars, alpha=0.1, max_lag=1,
        ci_test_name="parcorr",
        n_channels=n_ch, n_controls=n_ctrl,
    )
    y_idx = n_vars - 1
    assert adj[y_idx, :].sum() == 0, "y must not cause any variable"


def test_pcmci_discovery_controls_have_no_incoming(disc):
    """Controls (indices n_ch .. n_ch+n_ctrl-1) must have no incoming edges."""
    n_vars, n_ch, n_ctrl = 6, 4, 1
    data = _small_data(n_vars=n_vars, T=80)
    adj, _, _ = disc._pcmci_discovery(
        data, n_vars, alpha=0.1, max_lag=1,
        ci_test_name="parcorr",
        n_channels=n_ch, n_controls=n_ctrl,
    )
    ctrl_idxs = list(range(n_ch, n_ch + n_ctrl))
    for c in ctrl_idxs:
        assert adj[:, c].sum() == 0, f"control {c} must have no incoming edges"


def test_discover_graph_completes_on_small_business_parcorr(disc):
    """End-to-end: discover_graph on small_business with parcorr returns a CausalGraph."""
    from mmm_param_recovery.data_generator.core import generate_mmm_dataset
    from mmm_param_recovery.data_generator.presets import get_preset_config
    from mmm_param_recovery.benchmarking.data_loader import prepare_dataset_for_modeling

    config = get_preset_config("small_business", seed=42)
    result = generate_mmm_dataset(config)
    data_df, ch_cols, ctrl_cols, _ = prepare_dataset_for_modeling(result)

    graph = disc.discover_graph(
        data_df, ch_cols,
        control_columns=ctrl_cols,
        alpha=0.05, max_lag=1,
        ci_test="parcorr",
    )

    assert len(graph.variable_names) == len(ch_cols) + len(ctrl_cols) + 1
    total = (len(graph.direct_channels) + len(graph.mediated_channels)
             + len(graph.excluded_channels))
    assert total == len(ch_cols), "every channel must be classified"


@pytest.mark.slow
def test_cmiknn_runtime_improvement(disc):
    """CMIknn with new params should complete in under 60s on 10-var, 100-row data."""
    pytest.importorskip("numba", reason="CMIknn requires numba")
    n_vars, n_ch, n_ctrl = 10, 8, 1
    data = _small_data(n_vars=n_vars, T=100, seed=7)
    t0 = time.perf_counter()
    adj, pval, qval = disc._pcmci_discovery(
        data, n_vars, alpha=0.1, max_lag=1,
        ci_test_name="kci",
        n_channels=n_ch, n_controls=n_ctrl,
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 60, f"CMIknn took {elapsed:.1f}s — expected < 60s with new params"
    assert adj.shape == (n_vars, n_vars)


def test_zero_controls_y_receives_only_from_channels(disc):
    """With n_controls=0, y should only receive edges from channels."""
    n_ch, n_ctrl, n_vars, max_lag = 4, 0, 5, 1
    la = disc._build_mmm_link_assumptions(n_ch, n_ctrl, n_vars, max_lag)
    y_idx = n_vars - 1
    # y receives from channels only (no controls)
    assert len(la[y_idx]) == n_ch * max_lag, \
        f"expected {n_ch * max_lag} edges to y, got {len(la[y_idx])}"
    # no spurious edges from y itself
    for (i, _tau) in la[y_idx]:
        assert i < n_ch, f"non-channel source {i} in y's incoming edges"
