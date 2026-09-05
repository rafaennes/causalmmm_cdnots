"""Smoke tests for all causal discovery algorithms.

Each test verifies output format, not discovery quality.
All tests use the small_business preset (4 channels, 104 weeks) for speed.
"""
import numpy as np
import pytest

from causal_discovery.algorithms import (
    granger, pcmci_cmiknn, lpcmci, dynotears, cdnots,
    regime_pcmci, cedar, rpcmci,
)
from causal_discovery.compare import _load_preset


@pytest.fixture(scope="module")
def small_data():
    data_df, var_names, true_adj, truth_var_names = _load_preset("small_business")
    return data_df, var_names


def _check_graph(g, var_names, expected_algorithm):
    assert g.adjacency_matrix.shape == (len(var_names), len(var_names))
    assert g.variable_names == tuple(var_names)
    assert g.algorithm == expected_algorithm
    assert g.runtime_seconds > 0
    assert not np.any(np.isnan(g.adjacency_matrix))
    assert set(np.unique(g.adjacency_matrix)).issubset({0.0, 1.0})
    # y never causes anything
    y_idx = var_names.index("y")
    assert g.adjacency_matrix[y_idx, :].sum() == 0, f"{g.algorithm}: y caused something"


def test_granger(small_data):
    data_df, var_names = small_data
    g = granger.discover(data_df, var_names, alpha=0.05, max_lag=2)
    _check_graph(g, var_names, "granger")


def test_pcmci_cmiknn(small_data):
    data_df, var_names = small_data
    g = pcmci_cmiknn.discover(data_df, var_names, alpha=0.05, max_lag=2)
    _check_graph(g, var_names, "pcmci_cmiknn")
    assert "pvalues" in g.metadata


def test_lpcmci(small_data):
    data_df, var_names = small_data
    g = lpcmci.discover(data_df, var_names, alpha=0.05, max_lag=2)
    _check_graph(g, var_names, "lpcmci")
    assert "graph_matrix" in g.metadata


def test_dynotears(small_data):
    data_df, var_names = small_data
    g = dynotears.discover(data_df, var_names, max_lag=2, w_threshold=0.1)
    _check_graph(g, var_names, "dynotears")
    assert "w_threshold" in g.metadata
    assert "n_edges_raw" in g.metadata


def test_cdnots(small_data):
    """CD-NOTS surrogate variable: continuous time index as C."""
    data_df, var_names = small_data
    g = cdnots.discover(data_df, var_names, alpha=0.05, max_lag=2)
    _check_graph(g, var_names, "cdnots")
    assert "c_adjacent_vars" in g.metadata
    assert "n_c_adjacent" in g.metadata


def test_regime_pcmci(small_data):
    """Regime-PCMCI: PELT regime detection + PCMCI."""
    data_df, var_names = small_data
    g = regime_pcmci.discover(data_df, var_names, alpha=0.05, max_lag=2)
    _check_graph(g, var_names, "regime_pcmci")
    assert "regime_boundaries" in g.metadata
    assert "n_regimes" in g.metadata
    assert g.metadata["n_regimes"] >= 1


def test_regime_pcmci_fixed_regimes(small_data):
    data_df, var_names = small_data
    g = regime_pcmci.discover(data_df, var_names, alpha=0.05, max_lag=2, n_regimes=2)
    assert g.metadata["n_regimes"] == 2


def test_cedar(small_data):
    """CEDAR: distance correlation screening for low-N data."""
    data_df, var_names = small_data
    g = cedar.discover(data_df, var_names, alpha=0.05, max_lag=2)
    _check_graph(g, var_names, "cedar")
    assert "dcor_matrix" in g.metadata
    assert "use_c_node" in g.metadata


def test_rpcmci(small_data):
    """RPCMCI: joint regime + causal discovery (tigramite)."""
    data_df, var_names = small_data
    g = rpcmci.discover(data_df, var_names, alpha=0.05, max_lag=2, num_regimes=2)
    _check_graph(g, var_names, "rpcmci")
    assert "regimes" in g.metadata
    assert "num_regimes" in g.metadata
