import numpy as np
import pytest
from causal_discovery.graph import CausalGraph, bh_correct


def _make_graph(adj, var_names=None):
    n = adj.shape[0]
    if var_names is None:
        var_names = tuple(f"x{i}" for i in range(n - 1)) + ("y",)
    return CausalGraph(
        adjacency_matrix=adj,
        variable_names=var_names,
        algorithm="test",
        runtime_seconds=0.1,
        metadata={},
    )


def test_evaluate_perfect():
    true = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    g = _make_graph(true.copy())
    m = g.evaluate(true)
    assert m["precision"] == pytest.approx(1.0, abs=1e-6)
    assert m["recall"] == pytest.approx(1.0, abs=1e-6)
    assert m["f1"] == pytest.approx(1.0, abs=1e-6)
    assert m["shd"] == 0
    assert m["tp"] == 2
    assert m["fp"] == 0
    assert m["fn"] == 0


def test_evaluate_empty_prediction():
    true = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    pred = np.zeros((3, 3), dtype=float)
    g = _make_graph(pred)
    m = g.evaluate(true)
    assert m["recall"] == pytest.approx(0.0, abs=1e-5)
    assert m["fp"] == 0
    assert m["fn"] == 2
    assert m["shd"] == 2


def test_evaluate_full_prediction():
    true = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    pred = np.ones((3, 3)) - np.eye(3)
    g = _make_graph(pred)
    m = g.evaluate(true)
    assert m["recall"] == pytest.approx(1.0, abs=1e-5)
    assert m["tp"] == 2
    assert m["fp"] == 4


def test_evaluate_var_subset():
    # 4-var graph (x1, x2, c1, y); true is 3-var (x1, x2, y)
    true = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]], dtype=float)
    adj_4 = np.zeros((4, 4))
    adj_4[0, 3] = 1.0  # x1 → y
    adj_4[1, 3] = 1.0  # x2 → y
    g = CausalGraph(
        adjacency_matrix=adj_4,
        variable_names=("x1", "x2", "c1", "y"),
        algorithm="test",
        runtime_seconds=0.1,
        metadata={},
    )
    m = g.evaluate(true, var_subset=["x1", "x2", "y"])
    assert m["tp"] == 2
    assert m["fp"] == 0
    assert m["fn"] == 0


def test_bh_correct_all_significant():
    n = 4
    pval = np.full((n, n), 0.001)
    np.fill_diagonal(pval, 1.0)
    adj = bh_correct(pval, alpha=0.05)
    mask = ~np.eye(n, dtype=bool)
    assert adj[mask].all()
    assert adj.diagonal().sum() == 0


def test_bh_correct_none_significant():
    n = 4
    pval = np.full((n, n), 0.9)
    np.fill_diagonal(pval, 1.0)
    adj = bh_correct(pval, alpha=0.05)
    assert adj.sum() == 0
