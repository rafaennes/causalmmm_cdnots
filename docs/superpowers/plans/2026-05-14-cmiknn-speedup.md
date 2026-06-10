# CMIknn Speedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce CMIknn/PCMCI wall time by 15–30× for single-geo presets by capping conditioning-set dimensionality and restricting the search to structurally possible MMM edges.

**Architecture:** Two changes confined entirely to `arquivos_recentes/cdnots_discovery.py`. Approach A tunes three PCMCI/CMIknn parameters (`max_conds_dim`, `null_fit`, `sig_samples`). Approach B adds a `_build_mmm_link_assumptions()` helper that encodes prior structural knowledge (exogenous controls, y is terminal) so PCMCI skips ~30% of CI tests and uses smaller conditioning sets throughout the PC step.

**Tech Stack:** tigramite 5.2.x (`PCMCI`, `CMIknn`, `ParCorr`), numpy, pytest.

---

## File Map

| File | Action | What changes |
|---|---|---|
| `arquivos_recentes/cdnots_discovery.py` | Modify | Add `_build_mmm_link_assumptions()`; update `_pcmci_discovery` and `_discover_single_geo` signatures and bodies |
| `tests/test_cdnots_discovery_speedup.py` | Create | Unit + integration tests for the two approaches |

---

### Task 1: Test suite scaffold + unit tests for `_build_mmm_link_assumptions`

**Files:**
- Create: `tests/test_cdnots_discovery_speedup.py`

- [ ] **Step 1: Create the test file**

```python
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
    for i in range(n_vars - 1):          # channels + controls, not y
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
```

- [ ] **Step 2: Run tests — expect `AttributeError` (function not defined yet)**

```bash
cd /home/ennes/mestrado/causalmmm_with_cdnots
/home/ennes/mestrado/pymc_meridian_comparison/.pixi/envs/default/bin/python3.12 \
    -m pytest tests/test_cdnots_discovery_speedup.py::test_y_receives_from_all_channels_and_controls -v 2>&1 | tail -20
```

Expected: `AttributeError: module … has no attribute '_build_mmm_link_assumptions'`

---

### Task 2: Implement `_build_mmm_link_assumptions` and pass structural info down the call chain

**Files:**
- Modify: `arquivos_recentes/cdnots_discovery.py`

- [ ] **Step 1: Add `_build_mmm_link_assumptions` after `_granger_fallback`**

Insert the following function before `_has_path` (around line 470 in the current file — after the closing of `_granger_fallback`):

```python
def _build_mmm_link_assumptions(
    n_channels: int,
    n_controls: int,
    n_vars: int,
    max_lag: int,
) -> dict:
    """Restrict PCMCI to structurally possible edges in MMM.

    Allowed edges (tested):
      - channel/control → y      (direct / mediated detection)
      - channel_i → channel_j    (mediated path between channels)
      - channel_i → channel_i    (self-lag / autocorrelation conditioning)
      - control   → channel      (endogeneity detection)

    Forbidden edges (skipped, saves ~30% of CI tests):
      - y → anything             (y is the terminal outcome)
      - channel → control        (controls are exogenous by design)
      - control → control        (controls are assumed independent)
    """
    y_idx        = n_vars - 1
    channel_idxs = list(range(n_channels))
    control_idxs = list(range(n_channels, n_channels + n_controls))
    lags         = [-tau for tau in range(1, max_lag + 1)]

    la: dict = {j: {} for j in range(n_vars)}

    # anything → y
    for i in range(n_vars - 1):
        for lag in lags:
            la[y_idx][(i, lag)] = "?->"

    # channel/control → channel  (includes self-lags for autocorrelation)
    for j in channel_idxs:
        for i in channel_idxs + control_idxs:
            for lag in lags:
                la[j][(i, lag)] = "?->"

    # controls and y stay empty: no incoming edges

    return la
```

- [ ] **Step 2: Update `_discover_single_geo` signature to accept and forward `n_channels` / `n_controls`**

Replace the current function signature and body at line 299:

```python
def _discover_single_geo(
    data: np.ndarray,
    n_vars: int,
    alpha: float,
    max_lag: int,
    ci_test: str,
    n_channels: int = 0,
    n_controls: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run discovery on a single geo. PCMCI primary, PC fallback, Granger last resort.

    Returns (adj, pval, qval). For PC and Granger fallbacks qval=pval (no FDR
    correction available); this is documented and conservative.
    """
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    try:
        return _pcmci_discovery(data, n_vars, alpha, max_lag, ci_test,
                                n_channels, n_controls)
    except ImportError:
        warnings.warn(
            "tigramite not available, falling back to PC + temporal augmentation. "
            "Install with: pip install tigramite"
        )
        try:
            adj, pval = _pc_fallback(data, n_vars, alpha, max_lag, ci_test)
            return adj, pval, pval
        except ImportError:
            warnings.warn("causal-learn not available either, using Granger causality")
            adj, pval = _granger_fallback(data, n_vars, alpha, max_lag)
            return adj, pval, pval
```

- [ ] **Step 3: Update the call site in `discover_graph` (line 205)**

Replace:
```python
adj, pval, qval = _discover_single_geo(geo_data, n_vars, alpha, max_lag, ci_test)
```
With:
```python
adj, pval, qval = _discover_single_geo(
    geo_data, n_vars, alpha, max_lag, ci_test, n_channels, n_controls
)
```

- [ ] **Step 4: Run Task 1 tests — all 6 should pass**

```bash
cd /home/ennes/mestrado/causalmmm_with_cdnots
/home/ennes/mestrado/pymc_meridian_comparison/.pixi/envs/default/bin/python3.12 \
    -m pytest tests/test_cdnots_discovery_speedup.py -k "link_assumptions or channel or control or y_is_never or link_type" -v 2>&1 | tail -25
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add arquivos_recentes/cdnots_discovery.py tests/test_cdnots_discovery_speedup.py
git commit -m "feat: add _build_mmm_link_assumptions for targeted PCMCI search"
```

---

### Task 3: Apply Approach A — CMIknn parameter tuning and `max_conds_dim`

**Files:**
- Modify: `arquivos_recentes/cdnots_discovery.py:330–420`

- [ ] **Step 1: Update `_pcmci_discovery` signature to accept `n_channels` / `n_controls`**

Replace the function signature at line 330:

```python
def _pcmci_discovery(
    data: np.ndarray,
    n_vars: int,
    alpha: float,
    max_lag: int,
    ci_test_name: str,
    n_channels: int = 0,
    n_controls: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
```

- [ ] **Step 2: Apply A — tune CMIknn and add `max_conds_dim` + `link_assumptions` to `run_pcmci`**

Replace the CMIknn construction line (currently line 381):
```python
            cond_ind_test = CMIknn(knn=5)
```
With:
```python
            cond_ind_test = CMIknn(knn=5, null_fit=True, sig_samples=200)
```

Then replace the `run_pcmci` call (currently line 394):
```python
    results = pcmci.run_pcmci(tau_max=max_lag, tau_min=1, pc_alpha=alpha)
```
With:
```python
    # B: build targeted link_assumptions when structural info is available
    link_assumptions = None
    if n_channels > 0:
        link_assumptions = _build_mmm_link_assumptions(
            n_channels, n_controls, n_vars, max_lag
        )

    # A: max_conds_dim=4 caps k-NN dimensionality (curse of dimensionality);
    # link_assumptions restricts the PC and MCI phases to MMM-relevant edges.
    results = pcmci.run_pcmci(
        tau_max=max_lag,
        tau_min=1,
        pc_alpha=alpha,
        max_conds_dim=4,
        link_assumptions=link_assumptions,
    )
```

- [ ] **Step 3: Run link_assumptions tests again to confirm nothing broke**

```bash
cd /home/ennes/mestrado/causalmmm_with_cdnots
/home/ennes/mestrado/pymc_meridian_comparison/.pixi/envs/default/bin/python3.12 \
    -m pytest tests/test_cdnots_discovery_speedup.py -v 2>&1 | tail -25
```

Expected: all existing tests still pass.

- [ ] **Step 4: Commit**

```bash
git add arquivos_recentes/cdnots_discovery.py
git commit -m "perf: CMIknn null_fit+sig_samples=200, max_conds_dim=4, MMM link_assumptions"
```

---

### Task 4: Integration tests — correctness and runtime

**Files:**
- Modify: `tests/test_cdnots_discovery_speedup.py`

- [ ] **Step 1: Add integration tests to the existing test file**

Append to `tests/test_cdnots_discovery_speedup.py`:

```python
# ── Task 4 tests: integration ─────────────────────────────────────────────────

def test_pcmci_discovery_returns_correct_shapes():
    """_pcmci_discovery must return (adj, pval, qval) of shape (n_vars, n_vars)."""
    n_vars, n_ch, n_ctrl = 6, 4, 1
    data = _small_data(n_vars=n_vars, T=80)
    adj, pval, qval = disc._pcmci_discovery(
        data, n_vars, alpha=0.1, max_lag=1,
        ci_test_name="parcorr",   # parcorr for speed in tests
        n_channels=n_ch, n_controls=n_ctrl,
    )
    assert adj.shape  == (n_vars, n_vars)
    assert pval.shape == (n_vars, n_vars)
    assert qval.shape == (n_vars, n_vars)


def test_pcmci_discovery_y_has_no_outgoing_edges():
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


def test_pcmci_discovery_controls_have_no_incoming():
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


def test_discover_graph_completes_on_small_business_parcorr(tmp_path):
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
    total = len(graph.direct_channels) + len(graph.mediated_channels) + len(graph.excluded_channels)
    assert total == len(ch_cols), "every channel must be classified"


def test_cmiknn_runtime_improvement():
    """CMIknn with new params should complete in under 60s on 10-var, 100-row data."""
    import time
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
```

- [ ] **Step 2: Run the correctness tests (parcorr mode, fast)**

```bash
cd /home/ennes/mestrado/causalmmm_with_cdnots
/home/ennes/mestrado/pymc_meridian_comparison/.pixi/envs/default/bin/python3.12 \
    -m pytest tests/test_cdnots_discovery_speedup.py -k "not runtime" -v 2>&1 | tail -30
```

Expected: `9 passed`

- [ ] **Step 3: Run the runtime test (kci mode — this is the actual benchmark)**

```bash
cd /home/ennes/mestrado/causalmmm_with_cdnots
/home/ennes/mestrado/pymc_meridian_comparison/.pixi/envs/default/bin/python3.12 \
    -m pytest tests/test_cdnots_discovery_speedup.py::test_cmiknn_runtime_improvement -v -s 2>&1 | tail -15
```

Expected: `PASSED` (under 60 seconds). If it times out, the parameter changes in Task 3 need investigation.

- [ ] **Step 4: Commit**

```bash
git add tests/test_cdnots_discovery_speedup.py
git commit -m "test: integration + runtime tests for CMIknn speedup"
```

---

### Task 5: Smoke-test on `causal_business` preset with `ci_test=kci`

This confirms the full pipeline runs end-to-end with the new code before restarting the long experiments.

- [ ] **Step 1: Run a quick smoke test**

```bash
cd /home/ennes/mestrado/causalmmm_with_cdnots
timeout 300 \
/home/ennes/mestrado/pymc_meridian_comparison/.pixi/envs/default/bin/python3.12 - <<'EOF'
import sys, os, warnings
warnings.filterwarnings("ignore")
PYMC_ROOT = "/home/ennes/mestrado/pymc_meridian_comparison"
ROOT      = "/home/ennes/mestrado/causalmmm_with_cdnots"
for p in [PYMC_ROOT, ROOT]:
    if p not in sys.path: sys.path.insert(0, p)

import importlib.util

def _load(name, path):
    full = f"mmm_param_recovery.benchmarking.{name}"
    spec = importlib.util.spec_from_file_location(full, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod; spec.loader.exec_module(mod); return mod

ARQUIVOS = os.path.join(ROOT, "arquivos_recentes")
disc = _load("cdnots_discovery", os.path.join(ARQUIVOS, "cdnots_discovery.py"))

from mmm_param_recovery.data_generator.core import generate_mmm_dataset
from mmm_param_recovery.data_generator.presets import get_preset_config
from mmm_param_recovery.benchmarking.data_loader import prepare_dataset_for_modeling

config = get_preset_config("causal_business", seed=20250715)
result = generate_mmm_dataset(config)
data_df, ch_cols, ctrl_cols, _ = prepare_dataset_for_modeling(result)

import time; t0 = time.perf_counter()
graph = disc.discover_graph(data_df, ch_cols, control_columns=ctrl_cols,
                            alpha=0.05, max_lag=2, ci_test="kci")
print(f"\nRuntime: {time.perf_counter()-t0:.1f}s")
print(f"Edges:   {graph.n_edges}")
print(f"CI test: {graph.ci_test_used}")
print(f"Direct:  {graph.direct_channels}")
EOF
```

Expected: completes in under 5 minutes (vs. hours before), prints graph summary.

- [ ] **Step 2: If the smoke test passes, commit the final state**

```bash
git add arquivos_recentes/cdnots_discovery.py tests/test_cdnots_discovery_speedup.py
git commit -m "perf: CMIknn speedup complete — A+B, smoke-tested on causal_business"
```

---

## Self-Review

**Spec coverage:**
- ✅ A — `null_fit=True, sig_samples=200`: Task 3 Step 2
- ✅ A — `max_conds_dim=4`: Task 3 Step 2
- ✅ B — `_build_mmm_link_assumptions`: Tasks 1–2
- ✅ B — wired into `_pcmci_discovery` via `n_channels`/`n_controls`: Tasks 2–3
- ✅ Call-site update in `discover_graph`: Task 2 Step 3
- ✅ Integration + runtime tests: Task 4
- ✅ End-to-end smoke test: Task 5

**Placeholder scan:** No TBDs. All code blocks are complete.

**Type consistency:**
- `_build_mmm_link_assumptions(n_channels, n_controls, n_vars, max_lag)` — defined Task 2, called Task 3 ✅
- `_discover_single_geo(..., n_channels, n_controls)` — signature Task 2, call-site Task 2 Step 3 ✅
- `_pcmci_discovery(..., n_channels, n_controls)` — signature Task 3, called from `_discover_single_geo` Task 2 ✅
