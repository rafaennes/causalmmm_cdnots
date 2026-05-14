# Design: CD-NOTS Pipeline Fix — Data Format, FDR Control, Empirical Bayes Calibration

**Date:** 2026-05-13  
**Author:** Rafael Silva Ennes  
**Dissertation:** Investigação da Utilização de Descoberta de Estruturas Causais para Calibração de Modelos Bayesianos em MMM — Mackenzie 2026

---

## Context

The 4-arm experiment (`notebooks/experimento_4bracos.ipynb`) compares PyMC-Marketing and Meridian with and without CD-NOTS-informed priors. The current pipeline produces poor causal discovery results (Precision=0.27, FDR=0.73, F1=0.33 on the `causal_business` preset) and uses ad hoc prior calibration (`damping=0.5`, `floor=0.4` constants with no statistical grounding).

Root cause analysis identified two bugs and one design weakness:

| Issue | Root Cause | Effect |
|---|---|---|
| FDR=73% | `prepare_dataset_for_modeling` returns flat DataFrame; `discover_graph` treats all geos as 1 "national" series | PCMCI receives concatenated geos as single time series → artificial temporal dependencies |
| KCI instead of parcorr | Consequence of above (1 geo detected → KCI triggered) | Slower, higher false positive rate for N>150 |
| Calibration ad hoc | `sigma = sigma_base × (1 + 0.5 × direction × (1-p_value))` has no statistical grounding | Dissertation argument is weak; results hard to interpret |

---

## Scope

Targeted changes to two existing modules. Zero changes to the notebook or to PyMC/Meridian model architecture.

**In scope:**
- `arquivos_recentes/cdnots_discovery.py` — data format resolution + FDR correction
- `arquivos_recentes/cdnots_model_builder.py` — Empirical Bayes calibration formula

**Out of scope:**
- Changes to `mmm_param_recovery` (sibling repo)
- Changes to the notebook cells
- Changes to PyMC-Marketing or Meridian model specifications beyond prior parameters
- Re-designing the 4-arm experiment structure

---

## Component 1: Data Format Resolution (`cdnots_discovery.py`)

### 1a. `_resolve_data_format(data_df)` — new helper

Called at the top of `discover_graph`, before any geo extraction logic:

```python
def _resolve_data_format(data_df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data_df.index, pd.MultiIndex):
        return data_df                          # already correct
    time_col = next((c for c in ("date", "time") if c in data_df.columns), None)
    if time_col is not None and "geo" in data_df.columns:
        df = data_df.set_index([time_col, "geo"])
        df.index.names = ["date", "geo"]        # normalize to "date" for downstream compatibility
        return df
    return data_df                              # fallback: no geo info
```

**Handles both cases:**
- Single-geo presets (`causal_large`, `small_business`): 1 unique geo value → `len(geos)=1` → KCI (unchanged behavior)
- Multi-geo presets (`causal_business`, `medium_business`): N unique geo values → `len(geos)=N` → parcorr (corrected behavior)

The fallback (no `geo` column, no MultiIndex) preserves original "national" behavior.

### 1b. FDR correction in `_pcmci_discovery`

After `pcmci.run_pcmci(...)` returns raw `p_matrix`, apply Benjamini-Hochberg correction:

```python
q_matrix = pcmci.get_corrected_pvalues(
    p_matrix=results["p_matrix"],
    tau_min=1,
    tau_max=max_lag,
    fdr_method="fdr_bh",
)
```

Edge decisions use `q_matrix`; raw `p_matrix` is retained for display and debugging.

```python
for i in range(n_vars):
    for j in range(n_vars):
        if i == j:
            continue
        min_q = float(q_matrix[i, j, 1:max_lag+1].min())
        min_p = float(results["p_matrix"][i, j, 1:max_lag+1].min())
        qval[i, j] = min_q
        pval[i, j] = min_p
        if min_q < alpha:
            adj[i, j] = 1.0
```

**Rationale:** With 10 variables × 2 lags ≈ 180 CI tests at α=0.05, the expected number of false positives under the global null is 9. BH controls the false discovery rate (not FWER), which is appropriate for causal discovery where we expect a sparse true graph. Reference: Benjamini & Hochberg (1995), *JRSS-B*.

**PC and Granger fallbacks:** These do not have built-in FDR correction. They return `qval = pval` as a conservative approximation (undocumented as FDR-controlled). This is noted in code comments.

### 1c. `CausalGraph` dataclass — add `edge_qvalues`

```python
@dataclass(frozen=True)
class CausalGraph:
    adjacency_matrix: np.ndarray
    edge_pvalues: np.ndarray    # raw MCI p-values (display, debug)
    edge_qvalues: np.ndarray    # BH-corrected q-values (calibration)
    ...
```

`edge_pvalues` is kept for backward compatibility and interpretability. `edge_qvalues` is what `cdnots_model_builder` consumes.

---

## Component 2: Empirical Bayes Calibration (`cdnots_model_builder.py`)

### Theoretical grounding

Under the Empirical Bayes framework for multiple testing (Storey 2002; Efron 2010, *Large-Scale Inference*, Ch. 5), the BH q-value for hypothesis i satisfies:

```
q_i ≈ P(H₀ | data_i)
```

Therefore the **posterior inclusion probability** (PIP) — the probability that the causal edge is real given the data — is:

```
PIP_i = 1 - q_i
```

This connects FDR-controlled discovery directly to a Bayesian quantity, providing the theoretical bridge between the causal graph and the prior calibration.

### Prior adjustment formula

The adjusted sigma is a **continuous spike-and-slab relaxation** (Ishwaran & Rao 2005):

```
σ_adj = σ_base × (MIN_SIGMA_RATIO + (1 − MIN_SIGMA_RATIO) × PIP)
```

Where:
- `PIP = 1 - q_value` for **all** channels (direct, mediated, and excluded)
- For **direct** channels: q_value is the edge q-value of ch→y directly
- For **mediated** channels: q_value is the minimum q-value along the best path to y (weakest-link BFS, same logic as current `_path_min_confidence_pvalue` but on `edge_qvalues`)
- For **excluded** channels: q_value is the edge q-value of ch→y (which is high by definition since no edge was detected)
- `MIN_SIGMA_RATIO = 0.4` — ratio of spike width to slab width; preserves MCMC tractability (no degenerate delta-function prior)

The formula is **identical for all categories** — the classification (direct/mediated/excluded) determines which q_value to use, not the direction of the formula. This is intentional: a channel classified as excluded with q=0.99 gets PIP=0.01 → σ_adj ≈ 0.40 × σ_base (maximum shrinkage). A borderline excluded channel with q=0.10 gets PIP=0.90 → σ_adj ≈ 0.94 × σ_base (almost no shrinkage — the model correctly defers to the data in ambiguous cases).

**Properties:**

| Channel type | q_value | PIP = 1 - q | σ_adj / σ_base |
|---|---|---|---|
| Direct, strong evidence | 0.01 | 0.99 | ≈ 1.00 |
| Direct, moderate evidence | 0.20 | 0.80 | 0.88 |
| Mediated, borderline | 0.50 | 0.50 | 0.70 |
| Excluded, weak non-significance | 0.20 | 0.80 | 0.88 (model uncertain — data decides) |
| Excluded, confident | 0.90 | 0.10 | 0.46 |
| Excluded, very confident | 0.99 | 0.01 | ≈ 0.40 (max shrinkage) |

The `DAMPING = 0.5` constant is **removed**. The formula has no free parameters beyond `MIN_SIGMA_RATIO`.

### Implementation

Replace `_compute_sigma_multipliers` inner logic:

```python
# Before (ad hoc):
confidence = np.clip(1.0 - p_value, 0.0, 1.0)
multipliers[i] = 1.0 + damping * confidence  # if has_path
multipliers[i] = max(floor, 1.0 - damping * confidence)  # if excluded

# After (Empirical Bayes):
if has_direct:
    q_value = graph.edge_qvalues[ch_idx, y_idx]
elif ch in graph.mediated_channels:
    q_value = _path_min_confidence_pvalue(  # reused, now operates on qvalues
        graph.adjacency_matrix, graph.edge_qvalues, ch_idx, y_idx
    )
else:  # excluded
    q_value = graph.edge_qvalues[ch_idx, y_idx]  # high q → low PIP → max shrinkage

pip = np.clip(1.0 - q_value, 0.0, 1.0)   # same formula for all categories
multipliers[i] = MIN_SIGMA_RATIO + (1.0 - MIN_SIGMA_RATIO) * pip
```

For **mediated channels**, the weakest-link path q-value is used (same BFS logic as current `_path_min_confidence_pvalue`, but operating on `edge_qvalues` instead of `edge_pvalues`).

### Adstock parameters

`_compute_adstock_params` uses the same PIP to parametrize the Beta prior on geometric adstock decay:

```python
# BASE_B=3.0 (current default, Beta(1,3) = fast-decay prior)
# MIN_B=1.0  (Beta(1,1) = Uniform = maximally flexible)
# Channels with high PIP get more flexible adstock (wider Beta → lower alpha_b)
# Channels with low PIP get fast-decay prior (higher alpha_b)
BASE_B, MIN_B = 3.0, 1.0
alpha_b[i] = BASE_B - (BASE_B - MIN_B) * pip   # pip→1: alpha_b=1 (uniform); pip→0: alpha_b=3
alpha_b[i] = np.clip(alpha_b[i], MIN_B, 5.0)
```

### Constants cleanup

| Constant | Before | After |
|---|---|---|
| `DAMPING` | 0.5 | **removed** |
| `FLOOR` | 0.4 | renamed `MIN_SIGMA_RATIO = 0.4`, documented |

---

## References for dissertation

- Benjamini, Y.; Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS-B*, 57(1), 289-300.
- Storey, J.D. (2002). A direct approach to false discovery rates. *JRSS-B*, 64(3), 479-498.
- Efron, B. (2010). *Large-Scale Inference: Empirical Bayes Methods for Estimation, Testing, and Prediction*. Cambridge. Cap. 5.
- Ishwaran, H.; Rao, J.S. (2005). Spike and slab variable selection: frequentist and Bayesian strategies. *Annals of Statistics*, 33(2), 730-773.

---

## Testing / validation

After implementation, re-run the causal discovery on `causal_business` and verify:

1. **Format fix**: `ci_test_used` should be `"parcorr"` (not `"kci"`) for `causal_business` (4 geos)
2. **FDR improvement**: FDR should drop from 0.73 toward ≤0.30; Precision should rise toward ≥0.60
3. **Prior calibration**: `_log_adjustments` output should show meaningful differentiation between channels with PIP≈1 (direct, low q) vs PIP≈0 (excluded)
4. **MCMC convergence**: R-hat < 1.05 for all parameters (no divergences from prior-likelihood conflict)

---

## Files changed

| File | Type of change |
|---|---|
| `arquivos_recentes/cdnots_discovery.py` | Add `_resolve_data_format`, add FDR correction, add `edge_qvalues` to `CausalGraph` |
| `arquivos_recentes/cdnots_model_builder.py` | Replace calibration formula, remove `DAMPING`, rename `FLOOR` |
