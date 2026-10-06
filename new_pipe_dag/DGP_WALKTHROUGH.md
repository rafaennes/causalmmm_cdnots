# How synthetic MMM data is generated

Step-by-step walkthrough of `generate_ladder()` in `dgp.py`, with parallels
to how real Marketing Mix Modeling data works.

---

## The causal graph we're simulating

```
              season (latent or observed)
             /    |    \          \
            v     v     v          v
  x1 -----> x2   x3    x4        y
   \         \    |    /         ^
    \         \   |   /         /
     \         v  v  v         /
      +------> y <------------+
```

Every channel causes y. Some channels cause other channels (spillover).
Seasonality — when active — causes both channels and y simultaneously, making
it a confounder.

---

## Step 0: The seasonal confounder

**Real world.** Companies spend more during Christmas, Black Friday, summer.
Consumer demand also peaks in those periods. Seasonality drives *both* spend
and sales — it's a common cause.

```python
# dgp.py — seasonal driver
season_active = cfg.seasonality in ("S0", "S1")
omega_t = 2 * np.pi * np.arange(T) / 52  # annual cycle, weekly resolution

if season_active:
    if cfg.phase_spread == "equispaced":
        # [0, pi) avoids anticorrelated pairs: cos(wt-pi) = -cos(wt)
        phases = np.linspace(0, np.pi, n_ch, endpoint=False)
    elif cfg.phase_spread == "moderate":
        phases = rng.normal(0.0, 1.0, n_ch)
    else:  # concentrated — all channels peak near the same week
        phases = rng.normal(0.0, 0.2, n_ch)
    # Per-channel shifted seasonal signals
    season_signals = np.array([np.cos(omega_t - phi) for phi in phases])
    season_ref = np.cos(omega_t)  # unshifted reference for season→y
```

Each channel gets its own **phase-shifted** seasonal signal `cos(wt - phi_i)`.
The phase controls *when* the channel peaks in the year:

- **Equispaced** over `[0, pi)`: with 4 channels, phases = `[0, pi/4, pi/2, 3pi/4]`.
  Peaks at weeks 0, 6.5, 13, 19.5. Max pairwise `|corr| ≈ 0.71` — moderate,
  not orthogonal but not collinear. (Range `[0, 2pi)` would produce
  anticorrelated pairs since `cos(wt-pi) = -cos(wt)`.)
- **Concentrated** (`N(0, 0.2)`): all phases ≈ 0, all channels peak near
  the same week. They move together — massive collinearity from a shared
  hidden cause.

This is generated first because in reality seasonality exists before any
marketing decision. It's the environment the advertiser operates in.

---

## Step 1: Channel spend (innovations)

**Real world.** Each channel has a weekly budget. The media team sets a base
spend level with natural fluctuation (campaigns starting/ending, budget
adjustments). Spend is always positive — you can't un-spend money.

```python
# dgp.py — channel innovations
innovations = {}
for i in range(n_ch):
    base = rng.uniform(800, 2000)
    noise = rng.normal(0, base * 0.12, T)
    seasonal_part = base * 0.3 * season_signals[i] if season_active else 0.0
    innovations[i] = np.clip(base + seasonal_part + noise, 0, None)
```

Each channel gets:
- A constant base level (`U(800, 2000)`) — the typical weekly budget.
- Gaussian noise (12% of base) — week-to-week fluctuation.
- A seasonal component (`0.3 * base * cos(wt - phi_i)`) — spend rises and
  falls with the channel's seasonal phase.
- A floor at 0 — no negative spend.

At this point the channels are independent of each other (except for sharing
the seasonal driver from Step 0).

---

## Step 2: Inter-channel spillover

**Real world.** TV ads drive brand searches → paid search spend increases.
Display retargeting only works if people visited the site first (driven by
other channels). One channel's activity spills into another, with a delay.

```python
# dgp.py — spillover (order-independent)
# Freeze innovations so edge processing order doesn't matter.
pre_spillover = {k: v.copy() for k, v in innovations.items()}
for src, tgt in sorted(edges):  # canonical order for RNG determinism
    coeff = rng.uniform(0.15, 0.25)

    if cfg.saturation_on:
        src_signal = _hill(pre_spillover[src])
        src_signal = src_signal * pre_spillover[src].std() / (src_signal.std() + 1e-8)
    else:
        src_signal = pre_spillover[src]

    innovations[tgt][1:] += coeff * src_signal[:-1]         # lag 1
    innovations[tgt][2:] += (coeff * 0.4) * src_signal[:-2] # lag 2
```

For each causal edge `(src, tgt)`:
- The target receives 15–25% of the source's signal at **lag 1**.
- Plus 40% of that coefficient at **lag 2** (decaying echo).
- Optional Hill saturation on the source: large TV campaigns have diminishing
  spillover into search.

Two important design choices:
1. **Frozen pre-spillover copy**: spillover is computed from the *original*
   innovations, not the already-modified values. This prevents transitive
   effects (x1→x2→x3) that aren't explicitly declared in the edge list, and
   makes results independent of edge processing order.
2. **Sorted edges + canonical RNG**: edges are sorted before processing so the
   same coefficients are assigned regardless of the order specified by the
   caller.

---

## Step 3: Adstock (carryover)

**Real world.** A TV ad doesn't only affect sales the week it airs. Someone
sees the ad on Monday, remembers the brand, and buys three weeks later. The
effect of spend decays geometrically over time — this temporal smearing is
called adstock or carryover.

```python
# dgp.py — geometric adstock
spend = {}
for i in range(n_ch):
    x = innovations[i].copy()
    if cfg.adstock_on:
        decay = rng.uniform(0.3, 0.5)
        for t in range(1, T):
            x[t] = x[t] + decay * x[t - 1]
    spend[i] = x
```

A geometric filter: `spend[t] = innovation[t] + decay * spend[t-1]`, with
`decay ~ U(0.3, 0.5)`.

**Why it matters for causal discovery.** Adstock creates strong
autocorrelation in every channel. This inflates lag-1 correlations between
*any* pair of channels, even those with no causal relationship. A naive
Granger test might see "x1 predicts x2" just because both have similar
carryover shapes. In the benchmark, adstock raised `max_abs_corr` from 0.10
to 0.43.

---

## Step 4: Channel → y (media effect on sales)

**Real world.** Each dollar spent on channel *i* generates incremental sales,
but with diminishing returns (the 10th million in TV does less than the 1st)
and with a delay (this week's spend affects next week's sales).

### 4a. Beta calibration

```python
# dgp.py — beta calibration at fixed reference
# REFERENCE_NOISE = 500.0 — same for all levels, so SNR varies with noise_std
target_per_ch = REFERENCE_NOISE * 3.0 / max(n_ch, 1)

betas = []
for i in range(n_ch):
    s_i = _hill(spend[i]) if cfg.saturation_on else spend[i]
    std_i = s_i.std() + 1e-8
    betas.append(target_per_ch / std_i * rng.uniform(0.7, 1.0))
```

Betas are calibrated at a **fixed reference noise level** (`REFERENCE_NOISE=500`),
not coupled to `cfg.noise_std`. This means:
- L0 (noise≈0) and L1 (noise=0.02) get the **same betas** but different noise
  levels, so SNR actually changes between them.
- Total channel signal std ≈ `sqrt(n_ch) * target_per_ch` (not `n_ch * target_per_ch`,
  because independent signals add in quadrature). Per-channel SNR at the reference
  noise level is `target_per_ch / REFERENCE_NOISE = 0.75`.

### 4b. Hill saturation

```python
def _hill(x):
    """Hill saturation: x^s / (kappa^s + x^s), kappa=p25(x), slope=3."""
    kappa = np.percentile(x, 25) + 1e-8
    return x**3 / (kappa**3 + x**3)
```

When `saturation_on=True`, both the channel→y and spillover paths use this
Hill function. Over the realized spend range (`[~1300, ~2700]`, <2x dynamic
range), even the Hill function's nonlinearity is moderate (corr ≈ 0.994 with
raw spend). This is inherent to the narrow spend distribution — any smooth
monotone transform over a narrow range approximates linearity.

### 4c. Composing y

```python
# dgp.py — building y
base_sales = 10000.0
y = np.full(T, base_sales)
if season_active:
    y += 0.05 * base_sales * season_ref  # season → y (unshifted reference)
for i in range(n_ch):
    s_i = _hill(spend[i]) if cfg.saturation_on else spend[i]
    y[1:] += betas[i] * s_i[:-1]         # channels → y at lag 1
```

Sales are built up from:
1. **Baseline** (10,000) — organic demand, brand equity, what you'd sell with
   zero advertising.
2. **Seasonal effect** — 5% swing in baseline sales, using the unshifted
   reference `cos(wt)`. This is the second path through which seasonality
   hits y (the first was through channel spend in Step 1).
3. **Media effects at lag 1** — each channel's (possibly saturated) spend from
   last period contributes to this period's sales.

---

## Step 5: Controls and noise

**Real world.** Macro variables (GDP, weather, holidays, competitor pricing)
affect sales independently of advertising. And there's always irreducible
randomness — stockouts, viral tweets, random demand fluctuation.

```python
# dgp.py — controls and noise
# Controls cause y but are excluded from the ground truth by design:
# the discovery task is restricted to the media system {channels, [season], y}.
controls = {}
for j in range(n_co):
    c = rng.normal(0, 1, T)
    controls[j] = c
    y += rng.uniform(-30, 30) * c
noise_level = 500 * cfg.noise_std / 0.02
y += rng.normal(0, noise_level, T)
```

Controls are nuisance variables with a contemporaneous effect on y. They are
excluded from the ground truth because the discovery task is restricted to the
media system `{channels, [season], y}` — not because they have no causal
effect (they do: `c → y` is a real DGP edge), but because evaluating control
recovery is a separate question from evaluating media graph recovery.

---

## Step 6: What's observed vs. what's true

### The observed DataFrame

| Seasonality mode | Columns in DataFrame |
|-----------------|---------------------|
| `"off"` | `x1, x2, x3, x4, c1, y` |
| `"S1"` (observed) | `x1, x2, x3, x4, c1, season, y` |
| `"S0"` (hidden) | `x1, x2, x3, x4, c1, y` — season affected everything but is invisible |

### The ground truth adjacency

```python
# dgp.py — truth graph
# All channels → y
for ch in ch_names:
    true_adj[idx[ch], idx["y"]] = 1.0
# Inter-channel spillover
for src, tgt in edges:
    true_adj[idx[ch_names[src]], idx[ch_names[tgt]]] = 1.0
# Season → ALL channels and y (when active, every channel gets loading 0.3)
if season_active:
    true_adj[idx["season"], idx["y"]] = 1.0
    for ch in ch_names:
        true_adj[idx["season"], idx[ch]] = 1.0
```

Note: season enters the ground truth **even when hidden** (`S0`). This means
algorithms are evaluated against edges they cannot directly observe.

### Edge lags (for recall ceiling)

```python
edge_lags[(ch, "y")] = [1]                          # media effect: lag 1
edge_lags[(ch_names[src], ch_names[tgt])] = [1, 2]  # spillover: lag 1 and 2
edge_lags[("season", "y")] = [0]                     # contemporaneous
edge_lags[("season", ch)] = [0]                      # contemporaneous
```

Season edges are structurally **lag 0** (contemporaneous). However, because
`season(t-1) ≈ season(t)` (autocorrelation ≈ 0.993), methods constrained to
`tau_min=1` can detect them as lag-1 proxies. The `recall_ceiling` computation
treats them as unrecoverable, which is conservative.

---

## The ladder: each level mapped to the steps above

The design principle: each level turns on **one mechanism** from the steps
above, so when performance drops we know which mechanism caused it.

| Level | Steps active | What's new | Real-world analogue |
|-------|-------------|-----------|-------------------|
| **L0** | 1, 2, 4 | Baseline: T=1040, noise≈0 | 20 years of perfect data |
| **L1** | 1, 2, 4, 5 | + realistic noise (SNR drops) | Same long history, real measurement noise |
| **L2** | 1, 2, 4, 5 | + short series (T=156) | 3 years weekly — what most companies have |
| **L3** | 1, 2, **3**, 4, 5 | + adstock | Carryover creates autocorrelation |
| **L4** | 1, 2, 4, 5 | + Hill saturation (both paths) | Diminishing returns |
| **L5** | **0**, 1, 2, 4, 5 | + observed seasonality | Confounder exists but analyst can see it |
| **L6** | **0**, 1, 2, 4, 5 | + hidden seasonality (equispaced) | Confounder exists, analyst cannot see it |
| **L7** | **0**, 1, 2, 4, 5 | + hidden + concentrated phases | All channels peak together — collinearity bomb |
| **L8** | **0**, 1, 2, **3**, 4, 5 | Everything together | The "realistic" MMM dataset |

L0–L2 vary the statistical regime (noise, sample size). L3 introduces
carryover. L4 adds nonlinearity. L5–L7 introduce the seasonal confounder with
increasing adversarial strength. L8 is the full realistic scenario.

### Known limitation: saturation effect is inherently moderate

The Hill function (slope=3, kappa=p25) is genuinely nonlinear, unlike the
previous `log1p` (which was nearly linear: corr=0.999 with raw spend). But
over the narrow spend range generated by the DGP (`[~1300, ~2700]`), even the
Hill function's distortion is limited (corr ≈ 0.994). This is a property of
the spend distribution, not the saturation function. Widening the dynamic range
(e.g., on/off campaign patterns with zero-spend periods) would amplify the
effect, but that changes the DGP structure beyond the current scope.
