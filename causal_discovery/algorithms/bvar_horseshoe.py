"""Bayesian VAR with regularized horseshoe prior for causal discovery.

Fits a VAR(p) model where each lag coefficient gets a Finnish horseshoe
prior (Piironen & Vehtari, 2017). Edges are declared when the posterior
credible interval for the max-abs lag coefficient excludes zero.

Unlike frequentist Granger (F-test per pair), this fits all variables
jointly, so shrinkage is informed by the full system — which may handle
collinearity more gracefully.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import pymc as pm
from sklearn.preprocessing import StandardScaler

from causal_discovery.graph import CausalGraph


def discover(
    data: pd.DataFrame,
    var_names: list[str],
    alpha: float = 0.05,
    max_lag: int = 2,
    **kwargs,
) -> CausalGraph:
    """Bayesian VAR with horseshoe prior → binary adjacency.

    Parameters
    ----------
    data : DataFrame with columns matching var_names.
    var_names : column names; last element is "y".
    alpha : credible interval width = 1 - alpha (e.g. 0.05 → 95% CI).
    max_lag : number of lags in the VAR.
    """
    start = time.perf_counter()
    n_draws = kwargs.get("n_draws", 1000)
    n_tune = kwargs.get("n_tune", 1000)
    n_chains = kwargs.get("n_chains", 2)
    target_accept = kwargs.get("target_accept", 0.9)

    forbidden = _parse_forbidden(kwargs.get("forbidden_links"), var_names)

    X_raw = data[var_names].values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_raw)

    n_vars = len(var_names)
    T = X_sc.shape[0]

    # Build lag matrix: each row t has [X_{t-1}, X_{t-2}, ..., X_{t-p}]
    Y = X_sc[max_lag:]  # (T-p, n_vars)
    X_lags = np.concatenate(
        [X_sc[max_lag - lag: T - lag] for lag in range(1, max_lag + 1)],
        axis=1,
    )  # (T-p, n_vars * max_lag)

    T_eff, n_predictors = X_lags.shape

    # --- PyMC model: VAR with regularized horseshoe ---
    with pm.Model() as model:
        # Regularized (Finnish) horseshoe (Piironen & Vehtari 2017)
        # ponytail: slab keeps coefficients finite even when tau is large
        tau = pm.HalfCauchy("tau", beta=1.0)
        lam = pm.HalfCauchy("lam", beta=1.0, shape=(n_predictors, n_vars))
        c2 = pm.InverseGamma("c2", alpha=2.0, beta=1.0)
        lam_tilde = lam * pm.math.sqrt(c2 / (c2 + tau**2 * lam**2))
        beta = pm.Normal("beta", mu=0, sigma=tau * lam_tilde,
                         shape=(n_predictors, n_vars))
        sigma = pm.HalfNormal("sigma", sigma=1.0, shape=n_vars)

        mu = pm.math.dot(X_lags, beta)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=Y)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = pm.sample(
                draws=n_draws, tune=n_tune, chains=n_chains,
                target_accept=target_accept,
                progressbar=False, return_inferencedata=True,
                compute_convergence_checks=False,
            )

    # --- Extract adjacency from posterior ---
    beta_samples = trace.posterior["beta"].values  # (chains, draws, n_predictors, n_vars)
    beta_samples = beta_samples.reshape(-1, n_predictors, n_vars)  # (total_samples, pred, var)

    # Reshape to (samples, max_lag, n_vars_src, n_vars_tgt)
    beta_by_lag = beta_samples.reshape(-1, max_lag, n_vars, n_vars)

    # For edge i→j: take the max absolute coefficient across lags,
    # then check if the CI for that max excludes zero.
    # More robust: check if ANY lag's CI excludes zero.
    half_alpha = alpha / 2
    adj = np.zeros((n_vars, n_vars))
    pip = np.zeros((n_vars, n_vars))  # posterior inclusion probability

    for i in range(n_vars):
        for j in range(n_vars):
            if i == j or (i, j) in forbidden:
                continue
            # Collect coefficients for i→j across all lags
            # beta_by_lag[:, lag, src=i, tgt=j]
            for lag in range(max_lag):
                samples = beta_by_lag[:, lag, i, j]
                lo = np.percentile(samples, 100 * half_alpha)
                hi = np.percentile(samples, 100 * (1 - half_alpha))
                if lo > 0 or hi < 0:  # CI excludes zero at any lag → edge
                    adj[i, j] = 1.0
                    break
            # PIP: fraction of samples where |max-lag coeff| > threshold
            max_abs = np.max(np.abs(beta_by_lag[:, :, i, j]), axis=1)
            pip[i, j] = float(np.mean(max_abs > 0.05))

    return CausalGraph(
        adjacency_matrix=adj,
        variable_names=tuple(var_names),
        algorithm="bvar_horseshoe",
        runtime_seconds=time.perf_counter() - start,
        metadata={
            "ci_test": "bayesian_horseshoe",
            "pip": pip,
            "n_draws": n_draws,
            "n_tune": n_tune,
            "n_chains": n_chains,
        },
    )


def _parse_forbidden(forbidden_links, var_names):
    if not forbidden_links:
        return set()
    idx = {name: i for i, name in enumerate(var_names)}
    return {(idx[s], idx[t]) for s, t in forbidden_links if s in idx and t in idx}
