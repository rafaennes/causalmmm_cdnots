"""Shared background knowledge helpers for causal discovery algorithms.

Background knowledge encodes domain constraints (e.g. "y cannot cause channels")
as forbidden links. Algorithms skip testing these pairs, reducing runtime on
large variable sets without affecting recall on plausible edges.
"""
from __future__ import annotations


def parse_forbidden(forbidden_links, var_names) -> set[tuple[int, int]]:
    """Convert set of (src_name, tgt_name) to set of (src_idx, tgt_idx)."""
    if not forbidden_links:
        return set()
    idx = {name: i for i, name in enumerate(var_names)}
    return {(idx[s], idx[t]) for s, t in forbidden_links if s in idx and t in idx}


def merge_forbidden_into_link_assumptions(
    link_assumptions: dict | None,
    forbidden: set[tuple[int, int]],
    n_vars: int,
    max_lag: int,
) -> dict:
    """Remove forbidden (src, tgt) pairs from tigramite link_assumptions.

    If link_assumptions is None, builds default (all links allowed) first,
    then removes forbidden pairs. This tells PCMCI to skip testing them entirely.
    """
    if not forbidden:
        return link_assumptions if link_assumptions is not None else {}

    # Build default assumptions if none provided
    if link_assumptions is None:
        lags = [-tau for tau in range(1, max_lag + 1)]
        link_assumptions = {}
        for j in range(n_vars):
            link_assumptions[j] = {(i, lag): "-?>" for i in range(n_vars) for lag in lags}

    # Remove forbidden links
    for src, tgt in forbidden:
        if tgt in link_assumptions:
            keys_to_remove = [k for k in link_assumptions[tgt] if k[0] == src]
            for k in keys_to_remove:
                del link_assumptions[tgt][k]

    return link_assumptions
