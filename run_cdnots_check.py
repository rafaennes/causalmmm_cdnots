"""Verifica CD-NOTS no causal_business (1 geo, KCI)."""
import sys, os, warnings, pickle
warnings.filterwarnings("ignore")

PYMC_COMPARISON_ROOT = "/home/ennes/mestrado/pymc_meridian_comparison"
CAUSALMMM_ROOT       = "/home/ennes/mestrado/causalmmm_with_cdnots"
ARQUIVOS_RECENTES    = os.path.join(CAUSALMMM_ROOT, "arquivos_recentes")
RESULTS_DIR          = os.path.join(CAUSALMMM_ROOT, "notebooks", "resultados", "causal_business")
os.makedirs(RESULTS_DIR, exist_ok=True)

for p in [PYMC_COMPARISON_ROOT, CAUSALMMM_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np, importlib.util

from mmm_param_recovery.data_generator.core import generate_mmm_dataset
from mmm_param_recovery.data_generator.presets import get_preset_config
from mmm_param_recovery.benchmarking.data_loader import prepare_dataset_for_modeling

_s = importlib.util.spec_from_file_location("causal_evaluation",
        os.path.join(ARQUIVOS_RECENTES, "causal_evaluation.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
evaluate_causal_structure = _m.evaluate_causal_structure

def _load(name, filepath):
    full = f"mmm_param_recovery.benchmarking.{name}"
    spec = importlib.util.spec_from_file_location(full, filepath)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod; spec.loader.exec_module(mod)
    return mod

cdnots_discovery = _load("cdnots_discovery",
    os.path.join(ARQUIVOS_RECENTES, "cdnots_discovery.py"))

print("=== Gerando dados (causal_business — 1 geo nacional) ===", flush=True)
config  = get_preset_config("causal_business", seed=20250715)
result  = generate_mmm_dataset(config)
data_df, channel_columns, control_columns, _ = prepare_dataset_for_modeling(result)
true_adj = result["ground_truth"]["causal_graph"]["adjacency_matrix"]

print(f"shape: {data_df.shape}  |  geos: {config.regions.region_names}", flush=True)
print(f"true edges: {int(true_adj.sum())}\n", flush=True)

print("=== CD-NOTS (KCI, alpha=0.05) ===", flush=True)
causal_graph = cdnots_discovery.discover_graph(
    data_df, channel_columns,
    control_columns=control_columns,
    alpha=0.05, max_lag=2,
)

print(f"\nci_test_used:      {causal_graph.ci_test_used}")
print(f"n_edges:           {causal_graph.n_edges}")
print(f"direct:            {causal_graph.direct_channels}")
print(f"mediated:          {causal_graph.mediated_channels}")
print(f"excluded:          {causal_graph.excluded_channels}")
print(f"endogenous:        {causal_graph.endogenous_channels}\n")

var_names   = list(causal_graph.variable_names)
keep_idx    = [var_names.index(c) for c in channel_columns] + [var_names.index("y")]
learned_sub = causal_graph.adjacency_matrix[np.ix_(keep_idx, keep_idx)]

m = evaluate_causal_structure(learned_graph=learned_sub, true_graph=true_adj)
print("=== Métricas Estruturais ===")
print(f"  {'Métrica':12s}  {'Valor':>8}  {'Baseline':>10}")
baseline = {"precision":0.267,"recall":0.444,"f1":0.333,"fdr":0.733,"shd":16,"tp":4,"fp":11,"fn":5}
for k, v in m.items():
    b = baseline.get(k, "—")
    vf = f"{v:.3f}" if isinstance(v, float) else str(v)
    bf = f"{b:.3f}" if isinstance(b, float) else str(b)
    arrow = " ↑" if isinstance(v,float) and isinstance(b,float) and k in ("precision","recall","f1") and v>b else \
            " ↓" if isinstance(v,float) and isinstance(b,float) and k in ("fdr","shd","fp","fn") and v<b else ""
    print(f"  {k:12s}  {vf:>8}  {bf:>10}{arrow}")

with open(os.path.join(RESULTS_DIR, "cdnots_graph.pkl"), "wb") as f:
    pickle.dump(causal_graph, f)
print(f"\nGrafo salvo: {RESULTS_DIR}/cdnots_graph.pkl")
