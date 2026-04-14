# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

Official code for Rafael Ennes' master's dissertation (Mackenzie, 2026) integrating CD-NOTS causal discovery with Bayesian Marketing Mix Modeling. The repo holds two separate but related bodies of code:

1. **`causalmmm/`** — TensorFlow package implementing the CausalMMM (Gong et al., WSDM '24) Variational Graph Autoencoder for MMM. Self-contained library installable via `setup.py`.
2. **`arquivos_recentes/`** — CD-NOTS integration modules (`cdnots_discovery.py`, `cdnots_model_builder.py`, `cdnots_fitter.py`, `presets.py`) designed to be dropped into the sibling repo `/home/ennes/mestrado/pymc_meridian_comparison/mmm_param_recovery/benchmarking/`. They use relative imports (`from . import model_builder`) and only function when loaded inside that namespace.

The 4-arm experiment compares: (1) PyMC-Marketing baseline, (2) Meridian baseline, (3) PyMC + CD-NOTS priors, (4) Meridian + CD-NOTS priors. Notebook: [notebooks/experimento_4bracos.ipynb](notebooks/experimento_4bracos.ipynb).

## Install / run

```bash
pip install -e .                    # install causalmmm package
pip install -e ".[cdnots,viz,dev]"  # with optional extras
python causalmmm/examples/basic_usage.py
```

The `causalmmm` package needs TensorFlow. CD-NOTS modules additionally need `causal-learn`, `pymc-marketing`, `meridian`, `tensorflow-probability`. There is no test suite, linter config, or CI in this repo — `pytest`, `black`, `flake8`, `mypy` are declared as dev extras but no configuration or tests exist yet.

## Architecture notes

- **`causalmmm/models/causalmmm.py`** is the main model (Graph VAE with GRU/LSTM temporal layer, saturation + adstock). `encoder.py` / `decoder.py` are its components; `hybrid.py` exists but `HybridCausalMMM` is stubbed out in `__init__.py`. Many items in the top-level API are `TODO`-commented — check [causalmmm/__init__.py](causalmmm/__init__.py) before assuming an export exists.
- **`arquivos_recentes/cdnots_discovery.py`** produces a `CausalGraph` dataclass via PC/PCMCI over channel + control variables. Uses `parcorr` unconditionally (100-1000× faster than rcot); samples ≤5 geos; flags `endogenous_channels` (channels confounded by controls).
- **`arquivos_recentes/cdnots_model_builder.py`** applies "humble priors": `sigma_adj = sigma_base × (1 + damping × direction × confidence)` where `confidence = 1 - p_value` and `direction = ±1` from graph. V2 replaced fixed multipliers (0.1/0.8/1.2) that caused MCMC R-hat > 1.8.
- **Relative-import gotcha**: `cdnots_model_builder.py` / `cdnots_fitter.py` do `from . import model_builder`. To use them outside the benchmark repo, load via `importlib.util.spec_from_file_location` and register in `sys.modules` under `mmm_param_recovery.benchmarking.{name}`.
- **`CDNOTS_INTEGRATION.py`** is not executable code — it is a documentation file containing string snippets (`ARGPARSE_ADDITION`, `PHASE1_ADDITION`, etc.) meant to be hand-pasted into `pymc_meridian_comparison/run_benchmark.py`.

## Default presets

`small_business` (1 geo, 4 channels, 104 weeks) for fast iteration; `medium_business` (8 geos, 8 channels) for full benchmark runs. Invoke via `get_preset_config("small_business")` in [arquivos_recentes/presets.py](arquivos_recentes/presets.py) (note: this file lives in the sibling-repo-shaped `arquivos_recentes/` and imports from `.config`).

## Packaging for cloud runs

[pack_for_gcp.sh](pack_for_gcp.sh) tars both `causalmmm_with_cdnots/` and the sibling `pymc_meridian_comparison/` (excluding `.pixi`, `.git`, `__pycache__`, `resultados/`) for GCS upload. Accepts an optional `gs://bucket/path/` argument.
