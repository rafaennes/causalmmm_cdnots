#!/usr/bin/env bash
# Ladder experiment (escada) — mirrors RELATORIO_ESCADA.md config.
# Usage: tmux new -s escada './run_escada.sh 2>&1 | tee new_pipe_dag/results/escada/run.log'
set -euo pipefail
cd "$(dirname "$0")"

.venv/bin/python -m new_pipe_dag \
    --ladder all \
    --algorithms granger dynotears lpcmci cedar cdnots regime_pcmci rpcmci \
    --seeds 2025 \
    --alpha 0.05 \
    --max-lag 3 \
    --output-dir new_pipe_dag/results/escada \
    --background-knowledge
