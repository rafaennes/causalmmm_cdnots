#!/usr/bin/env bash
# pack_for_gcp.sh — creates a single archive ready to upload to GCS
# Usage: bash pack_for_gcp.sh [gcs-bucket]
#   e.g. bash pack_for_gcp.sh gs://my-bucket/mmm-experiment/
set -euo pipefail

CAUSALMMM="/home/ennes/mestrado/causalmmm_with_cdnots"
PYMC_COMP="/home/ennes/mestrado/pymc_meridian_comparison"
ARCHIVE="/tmp/mmm_experiment_$(date +%Y%m%d).tar.gz"

echo "Packing code (excluding pixi env, git history, results)..."

tar -czf "$ARCHIVE" \
  --exclude='*/.pixi' \
  --exclude='*/.git' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='*/resultados' \
  --exclude='*/node_modules' \
  -C /home/ennes/mestrado \
  causalmmm_with_cdnots \
  pymc_meridian_comparison

SIZE=$(du -sh "$ARCHIVE" | cut -f1)
echo "Archive: $ARCHIVE ($SIZE)"

if [ "${1:-}" != "" ]; then
  echo "Uploading to $1 ..."
  gcloud storage cp "$ARCHIVE" "$1"
  echo "Done: $1$(basename $ARCHIVE)"
else
  echo "To upload: gcloud storage cp $ARCHIVE gs://YOUR-BUCKET/path/"
fi
