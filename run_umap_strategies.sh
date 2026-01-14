#!/usr/bin/env bash

# Sweep all predefined modality pairs with all UMAP strategies:
#   - weighted_umap_strategy.py
#   - error_weighted_umap_strategy.py
#   - hybrid_umap_strategy.py
#   - supervised_metric_umap_strategy.py
# across both preprocessing pipelines (MinMax, AUC).
#
# Usage:
#   bash run_umap_strategies.sh
#
# Environment overrides (optional):
#   PYTHON_BIN   - python interpreter (default: python)
#   N_COMPONENTS - NMF components (default: 6)
#   N_NEIGHBORS  - UMAP n_neighbors (default: 8)
#   MIN_DIST     - UMAP min_dist (default: 0.25)
#   RANDOM_STATE - RNG seed (default: 42)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"

# Resolve strategy scripts
WEIGHTED_SCRIPT="${SCRIPT_DIR}/src/weighted_umap_strategy.py"
ERROR_WEIGHTED_SCRIPT="${SCRIPT_DIR}/src/error_weighted_umap_strategy.py"
HYBRID_SCRIPT="${SCRIPT_DIR}/src/hybrid_umap_strategy.py"
SUPERVISED_SCRIPT="${SCRIPT_DIR}/src/supervised_metric_umap_strategy.py"

for f in "$WEIGHTED_SCRIPT" "$ERROR_WEIGHTED_SCRIPT" "$HYBRID_SCRIPT" "$SUPERVISED_SCRIPT"; do
  [[ -f "$f" ]] || { echo "Missing strategy script: $f" >&2; exit 1; }
done

# Preprocessing choices
STRATEGIES=(MinMax AUC)

# Defaults (override via env)
N_COMPONENTS=${N_COMPONENTS:-6}
N_NEIGHBORS=${N_NEIGHBORS:-8}
MIN_DIST=${MIN_DIST:-0.25}
RANDOM_STATE=${RANDOM_STATE:-42}

# Read the predefined modality pairs from Python helper to stay in sync
mapfile -t PAIRS < <(PYTHONPATH="${PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
from strategy_utils import MODALITY_PAIRS
for a,b in MODALITY_PAIRS:
    print(f"{a}+{b}")
PY
) || PAIRS=()

# Fallback to hard-coded default pairs if import fails
if [[ ${#PAIRS[@]} -eq 0 ]]; then
  echo "[WARN] Could not import MODALITY_PAIRS from strategy_utils; using default list." >&2
  PAIRS=(
    "f378+f445" "f378+REFL1" "f378+REFL2" "f378+RAM" \
    "f445+REFL1" "f445+REFL2" "f445+RAM" \
    "REFL1+REFL2" "REFL1+RAM" "REFL2+RAM"
  )
fi

TOTAL_RUNS=$(( ${#STRATEGIES[@]} * 4 * ${#PAIRS[@]} ))
COUNT=0

echo "Running UMAP strategies across ${#PAIRS[@]} pairs, ${#STRATEGIES[@]} preprocessing modes..."

for prep in "${STRATEGIES[@]}"; do
  for pair in "${PAIRS[@]}"; do
    # Weighted UMAP
    COUNT=$((COUNT+1))
    printf '[%3d/%3d] weighted | %s | %s\n' "$COUNT" "$TOTAL_RUNS" "$pair" "$prep"
    "$PYTHON_BIN" "$WEIGHTED_SCRIPT" \
      --base-dir "$SCRIPT_DIR" \
      --pairs "$pair" \
      --preprocessing "$prep" \
      --n-components "$N_COMPONENTS" \
      --n-neighbors "$N_NEIGHBORS" \
      --min-dist "$MIN_DIST" \
      --random-state "$RANDOM_STATE"

    # Error-weighted UMAP
    COUNT=$((COUNT+1))
    printf '[%3d/%3d] error_weighted | %s | %s\n' "$COUNT" "$TOTAL_RUNS" "$pair" "$prep"
    "$PYTHON_BIN" "$ERROR_WEIGHTED_SCRIPT" \
      --base-dir "$SCRIPT_DIR" \
      --pairs "$pair" \
      --preprocessing "$prep" \
      --n-components "$N_COMPONENTS" \
      --n-neighbors "$N_NEIGHBORS" \
      --min-dist "$MIN_DIST" \
      --random-state "$RANDOM_STATE"

    # Hybrid UMAP
    COUNT=$((COUNT+1))
    printf '[%3d/%3d] hybrid | %s | %s\n' "$COUNT" "$TOTAL_RUNS" "$pair" "$prep"
    "$PYTHON_BIN" "$HYBRID_SCRIPT" \
      --base-dir "$SCRIPT_DIR" \
      --pairs "$pair" \
      --preprocessing "$prep" \
      --n-components "$N_COMPONENTS" \
      --n-neighbors "$N_NEIGHBORS" \
      --min-dist "$MIN_DIST" \
      --random-state "$RANDOM_STATE"

    # Supervised metric UMAP (LDA)
    COUNT=$((COUNT+1))
    printf '[%3d/%3d] supervised_metric | %s | %s\n' "$COUNT" "$TOTAL_RUNS" "$pair" "$prep"
    "$PYTHON_BIN" "$SUPERVISED_SCRIPT" \
      --base-dir "$SCRIPT_DIR" \
      --pairs "$pair" \
      --preprocessing "$prep" \
      --n-components "$N_COMPONENTS" \
      --lda-components 1 \
      --n-neighbors "$N_NEIGHBORS" \
      --min-dist "$MIN_DIST" \
      --random-state "$RANDOM_STATE"
  done
done

echo "All UMAP strategy runs complete. Outputs in results/strategies/*"
