#!/bin/bash

# Run multimodal UMAP visualisations across all modality combinations,
# preprocessing strategies, and fusion modes. Results (plots, embeddings,
# metrics) are written into the organised multimodal results tree.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
UMAP_SCRIPT="${SCRIPT_DIR}/src/umap_visualize_multimodal.py"

if [[ ! -f "${UMAP_SCRIPT}" ]]; then
  echo "Unable to locate ${UMAP_SCRIPT}" >&2
  exit 1
fi

# Modalities defined in the Python helper; keep in sync when updating defaults.
MODALITIES=(f378 f445 REFL1 REFL2 RAM)
FUSION_MODES=(concat concat_stats)
STRATEGIES=(MinMax AUC)

RESULTS_DIR="${SCRIPT_DIR}/results"
METRICS_RELATIVE="multimodal/multimodal_separation_summary.csv"
METRICS_PATH="${RESULTS_DIR}/${METRICS_RELATIVE}"

mkdir -p "$(dirname "${METRICS_PATH}")"
rm -f "${METRICS_PATH}"

mapfile -t MODALITY_COMBOS < <("${PYTHON_BIN}" - <<'PY'
import itertools
modalities = ["f378", "f445", "REFL1", "REFL2", "RAM"]
for r in range(1, len(modalities) + 1):
    for combo in itertools.combinations(modalities, r):
        print(" ".join(combo))
PY
)

TOTAL_COMBOS=${#MODALITY_COMBOS[@]}
TOTAL_RUNS=$((TOTAL_COMBOS * ${#FUSION_MODES[@]} * ${#STRATEGIES[@]}))
CURRENT=0

echo "Starting multimodal UMAP sweep with ${TOTAL_RUNS} runs..."

for combo in "${MODALITY_COMBOS[@]}"; do
  read -r -a combo_array <<< "${combo}"
  for strategy in "${STRATEGIES[@]}"; do
    for fusion in "${FUSION_MODES[@]}"; do
      CURRENT=$((CURRENT + 1))
      printf '[%3d/%3d] Modalities=%s | Strategy=%s | Fusion=%s\n' \
        "${CURRENT}" "${TOTAL_RUNS}" "${combo}" "${strategy}" "${fusion}"

      "${PYTHON_BIN}" "${UMAP_SCRIPT}" \
        --base-dir "${SCRIPT_DIR}" \
        --modalities "${combo_array[@]}" \
        --strategy "${strategy}" \
        --fusion-mode "${fusion}" \
        --output-dir "results" \
        --metrics-csv "${METRICS_RELATIVE}"
    done
  done
done

echo "Multimodal UMAP sweep complete."
echo "Metrics summary: ${METRICS_PATH}"
