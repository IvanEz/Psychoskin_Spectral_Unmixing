#!/bin/bash

# Script to run spectral analysis with different configurations

# Define paths
CONFIG_PATH="/mnt/c/Spectral Analysis/spectral-analysis/config/config.ini"
SCRIPT_PATH="/mnt/c/Spectral Analysis/spectral-analysis/src/perform_analysis.py"
DATA_BASE="/mnt/c/Spectral Analysis/spectral-analysis/data"

# Create backup of original config
cp "$CONFIG_PATH" "${CONFIG_PATH}.backup"
echo "Original config backed up to ${CONFIG_PATH}.backup"

# Define parameters to iterate through
PREFIXES=("f378" "f445" "RAM" "REFL1" "REFL2")
SAMPLE_TYPES=("STR" "CTR")
STRATEGIES=("MinMax" "AUC")
UNKNOWN_RANGE=(0 1 2 3 4 5 6)

# Calculate total runs
TOTAL_RUNS=$((${#PREFIXES[@]} * ${#SAMPLE_TYPES[@]} * ${#STRATEGIES[@]} * ${#UNKNOWN_RANGE[@]}))
CURRENT_RUN=0

echo "Starting analysis with $TOTAL_RUNS total configurations..."

# Loop through all combinations
for prefix in "${PREFIXES[@]}"; do
  for type in "${SAMPLE_TYPES[@]}"; do
    # Set the correct data directory based on type
    if [ "$type" == "STR" ]; then
      DATA_DIR="${DATA_BASE}/STR"
    else
      DATA_DIR="${DATA_BASE}/CTR"
    fi
    
    for strategy in "${STRATEGIES[@]}"; do
      for n_unknown in "${UNKNOWN_RANGE[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        
        echo "[$CURRENT_RUN/$TOTAL_RUNS] Running: prefix=${prefix}_${type}, known=${prefix}, strategy=${strategy}, unknown=${n_unknown}"
        
        # Update config file
        cat > "$CONFIG_PATH" << EOF
[Parameters]
data_dir = "${DATA_DIR}"
data_dir_known = "${DATA_BASE}/substances"
prefix = "${prefix}_${type}"
prefix_known = "${prefix}"
strategy = "${strategy}"
n_unknown = ${n_unknown}
EOF

        # Run the analysis
        python "$SCRIPT_PATH"
        
        # Add a short pause between runs
        sleep 1
      done
    done
  done
done

# Restore original config
cp "${CONFIG_PATH}.backup" "$CONFIG_PATH"
echo "Analysis complete! Original config restored."