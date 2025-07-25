#!/bin/bash
#
# A script to run multiple experiments concurrently on a single GPU.
#

# --- Configuration ---

# An array of YAML configuration files for the experiments.
configs=(
  "config/exp_d4_n2000_wd0.01.yaml"
  "config/exp_d4_n5000_wd0.01.yaml"
  "config/exp_d4_n7000_wd0.01.yaml"
  "config/exp_d8_n2000_wd0.01.yaml"
  "config/exp_d8_n5000_wd0.01.yaml"
  "config/exp_d8_n7000_wd0.01.yaml"
  "config/exp_d12_n2000_wd0.01.yaml"
  "config/exp_d12_n5000_wd0.01.yaml"
  "config/exp_d12_n7000_wd0.01.yaml"
)

# Set the maximum number of concurrent jobs.
# Adjust this based on your GPU's VRAM and the model's memory footprint.
# Start with 2 and increase cautiously.
MAX_JOBS=3

# --- Execution Logic ---

# Clean up background processes if the script is interrupted (e.g., Ctrl+C)
trap 'kill $(jobs -p) 2>/dev/null' EXIT

# Loop through each configuration file
for cfg in "${configs[@]}"; do
  # Wait until there is a free slot to run a new job.
  while [[ $(jobs -r -p | wc -l) -ge $MAX_JOBS ]]; do
    sleep 2 # Wait for a moment before checking again
  done

  echo "🚀 Starting job with config: $cfg"
  # Run the python script in the background
  python src/train.py --config "$cfg" &
done

# Wait for all remaining background jobs to complete
echo "⏳ Waiting for the last batch of jobs to finish..."
wait

echo "✅ All experiments completed successfully."