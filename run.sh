#!/bin/bash

# Fail on error
set -e

# Print what is happening
echo "Running YMSF Problem 2 Simulation..."

# Activate python environment if needed (optional)
# source venv/bin/activate

# Run simulation
python run_sim.py --entry 2.0 --tp_off 0.0 --stop_off 4.0 --data_folder customdata_new --result_folder results

# Combine into final results file
python combine_results.py

echo "Done. Results generated."
