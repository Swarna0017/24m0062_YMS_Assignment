# 24m0062_YMS_Assignment
Contains the solution files for the problem statements.
Overview

This repository contains the full implementation of my solution for YMSF Quant Research – Problem 2.
The project builds a complete pipeline for generating tradable spreads using a mean-reversion framework on minute-level futures data (FUT1, FUT2) and the cash market (CM).

The code replicates the workflow of the assignment:

Load and preprocess minute-level CM/FUT1/FUT2 data

Construct normalized spreads per underlying

Compute rolling winsorized statistics

Run a per-underlying mean-reversion trading simulation

Generate trade logs, per-minute MTM, and final summary statistics

Export consolidated results in the required Problem2_results.csv format

All components are reproducible end-to-end via the provided run.sh script.

/
├── run_sim.py              # Main simulation engine
├── combine_results.py      # Post-processing utility
├── run.sh                  # Script to regenerate the full results
├── Problem2_readme.pdf     # One-page explanation of strategy & assumptions
├── make_results.py         # Converter for summary → Problem2_results.csv

Important Note — Data Not Included

The raw dataset (CM/FUT files, ~140+ CSVs) is not included in this repository due to size constraints and upload limits.

To run the full simulation, download the original dataset from the official assignment link:
customdata_new.tar.gz
./customdata_new/

How to Run the Simulation
Linux / macOS
chmod +x run.sh
./run.sh

Windows (PowerShell)
bash run.sh

This will:

Load and preprocess all underlyings

Run the mean-reversion engine

Write output files under results/

Produce results.<timestamp>.csv or the equivalent summary files

(Optional) make_results.py can be used to convert summary/trades into the final Problem2_results.csv

Key Features of the Implementation

Robust timestamp rebuilding from file names

Spread construction using normalized FUT2–FUT1 mid-price difference

Rolling 60-day winsorized SMA and SD for z-score computation

Per-underlying parallel simulation (ProcessPoolExecutor)

Expiry handling and forced position closure

Execution throttled by spreadable volume (5/15/30-min windows)

Slippage and cost modeling via half-spread and cost factors

Full trade logs and per-minute MTM stored as CSVs

Output Files

After running the simulation, the following files will be created:

mr_spread_rel_summary_<params>.csv

mr_spread_rel_trades_<params>.csv

mr_spread_rel_perminute_<params>.csv

Problem2_results.csv (final consolidated file for submission)

Reproducibility

The repository includes:


All code required to regenerate the assignment outputs

A run script (run.sh) following YMSF’s requirement that the simulation be reproducible from a single command

The only missing component is the raw dataset, which must be downloaded separately.

Contact

If you have questions regarding assumptions, structure, or reproducibility, please refer to:

Problem2_readme.pdf
or contact me via the email listed in the assignment submission.

