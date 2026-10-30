#!/usr/bin/env bash
# Runs every test, then every experiment, writing results and figures to results/.
set -euo pipefail
cd "$(dirname "$0")/code"
SIM=../data/sim.npz; OUT=../results

echo "== tests"
python3 test_sensor.py  $SIM
python3 test_harness.py $SIM
python3 test_kalman.py  $SIM
python3 test_ident.py

echo "== Piece 2: scenario report"
python3 scenario_report.py $SIM $OUT

echo "== Piece 3: baselines"
python3 run_baselines.py $SIM $OUT
python3 plot_baselines.py $OUT/baseline_results.json $OUT/fig_baselines.png

echo "== Piece 4: Kalman filter, smoother, regularised inversion"
python3 run_kalman.py $SIM $OUT
python3 plot_kalman.py $OUT/kalman_results.json $OUT/fig_kalman.png

echo "== Piece 5: identifiability and fingerstick tuning"
python3 run_ident.py  $SIM $OUT
python3 run_fstune.py $SIM $OUT
python3 plot_ident.py $OUT/ident_results.json $OUT/fstune_results.json $OUT/fig_ident.png
echo "done"
