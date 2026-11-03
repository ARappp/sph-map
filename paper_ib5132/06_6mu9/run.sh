#!/bin/sh
# Step 06: 6MU9 (needs only 01_input_data; about 7 min).
set -e; cd "$(dirname "$0")"
for s in a_carbonyl_models b_resolution_sweep c_coordinate_robustness d_difference_peaks e_amide_h f_aromatics g_pdbredo_table_row h_ball_vs_lattice; do
  echo "== $s"; python3 $s.py
done
