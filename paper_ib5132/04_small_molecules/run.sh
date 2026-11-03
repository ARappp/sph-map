#!/bin/sh
# Step 04: small-molecule results (needs steps 02 and 03; about 5 min, runs sph-map 42 times).
set -e; cd "$(dirname "$0")"
for s in a_numerical_checks b_bond_descriptors c_window_sensitivity d_profiles e_incompleteness f_radial_weight g_grid_vs_exact h_taper; do
  echo "== $s"; python3 $s.py
done
