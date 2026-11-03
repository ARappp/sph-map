#!/bin/sh
# Step 02: prepare the P1 coefficient sets for urea and Gly-Ala (about 1 min).
set -e; cd "$(dirname "$0")"
python3 a_fcf_to_mtz.py
python3 b_make_full_p1.py
python3 c_make_iam.py
