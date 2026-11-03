#!/bin/sh
# Step 07: PDB-REDO resolution series (needs only 01_input_data; about 6 min).
set -e; cd "$(dirname "$0")"
for s in a_prepare b_sites_exact c_amide_h; do echo "== $s"; python3 $s.py; done
