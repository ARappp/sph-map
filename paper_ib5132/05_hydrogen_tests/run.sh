#!/bin/sh
# Step 05: hydrogen tests for urea and Gly-Ala (needs step 02; about 1 min).
set -e; cd "$(dirname "$0")"
for s in a_known_h b_taper_h c_homit_peaks; do echo "== $s"; python3 $s.py; done
