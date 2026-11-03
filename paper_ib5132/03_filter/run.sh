#!/bin/sh
# Step 03: sph-map filtering of the small-molecule sets (needs step 02; about 1 min).
set -e; cd "$(dirname "$0")"
python3 run_filter.py
