#!/bin/sh
# Step 08: figures (needs steps 02-06; about 4 min; Fig. 3 needs PyMOL).
set -e; cd "$(dirname "$0")"
python3 a_figure_data.py
python3 b_make_figures.py
