"""Paths for all steps of the IB5132 pipeline. No other file contains a path.

INPUT : 01_input_data of this record (read only).
WORK  : where every step writes its results, one sub-folder per step
        (default: ./work next to this file; override with the environment variable IB5132_WORK).
SPHMAP: sph-map executable (default: 'sph-map' on PATH; override with IB5132_SPHMAP).
"""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "01_input_data"
WORK = Path(os.environ.get("IB5132_WORK", ROOT / "work"))
SPHMAP = os.environ.get("IB5132_SPHMAP", "sph-map")

def step(name):
    """Output folder of a step, e.g. step('02_prepare'); created if missing."""
    d = WORK / name
    d.mkdir(parents=True, exist_ok=True)
    return d
