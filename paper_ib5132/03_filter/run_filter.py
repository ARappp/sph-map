"""Step 03: spherical-harmonic filtering (l <= 1) of the urea and Gly-Ala coefficient sets with sph-map,
using the settings of the paper (Section 3.2). Reads WORK/02_prepare, writes WORK/03_filter:
  <mol>_<set>.mtz (+ .manifest.json, .log) and <mol>_<set>_maps/<mol>_<set>.L<l>M<m>.ccp4
with <mol> = urea, glyala and <set> = obs (HAR data), iam, iamph (|F_obs| with IAM phases), deform (F_obs - F_IAM).
Map channels: L0M+0 = rho_00; L1M+1, L1M-1, L1M+0 = d_x, d_y, d_z (real tesseral components)."""
import _paths, subprocess, sys
from _paths import step
from config import SPHMAP
IN, OUT = step("02_prepare"), step("03_filter")
OPTS = ["-s", "F", "-p", "PHI", "-lmax", "1", "--overwrite", "--phase-convention", "ccp4",
        "--symmetry-policy", "p1_complete_friedel", "--norm", "none", "--taper", "none",
        "--radial-weight-convention", "crystallographic_q", "--shell-weights", "off",
        "--map-samples-per-dmin", "6"]
for mol in ("urea", "glyala"):
    for s, src in (("obs", ""), ("iam", "_iam"), ("iamph", "_iamph"), ("deform", "_deform")):
        name = f"{mol}_{s}"
        (OUT / f"{name}_maps").mkdir(exist_ok=True)
        cmd = [SPHMAP, "-m", str(IN / f"{mol}{src}_fullP1.mtz"), "-o", f"{name}.mtz", "--map-dir", f"{name}_maps"] + OPTS
        with open(OUT / f"{name}.log", "w") as log:
            rc = subprocess.call(cmd, cwd=OUT, stdout=log, stderr=subprocess.STDOUT)
        print(f"{name:14s} rc={rc}")
        if rc: sys.exit(f"sph-map failed for {name}; see {OUT / (name + '.log')}")
