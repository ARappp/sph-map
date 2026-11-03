"""Running sph-map with the settings of the paper (main text Section 3.2)."""
import subprocess
from pathlib import Path
from config import SPHMAP
PAPER = ["-s", "F", "-p", "PHI", "-lmax", "1", "--overwrite", "--phase-convention", "ccp4",
         "--symmetry-policy", "p1_complete_friedel", "--norm", "none", "--taper", "none",
         "--radial-weight-convention", "crystallographic_q", "--shell-weights", "off"]
def run(src, outdir, stem, spd=6, extra=(), amp="F", phase="PHI"):
    """Filter the P1 coefficients in `src`; writes outdir/stem.mtz, .log and outdir/stem_maps/stem.L*M*.ccp4.
    `extra` options are appended (later options override earlier ones). Returns the map folder."""
    outdir = Path(outdir); (outdir / f"{stem}_maps").mkdir(parents=True, exist_ok=True)
    opts = [o if o not in ("F", "PHI") else {"F": amp, "PHI": phase}[o] for o in PAPER]
    cmd = [SPHMAP, "-m", str(Path(src).resolve()), "-o", f"{stem}.mtz", "--map-dir", f"{stem}_maps",
           "--map-samples-per-dmin", str(spd)] + opts + list(extra)
    with open(outdir / f"{stem}.log", "w") as log:
        rc = subprocess.call(cmd, cwd=outdir, stdout=log, stderr=subprocess.STDOUT)
    if rc: raise RuntimeError(f"sph-map failed; see {outdir / (stem + '.log')}")
    return outdir / f"{stem}_maps"
