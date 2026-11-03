"""Reading the sph-map maps of step 03 and the bond descriptors of the paper (Section 3.6, Eq. 6):
<d>_AB and <rho_00>_AB are means over five points at 30-70 % of the bond A-B (trilinear interpolation);
D_AB = |<d>_AB| / <rho_00>_AB; the angle is between <d>_AB and the bond vector A->B."""
import gemmi, numpy as np
from config import WORK
CHANNELS = [("L0M+0", "L0"), ("L1M+1", "x"), ("L1M-1", "y"), ("L1M+0", "z")]   # rho_00, d_x, d_y, d_z
def loadmaps(d, stem):
    M = {}
    for lab, key in CHANNELS:
        m = gemmi.read_ccp4_map(f"{d}/{stem}.{lab}.ccp4"); m.setup(0.0); M[key] = m.grid
    return M
def load_set(mol, s, base=None):
    """Maps of set s (obs, iam, iamph, deform) of molecule mol from step 03 (or from another folder 'base')."""
    base = base or WORK / "03_filter"
    return loadmaps(f"{base}/{mol}_{s}_maps", f"{mol}_{s}")
def val(g, f): return g.interpolate_value(gemmi.Fractional(*f))
def vec(M, f): return np.array([val(M[a], f) for a in "xyz"])
def angle(u, v): return float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))
def acute(a): return min(a, 180 - a)
def descriptors(M, atoms, A, bonds, lo=0.3, hi=0.7, n=5):
    out = {}
    for a, b in bonds:
        fa, fb = np.array(atoms[a]), np.array(atoms[b]); bv = (fb - fa) * A
        pts = [fa + t * (fb - fa) for t in np.linspace(lo, hi, n)]
        d = np.mean([vec(M, p) for p in pts], axis=0); r0 = np.mean([val(M["L0"], p) for p in pts])
        out[a + "-" + b] = dict(angle=angle(d, bv), D=float(np.linalg.norm(d) / r0), dnorm=float(np.linalg.norm(d)), rho0=float(r0), d=d.tolist())
    return out
