"""Hydrogen tests for urea and Gly-Ala (main text Section 3.7; Supplementary Tables 'Known-H test' and
'Model-phase control'). Sites: non-H atoms bonded to H (<= 1.2 A); e = sum of the unit X->H vectors.
Coefficient sets: data = F_obs with HAR phases; model_I = IAM; model_noH = IAM - F_H (F_H: analytic transform of the
H atoms, on the IAM scale); data_omit (P3) = |F_obs| with the phases of model_noH. F(000) is included.
Probes: the dipolar field at the atom and the field averaged over spheres of radius r0 = 0.3 and 0.6 A."""
import numpy as np
from config import WORK
import direct as D, structure as S, molecules
FILES = {"urea": ("urea_fullP1.mtz", "urea_iam_fullP1.mtz"), "glyala": ("glyala_fullP1.mtz", "glyala_iam_fullP1.mtz")}
def ang(a, b): return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))
def exactA(co, X):
    """Exact direct sums: field at the atoms and sphere averages (cubic lattice of 0.1 A points, |p| <= r0)."""
    out = {"A at atom": D.l1_to_xyz(D.exact(co, X, 1)[:, 1:4])}
    for r0 in (0.3, 0.6):
        off = D.sphere_offsets(r0); out[f"A sphere r0 {r0:g}"] = np.array([D.l1_to_xyz(D.exact(co, x + off, 1)[:, 1:4]).mean(0) for x in X])
    return out
def fftA(co, X, Ainv):
    """As exactA, but the sphere averages from FFT maps (3 points per d_min) with trilinear interpolation (controls)."""
    out = {"A at atom": D.l1_to_xyz(D.exact(co, X, 1)[:, 1:4])}
    G = D.multipliers(co, 1); shape = D.sphmap_grid_shape(co, 3); mp = [D.fft_map(co, G[:, j], shape) for j in (1, 2, 3)]
    for r0 in (0.3, 0.6):
        off = D.sphere_offsets(r0)
        out[f"A sphere r0 {r0:g}"] = np.array([D.l1_to_xyz(np.stack([D.trilinear_np(m, (x + off) @ Ainv.T) for m in mp], 1)).mean(0) for x in X])
    return out
def context(ds):
    fH, fI = FILES[ds]; IN = WORK / "02_prepare"
    cr = S.crystal_from_cif(molecules.CIF[ds]); isH = np.array([a.element == "H" for a in cr.atoms])
    coI = D.load(IN / fI, "F", "PHI"); coD = D.load(IN / fH, "F", "PHI")
    Fall = S.sf(cr, range(len(cr.atoms)), coI.q); FHy = S.sf(cr, np.nonzero(isH)[0], coI.q)
    k = np.vdot(Fall, coI.C) / np.vdot(Fall, Fall)
    scale = dict(scale=float(k.real), R_after=float(np.abs(coI.C - k * Fall).sum() / np.abs(coI.C).sum()),
                 FH_frac=float(np.abs(FHy).sum() / np.abs(Fall).sum()), nH_e=float(sum(cr.free[i].electrons for i in np.nonzero(isH)[0])))
    FHy = k.real * FHy; CD = D.align(coI, coD); CnoH = coI.C - FHy
    parents = []
    for i, a in enumerate(cr.atoms):
        if isH[i] or not a.label.endswith("#1"): continue
        df = (cr.cart[isH] - a.cart) @ cr.Ainv.T; df -= np.round(df); v = df @ cr.A.T; d = np.linalg.norm(v, axis=1); b = v[d <= 1.2]
        if len(b): parents.append((a.label[:-2], a.cart, (b / np.linalg.norm(b, axis=1)[:, None]).sum(0), len(b)))
    return dict(coI=coI, CD=CD, CnoH=CnoH, FHy=FHy, phnoH=np.exp(1j * np.angle(CnoH)), F0=molecules.F000[ds], nH_e=scale["nH_e"], scale=scale,
                X=np.array([p[1] for p in parents]), E=[p[2] for p in parents], labels=[p[0] for p in parents], nH=[p[3] for p in parents], Ainv=cr.Ainv, cr=cr)
