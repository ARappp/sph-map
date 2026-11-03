"""Amide-H test (Supplementary Table 'amide H'): difference dipolar field at the amide N atoms (Pro excluded) of a model
without H against one with H; expected direction: bisector of N<-C(prev) and N<-CA (towards the amide H).
Probes: field at N (exact sum) and averages over spheres r0 = 0.3, 0.6 A from FFT maps (3 points per d_min, trilinear)."""
import numpy as np, gemmi
import direct as D
def amide_N(pdb):
    """H removed; polymer; not PRO; N, CA and previous C (< 1.5 A) with altloc '' or 'A'."""
    st = gemmi.read_structure(str(pdb)); st.setup_entities(); st.remove_hydrogens()
    def first(r, n):
        for a in r:
            if a.name == n and a.altloc in ("\0", "A"): return a
    out = []
    for ch in st[0]:
        pol = ch.get_polymer()
        for i in range(1, len(pol)):
            r = pol[i]; N, CA, Cp = first(r, "N"), first(r, "CA"), first(pol[i - 1], "C")
            if r.name == "PRO" or not (N and CA and Cp) or Cp.pos.dist(N.pos) >= 1.5: continue
            n, c, ca = (np.array(a.pos.tolist()) for a in (N, Cp, CA))
            b = (n - c) / np.linalg.norm(n - c) + (n - ca) / np.linalg.norm(n - ca)
            out.append((f"{r.name}{r.seqid.num}{ch.name}", n, b / np.linalg.norm(b)))
    return out
def vectors(co, X):
    """{probe: (n, 3)} dipolar field at the sites X for the coefficient set co."""
    Ainv = np.linalg.inv(np.array(co.cell.orth.mat)); out = {"at N": D.l1_to_xyz(D.exact(co, X, 1)[:, 1:4])}
    G = D.multipliers(co, 1); shape = D.sphmap_grid_shape(co, 3); mp = [D.fft_map(co, G[:, j], shape) for j in (1, 2, 3)]
    for r0 in (0.3, 0.6):
        off = D.sphere_offsets(r0)
        out[f"sphere r0 {r0:g}"] = np.array([D.l1_to_xyz(np.stack([D.trilinear_np(m, (x + off) @ Ainv.T) for m in mp], 1)).mean(0) for x in X])
    return out
def angles(a, b): return np.degrees(np.arccos(np.clip(np.sum(a * b, 1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1), -1, 1)))
def summary(V, E):
    """V[input][probe]; inputs data_noH, model_noH, data, model. Angle of (data_noH - model_noH) to the expected direction."""
    out = {}
    for pr in V["data"]:
        dw = V["data_noH"][pr] - V["model_noH"][pr]; dh = V["data"][pr] - V["model"][pr]; a = angles(dw, E)
        out[pr] = dict(n=len(a), median_angle_noH=float(np.median(a)), frac_le30=float(np.mean(a <= 30)),
                       median_angle_withH=float(np.median(angles(dh, E))), median_ratio=float(np.median(np.linalg.norm(dw, axis=1) / np.linalg.norm(dh, axis=1))))
    return out
