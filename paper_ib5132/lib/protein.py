"""Exact (grid-free) site values for the protein structures (main text Section 3.6, Eqs 7 and 9).
Coefficients C(h) = |F| exp(-i phi) of the P1 half set (Friedel mates implicit: field = 2 Re sum_h ...).
Dipolar field at x: d(x) = 2 Re sum_h C(h) i R_1(k^) W(h) exp(2 pi i h.x), R_1 = sqrt(3/4pi) k^ (x, y, z).
W = 1 for the field at the atom ('point'); for the sphere average <d> (radius r0), W is the mean of exp(2 pi i k.p) over
the 0.1 A cubic-lattice points p with |p| <= r0 (the sphere rule of the paper; points on the surface included), so <d> is one Fourier sum per site."""
import numpy as np, gemmi
from config import INPUT
D6 = INPUT / "6mu9"
REFMAC = ("FWT", "PHWT", "FC_ALL", "PHIC_ALL")                                       # deposited 6MU9 (REFMAC)
PHENIX = ("2FOFCWT_no_fill", "PH2FOFCWT_no_fill", "F-model", "PHIF-model")           # re-refined models (a)-(c)
MODELS = {"deposited": ("6mu9_final.mtz", REFMAC, "6MU9_deposited.pdb", ("DELFWT", "PHDELWT")),
          "a_noH": ("rerefined_a_noH.mtz", PHENIX, "rerefined_a_noH.pdb", ("FOFCWT", "PHFOFCWT")),
          "b_ridingH": ("rerefined_b_ridingH.mtz", PHENIX, "rerefined_b_ridingH.pdb", ("FOFCWT", "PHFOFCWT")),
          "c_plusH_O299": ("rerefined_c_ridingH_plus_H_O299.mtz", PHENIX, "rerefined_c_ridingH_plus_H_O299.pdb", ("FOFCWT", "PHFOFCWT"))}
def load(mtz, fd, pd_, fm, pm):
    """Observed (fd, pd_) and model (fm, pm) coefficients; returns H, Cd, Cm, Fr (fractionalisation matrix), kc (cycles/A)."""
    m = gemmi.read_mtz_file(str(mtz)); a = np.array(m, copy=False); L = [c.label for c in m.columns]
    ok = np.isfinite(a[:, L.index(fd)]) & np.isfinite(a[:, L.index(fm)])
    H = a[ok, :3].astype(float)
    Cd = a[ok, L.index(fd)] * np.exp(-1j * np.radians(a[ok, L.index(pd_)]))
    Cm = a[ok, L.index(fm)] * np.exp(-1j * np.radians(a[ok, L.index(pm)]))
    Fr = np.array(m.cell.frac.mat); return H, Cd, Cm, Fr, H @ Fr
def sphere_offsets(r0, step=0.1):
    """0.1 A cubic-lattice points with |p| <= r0, surface points included (exact integer test, lattice.ball)."""
    from lattice import ball
    return ball(r0, step)
def mult(kc, r0=None):
    """Multipliers i R_1(k^) W(h) (n, 3); r0 None: field at the atom. Also returns |k|."""
    k = np.linalg.norm(kc, axis=1); kh = kc / k[:, None]
    if r0 is None: W = np.ones(len(k))
    else:
        off = sphere_offsets(r0); W = np.concatenate([np.exp(2j * np.pi * (kc[i:i + 4000] @ off.T)).mean(1) for i in range(0, len(kc), 4000)])
    return 1j * np.sqrt(3 / (4 * np.pi)) * kh * W[:, None], k
def fields(H, C, Mult, Fr, X, chunk=3000):
    """{name: (n_sites, 3)} for every multiplier in Mult, at Cartesian sites X."""
    xf = X @ Fr.T; out = {f: np.zeros((len(X), 3)) for f in Mult}
    for i in range(0, len(H), chunk):
        EC = np.exp(2j * np.pi * (xf @ H[i:i + chunk].T)) * C[None, i:i + chunk]
        for f, M in Mult.items(): out[f] += 2 * (EC @ M[i:i + chunk]).real
    return out
def ang(u, v): return np.degrees(np.arccos(np.clip((u * v).sum(1) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1), -1, 1)))
def rank(a, i): return int((a > a[i]).sum()) + 1
def taper(kind, k, kmax):
    """Resolution cut-off at kmax with no taper, a raised cosine over the outer 10 % (eta 0.9) or a Kaiser window (beta 8)."""
    if kind == "none": return (k <= kmax).astype(float)
    if kind == "cosine":
        t = 0.9 * kmax; v = np.ones_like(k); s = (k > t) & (k <= kmax)
        v[s] = 0.5 * (1 + np.cos(np.pi * (k[s] - t) / (kmax - t))); v[k > kmax] = 0; return v
    arg = np.clip(1 - (k / kmax) ** 2, 0, 1); v = np.i0(8.0 * np.sqrt(arg)) / np.i0(8.0); v[k > kmax] = 0; return v
def carbonyl_sites(pdb):
    """Paper rule: non-HETATM residues having atoms O and C; first O and first C. Returns names, O (n,3), C (n,3)."""
    st = gemmi.read_structure(str(pdb)); out = []
    for ch in st[0]:
        for res in ch:
            if res.het_flag == "H": continue
            O = [a for a in res if a.name == "O"]; C = [a for a in res if a.name == "C"]
            if O and C: out.append((f"{ch.name}{res.seqid.num}{res.name}", np.array(O[0].pos.tolist()), np.array(C[0].pos.tolist())))
    return [n for n, _, _ in out], np.array([o for _, o, _ in out]), np.array([c for _, _, c in out])
