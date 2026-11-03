"""Step 07b: observed-model comparison for the five PDB-REDO structures (Supplementary Table 'PDB-REDO'), every value
one Fourier sum at the site (no grid):
  carbonyl angles: <d> of data (FWT/PHWT, P1 half set) vs model (FC_ALL/PHIC_ALL), sphere rule r0 = 0.3, 0.6 A;
  amide-H and Phe/Tyr ring-H control: l=0 value of the H-omit difference (data_noH - model_noH), in units of its r.m.s.
      over the cell (Parseval: rms = sqrt(2 sum_half |C|^2) for the zero-mean map);
  mFo-DFc peaks: DELFWT map (gemmi FFT, sample rate 3), maximum within 1.6 A of O (0.2 A lattice), for the 5 % of sites
      with the largest r0 = 0.6 A angles and for 40 random sites (rng 0).
Also the R values written in the PDB files. Needs a_prepare.py. About 5 min.
Output: WORK/07_pdbredo/pdbredo_sites.json (all site values and a summary per entry)"""
import _paths, json, numpy as np, gemmi
from lattice import ball
from _paths import INPUT, step
W = step("07_pdbredo"); IDS = ["4y44", "3x3r", "3ume", "3la9", "4gyb"]
def load(f):
    m = gemmi.read_mtz_file(f); a = np.array(m, copy=False); L = [c.label for c in m.columns]
    ok = np.isfinite(a[:, L.index('F')]) & np.isfinite(a[:, L.index('PHI')])
    return m.cell, a[ok, :3].astype(float), a[ok, L.index('F')] * np.exp(-1j * np.radians(a[ok, L.index('PHI')]))
def align(Ha, Ca, Hb, Cb):
    """put coefficient set b on the reflection list of a (missing -> 0)"""
    key = lambda H: (H[:, 0].astype(np.int64) * 1000003 + H[:, 1].astype(np.int64)) * 1000003 + H[:, 2].astype(np.int64)
    ka, kb = key(Ha), key(Hb); d = dict(zip(kb.tolist(), range(len(kb)))); out = np.zeros(len(Ha), complex)
    idx = np.array([d.get(k, -1) for k in ka.tolist()]); out[idx >= 0] = Cb[idx[idx >= 0]]
    extra = len(kb) - (idx >= 0).sum(); return out, extra
def sphere(r0, step=0.1):
    return ball(r0, step)            # lattice points with |p| <= r0, surface included (exact integer test)
def W_grid(kc, r0):
    off = sphere(r0); return np.concatenate([np.exp(2j * np.pi * (kc[i:i+3000] @ off.T)).mean(1) for i in range(0, len(kc), 3000)])
def fsum(H, C, Fr, X, M=None, chunk=3000):
    """2 Re sum_h C(h) M(h) exp(2 pi i h.x); M: (n,) or (n,3)"""
    xf = X @ Fr.T; out = np.zeros((len(X),) + (() if M is None or M.ndim == 1 else (M.shape[1],)))
    for i in range(0, len(H), chunk):
        E = np.exp(2j * np.pi * (xf @ H[i:i+chunk].T)); c = C[i:i+chunk] if M is None else (C[i:i+chunk, None] * M[i:i+chunk] if M.ndim == 2 else C[i:i+chunk] * M[i:i+chunk])
        out += 2 * (E @ c).real
    return out
def first(r, name):
    for a in r:
        if a.name == name and a.altloc in ('\0', 'A'): return a
def pos(a): return np.array(a.pos.tolist())
def ang(u, v): return np.degrees(np.arccos(np.clip((u * v).sum(1) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1), -1, 1)))
def analyse(p):
    st = gemmi.read_structure(str(INPUT / 'pdbredo' / f'{p}_final.pdb')); st.setup_entities(); st.remove_hydrogens()
    cell, Hd, Cd = load(f'{W}/{p}/{p}_data_P1.mtz'); _, Hm, Cm0 = load(f'{W}/{p}/{p}_model_P1.mtz'); Cm, ex1 = align(Hd, Cd, Hm, Cm0)
    _, Hdn, Cdn = load(f'{W}/{p}/{p}_data_noH_P1.mtz'); _, Hmn, Cmn0 = load(f'{W}/{p}/{p}_model_noH_P1.mtz'); Cmn, ex2 = align(Hdn, Cdn, Hmn, Cmn0)
    Fr = np.array(cell.frac.mat)
    pol = [r for ch in st[0] for r in ch.get_polymer()]
    carb, amide = [], []
    for i, r in enumerate(pol):
        C, O, N, CA = first(r, 'C'), first(r, 'O'), first(r, 'N'), first(r, 'CA')
        if C and O and N: carb.append((f'{r.name}{r.seqid.num}', pos(O), O.b_iso))
        if i and r.name != 'PRO' and N and CA:
            Cp = first(pol[i - 1], 'C')
            if Cp and Cp.pos.dist(N.pos) < 1.5:
                c, n, ca = pos(Cp), pos(N), pos(CA); b = (n - c) / np.linalg.norm(n - c) + (n - ca) / np.linalg.norm(n - ca); b /= np.linalg.norm(b)
                amide.append(n + 0.95 * b)
    arom = []
    for r in pol:
        if r.name in ('PHE', 'TYR'):
            at = [first(r, a) for a in ('CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2')]
            if any(a is None for a in at): continue
            P = np.array([pos(a) for a in at]); c = P.mean(0)
            for j in ((1, 2, 4, 5, 3) if r.name == 'PHE' else (1, 2, 4, 5)):
                u = P[j] - c; u /= np.linalg.norm(u); arom.append(P[j] + 0.95 * u)
    out = dict(pid=p, n_refl=len(Hd), model_refl_not_in_data=int(ex1), noH_model_refl_not_in_data=int(ex2), angles={})
    kc = Hd @ Fr; kh = kc / np.linalg.norm(kc, axis=1)[:, None]; X = np.array([o for _, o, _ in carb]); labs = [l for l, _, _ in carb]
    for r0 in (0.3, 0.6):
        M = 1j * kh * W_grid(kc, r0)[:, None]
        a = ang(fsum(Hd, Cd, Fr, X, M), fsum(Hd, Cm, Fr, X, M)); out['angles'][str(r0)] = dict(zip(labs, a.round(3).tolist()))
    Dn = Cdn - Cmn; rms = float(np.sqrt(2 * (np.abs(Dn) ** 2).sum()))
    out['amideH'] = (fsum(Hdn, Dn, Fr, np.array(amide)) / rms).round(4).tolist()
    out['aromH'] = (fsum(Hdn, Dn, Fr, np.array(arom)) / rms).round(4).tolist() if arom else []
    # mFo-DFc peaks (as analyse_set.py)
    mt = gemmi.read_mtz_file(str(INPUT / 'pdbredo' / f'{p}_final.mtz')); dg = mt.transform_f_phi_to_map('DELFWT', 'PHDELWT', sample_rate=3); drms = np.array(dg, copy=False).std()
    sh = sphere(1.6, 0.2); Opos = {l: o for l, o, _ in carb}
    peak = lambda x: max(dg.interpolate_value(gemmi.Position(*(x + o))) for o in sh) / drms
    a6 = out['angles']['0.6']; top = sorted(a6, key=lambda k: -a6[k])[:max(5, len(a6) // 20)]
    out['top5pct_peaks'] = {k: float(peak(Opos[k])) for k in top}
    rnd = np.random.default_rng(0).choice(list(a6), size=min(40, len(a6)), replace=False); out['random_peaks'] = [float(peak(Opos[k])) for k in rnd]
    return out
def summ(o):
    a3 = np.array(list(o['angles']['0.3'].values())); a6 = np.array(list(o['angles']['0.6'].values())); h = np.array(o['amideH']); ar = np.array(o['aromH'])
    tp = np.array(list(o['top5pct_peaks'].values())); rp = np.array(o['random_peaks'])
    return dict(n_CO=len(a6), med06=round(float(np.median(a6)), 2), med03=round(float(np.median(a3)), 2), p90_06=round(float(np.percentile(a6, 90)), 2),
                gt15=round(float((a6 > 15).mean() * 100), 1), top5_ge3=round(float((tp >= 3).mean() * 100), 1), rnd_ge3=round(float((rp >= 3).mean() * 100), 1),
                amideH=round(float(h.mean()), 3), amideH_se=round(float(h.std() / np.sqrt(len(h))), 3), amideH_n=len(h), amideH_gt3=round(float((h > 3).mean() * 100), 1),
                amideH_gt2=round(float((h > 2).mean() * 100), 1), aromH=round(float(ar.mean()), 3) if len(ar) else None, aromH_n=len(ar))
def rvalues(p):
    for l in open(INPUT / "pdbredo" / f"{p}_final.pdb"):
        if l.startswith("REMARK   3   R VALUE            (WORKING SET)"): rw = float(l.split(":")[1])
        if l.startswith("REMARK   3   FREE R VALUE      "): rf = float(l.split(":")[1])
    return [rw, rf]
R = {}
for p in IDS:
    o = analyse(p); o["summary"] = summ(o); o["summary"]["R_work_R_free_pdb"] = rvalues(p); R[p] = o
    print(p, o["n_refl"], o["summary"], flush=True)
json.dump(R, open(W / "pdbredo_sites.json", "w"))
