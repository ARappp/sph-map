"""Difference density at omitted H atoms, bond midpoints and aromatic rings of 6MU9 (main text Fig. 5 and Section 4.4;
Supplementary Section 'omitted hydrogen atoms and aromatic rings' and Fig. S6), deposited coordinates.
Difference coefficients C = FWT - FC_ALL; with uniform weights the l=0 channel is the difference map / sqrt(4 pi) and the
l=1 channels are the dipolar field. Class values (expected aromatic C-H and amide H positions, ring-bond midpoints, ring
centres, +-0.7 A above/below ring atoms, backbone C-N and C=O midpoints) are exact Fourier sums at each position in units
of the r.m.s. over the cell; standard errors by bootstrap over whole rings or residues (blocks of 5 for the backbone).
Sections through Phe85, Tyr210 and Trp227 (Fig. S6, panels A-F) are sampled from FFT maps by trilinear interpolation.
About 2 min. Output (WORK/06_6mu9): aromatics_stats.json, aromatics_cluster_se.json, aromatics_sections.npz"""
import _paths, json, numpy as np, gemmi
from scipy.ndimage import map_coordinates
from _paths import step
import protein as PR
OUT = step("06_6mu9"); mtz = str(PR.D6 / "6mu9_final.mtz"); pdb = str(PR.D6 / "6MU9_deposited.pdb")
m = gemmi.read_mtz_file(mtz); a = np.array(m, copy=False); L = [c.label for c in m.columns]
ok = np.isfinite(a[:, L.index('FWT')]) & np.isfinite(a[:, L.index('FC_ALL')]); H = a[ok, :3].astype(int)
Cd = a[ok, L.index('FWT')] * np.exp(-1j * np.radians(a[ok, L.index('PHWT')]))
Cm = a[ok, L.index('FC_ALL')] * np.exp(-1j * np.radians(a[ok, L.index('PHIC_ALL')]))
cell = m.cell; Fr = np.array(cell.frac.mat); kc = H @ Fr; kh = kc / np.linalg.norm(kc, axis=1)[:, None]
N = [int(np.ceil(abs(H[:, i]).max() * 2 * 2.2 / 2) * 2) + 2 for i in range(3)]   # ~4.4 points per d_min
N = [gemmi.good_grid_size([n], False, gemmi.SpaceGroup('P 1'))[0] if hasattr(gemmi, 'good_grid_size') else n for n in N]
def synth(C):
    G = np.zeros(N, complex); idx = tuple((H % N).T); G[idx] += C
    idx2 = tuple((-H % N).T); G[idx2] += np.conj(C)            # Friedel mates (P1 list holds one of each pair)
    return np.real(np.fft.ifftn(G)) * np.prod(N)
D = Cd - Cm
maps = {'dL0': synth(D), 'oL0': synth(Cd)}
for j, comp in enumerate('xyz'): maps['dL1' + comp] = synth(D * 1j * kh[:, j])
rms = float(maps['dL0'].std()); print('grid', N, 'rms(diff L0)', rms)
def sample(arr, X):   # X: (..., 3) cartesian
    f = (X.reshape(-1, 3) @ Fr.T) * np.array(N)
    return map_coordinates(arr, f.T, order=1, mode='grid-wrap').reshape(X.shape[:-1])
st = gemmi.read_structure(pdb); st.setup_entities()
A = {}
for ch in st[0]:
    for r in ch:
        if r.name == 'HOH': continue
        for at in r:
            if at.altloc not in ('\0', 'A'): continue
            A.setdefault((r.seqid.num, at.name), np.array(at.pos.tolist()))
# panel-G values: exact Fourier sums at each position (no grid); the r.m.s. over the cell is that of the FFT map
# (Parseval-exact on a grid finer than the Nyquist limit). val() returns a lazy index; values are filled in later.
PTS = []; GRP = []; CUR = [None]
class Lazy(float):
    pass
def val(p):
    PTS.append(np.array(p, float)); GRP.append(CUR[0]); return len(PTS) - 1
def exact_L0(X, chunk=4000):
    xf = X @ Fr.T; out = np.zeros(len(X))
    for i in range(0, len(H), chunk):
        out += 2 * (np.exp(2j * np.pi * (xf @ H[i:i+chunk].T)) @ D[i:i+chunk]).real
    return out
# ---- expected H positions (0.95 A, as in the paper scripts) ----
RINGH = {'PHE': ['CD1', 'CE1', 'CZ', 'CE2', 'CD2'], 'TYR': ['CD1', 'CE1', 'CE2', 'CD2'], 'TRP': ['CD1', 'CE3', 'CZ3', 'CH2', 'CZ2'], 'HIS': ['CE1', 'CD2']}
def ring_H(r, names):
    out = []
    for nm in names:
        Aa = r.find_atom(nm, '*')
        if Aa is None: continue
        nb = [b for b in r if b.name != nm and b.element.name in 'CNO' and 1.2 < b.pos.dist(Aa.pos) < 1.6]
        if len(nb) != 2: continue
        p = np.array(Aa.pos.tolist()); v = sum((p - np.array(b.pos.tolist())) / b.pos.dist(Aa.pos) for b in nb); v /= np.linalg.norm(v)
        out.append((nm, p + 0.95 * v))
    return out
aromH = []; Hexp = {}
for ch in st[0]:
    for r in ch:
        CUR[0] = ('res', ch.name, r.seqid.num)
        if r.name in RINGH:
            hs = ring_H(r, RINGH[r.name]); aromH += [val(p) for _, p in hs]
            Hexp[r.seqid.num] = hs + (ring_H(r, ['NE1']) if r.name == 'TRP' else [])   # indole N-H drawn in the figure only
RINGS = {'PHE': ['CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2'], 'TYR': ['CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2'],
         'TRP6': ['CD2', 'CE2', 'CZ2', 'CH2', 'CZ3', 'CE3'], 'TRP5': ['CG', 'CD1', 'NE1', 'CE2', 'CD2'], 'HIS': ['CG', 'ND1', 'CE1', 'NE2', 'CD2']}
mid, centre, pi, ringcount = [], [], [], {}
for ch in st[0]:
    for r in ch:
        CUR[0] = ('res', ch.name, r.seqid.num)
        keys = [r.name] if r.name in RINGS else (['TRP6', 'TRP5'] if r.name == 'TRP' else [])
        for k in keys:
            at = [r.find_atom(x, '*') for x in RINGS[k]]
            if any(x is None for x in at): continue
            P = np.array([x.pos.tolist() for x in at]); c = P.mean(0); nrm = np.linalg.svd(P - c)[2][2]
            mid += [val((P[i] + P[(i + 1) % len(P)]) / 2) for i in range(len(P))]; centre.append(val(c))
            pi += [(val(p + 0.7 * nrm), val(p - 0.7 * nrm)) for p in P]; ringcount[k] = ringcount.get(k, 0) + 1
cn, co, hh = [], [], []
for ch in st[0]:
    pol = ch.get_polymer()
    for i in range(len(pol) - 1):
        CUR[0] = ('blk', ch.name, pol[i].seqid.num // 5)
        C = pol[i].find_atom('C', '*'); O = pol[i].find_atom('O', '*'); Nn = pol[i + 1].find_atom('N', '*')
        if C and Nn and C.pos.dist(Nn.pos) < 1.5: cn.append(val((np.array(C.pos.tolist()) + np.array(Nn.pos.tolist())) / 2))
        if C and O: co.append(val((np.array(C.pos.tolist()) + np.array(O.pos.tolist())) / 2))
    for i in range(1, len(pol)):
        p, r = pol[i - 1], pol[i]; CUR[0] = ('blk', ch.name, r.seqid.num // 5)
        if r.name == 'PRO': continue
        C = p.find_atom('C', '*'); Nn = r.find_atom('N', '*'); CA = r.find_atom('CA', '*')
        if not (C and Nn and CA) or C.pos.dist(Nn.pos) > 1.5: continue
        C, Nn, CA = [np.array(x.pos.tolist()) for x in (C, Nn, CA)]
        b = (Nn - C) / np.linalg.norm(Nn - C) + (Nn - CA) / np.linalg.norm(Nn - CA); b /= np.linalg.norm(b); hh.append(val(Nn + 0.95 * b))
Vx = exact_L0(np.array(PTS)) / rms
# exact sums and FFT map share the same scale only if synth() and exact_L0 use the same normalisation: check on a few points
chk = sample(maps['dL0'], np.array(PTS[:200])) / rms; print('grid vs exact (first 200 pts): r =', round(float(np.corrcoef(chk, Vx[:200])[0, 1]), 4), ' slope', round(float(np.polyfit(Vx[:200], chk, 1)[0]), 3))
# ---- cluster-robust standard errors (bootstrap over rings / residue blocks)
G = lambda L: [GRP[i] for i in L]
cls = dict(amideH=hh, aromH=aromH, ring_mid=mid, centre=centre, cn=cn, co=co)
out = {}
rng = np.random.default_rng(20261028)
def stats(v, g, B=20000):
    v = np.asarray(v); keys = sorted(set(g), key=str); gi = np.array([keys.index(x) for x in g]); K = len(keys)
    S = np.bincount(gi, v, K); n = np.bincount(gi, None, K)
    boot = []
    for _ in range(B):
        s = rng.integers(0, K, K); boot.append(S[s].sum() / n[s].sum())
    boot = np.array(boot); m = v.mean()
    return dict(n=len(v), groups=K, mean=float(m), se_naive=float(v.std(ddof=1) / np.sqrt(len(v))), se_cluster=float(boot.std(ddof=1)),
                ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))])
for k, L in cls.items(): out[k] = stats([float(Vx[i]) for i in L], G(L))
out['pi'] = stats([float((Vx[i] + Vx[j]) / 2) for i, j in pi], [GRP[i] for i, j in pi])
for k, s in out.items(): print(f"{k:9s} n={s['n']:4d} groups={s['groups']:3d} mean {s['mean']:+.3f} se_naive {s['se_naive']:.3f} se_cluster {s['se_cluster']:.3f} 95% [{s['ci95'][0]:+.2f},{s['ci95'][1]:+.2f}]")
json.dump(out, open(OUT / 'aromatics_cluster_se.json', 'w'), indent=1)

# ---- class values and sections
f = lambda L: [float(Vx[i]) for i in L]
mid, centre, cn, co, hh, aromH = f(mid), f(centre), f(cn), f(co), f(hh), f(aromH)
pi = [float((Vx[i] + Vx[j]) / 2) for i, j in pi]
J = dict(rms=rms, coordinates='6MU9_deposited.pdb', ring_count=ringcount, ring_mid=mid, pi=pi, centre=centre, cn=cn, co=co, amideH=hh, aromH=aromH)
for k in ('amideH', 'aromH', 'ring_mid', 'pi', 'centre', 'cn', 'co'):
    x = np.array(J[k]); print(f'{k:9s} n={len(x):4d} mean {x.mean():+.2f}  sem {x.std() / np.sqrt(len(x)):.2f}')
print('rings', ringcount)
json.dump(J, open(OUT / 'aromatics_stats.json', 'w'), indent=0)
# ---- sections (panels A-F), same frames and extents as sample_sections_6mu9.py ----
R = {}
def section(c, ex, ey, ur, vr, step=0.04):
    u = np.linspace(*ur, int(round((ur[1] - ur[0]) / step)) + 1); v = np.linspace(*vr, int(round((vr[1] - vr[0]) / step)) + 1); U, V = np.meshgrid(u, v)
    X = c + U[..., None] * ex + V[..., None] * ey
    return u, v, {k: sample(arr, X) for k, arr in maps.items()}
def store(key, u, v, out, c, ex, ey, n, resi):
    R[key + '_u'] = u; R[key + '_v'] = v; R[key + '_frame'] = np.array([ex, ey, n])
    for k, arr in out.items(): R[f'{key}_{k}'] = arr
    near = {k: p for k, p in A.items() if np.linalg.norm(p - c) < 6}
    R[key + '_atomnames'] = np.array([f'{k[0]}:{k[1]}' for k in near]); R[key + '_atomxyz'] = np.array([[(p - c) @ ex, (p - c) @ ey, (p - c) @ n] for p in near.values()])
    R[key + '_Hxyz'] = np.array([[(p - c) @ ex, (p - c) @ ey, (p - c) @ n] for _, p in Hexp.get(resi, [])])
for resi, ring in ((85, ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']), (210, ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']), (227, ['CG', 'CD1', 'NE1', 'CE2', 'CD2', 'CE3', 'CZ3', 'CH2', 'CZ2'])):
    P = np.array([A[(resi, x)] for x in ring]); c = P.mean(0); nrm = np.linalg.svd(P - c)[2][2]
    ax = A[(resi, 'CG')] - c; ax -= (ax @ nrm) * nrm; ax /= np.linalg.norm(ax); side = np.cross(nrm, ax)
    u, v, o = section(c, side, ax, (-3.2, 3.2), (-3.2, 3.2)); store(f'r{resi}_face', u, v, o, c, side, ax, nrm, resi)
    u, v, o = section(c, nrm, ax, (-2.4, 2.4), (-3.2, 3.2)); store(f'r{resi}_edge', u, v, o, c, nrm, ax, side, resi)
R['rms'] = rms; np.savez_compressed(OUT / 'aromatics_sections.npz', **R); print('wrote sections')
