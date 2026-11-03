"""Effect of incomplete data on the Gly-Ala bond descriptors (Supplementary Table 'Incomplete data', 14 bonds,
C1-C2 excluded). Reflections and their Friedel mates are removed from the HAR and the IAM sets before filtering
(numpy FFT synthesis of Eq. 2 on the sph-map grid). Output: WORK/04_small_molecules/incompleteness_glyala.json"""
import _paths, json, numpy as np
from _paths import step
import molecules, maps as MP
from mtz import read_mtz
OUT = step("04_small_molecules"); IN = step("02_prepare")
m = molecules.load("glyala"); A = m["cell"]
def load(f):
    x = read_mtz(str(f)); d = x["data"]; return d[:, :3].astype(int), d[:, 3] * np.exp(1j * np.deg2rad(d[:, 4]))
h, Ch = load(IN / "glyala_fullP1.mtz"); h2, Ci = load(IN / "glyala_iam_fullP1.mtz"); assert (h == h2).all()
grid = np.array(MP.load_set("glyala", "obs")["L0"], copy=False).shape; c3 = np.sqrt(3 / (4 * np.pi))
kc = h / A; kn = np.linalg.norm(kc, axis=1); u = kc / kn[:, None]; dsp = 1 / kn
bonds = [b for b in m["bonds"] if b != ("C1", "C2")]
def interp(M, f):
    g = np.array(M.shape); p = (np.array(f) % 1) * g; i0 = np.floor(p).astype(int); dd = p - i0; v = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (dd[0] if dx else 1 - dd[0]) * (dd[1] if dy else 1 - dd[1]) * (dd[2] if dz else 1 - dd[2])
                v += w * M[(i0[0] + dx) % g[0], (i0[1] + dy) % g[1], (i0[2] + dz) % g[2]]
    return v
def maps(C, mask):
    hh = h[mask]; cc = np.conj(C[mask]); uu = u[mask]
    hh = np.vstack([hh, -hh]); cc = np.concatenate([cc, np.conj(cc)]); uu = np.vstack([uu, -uu])
    out = []
    for coef in (cc / np.sqrt(4 * np.pi), cc * 1j * c3 * uu[:, 0], cc * 1j * c3 * uu[:, 1], cc * 1j * c3 * uu[:, 2]):
        G = np.zeros(grid, complex); np.add.at(G, (hh[:, 0] % grid[0], hh[:, 1] % grid[1], hh[:, 2] % grid[2]), coef)
        out.append(np.fft.ifftn(G).real * np.prod(grid))
    return out
def desc(M):
    res = []
    for a, b in bonds:
        fa, fb = np.array(m["atoms"][a]), np.array(m["atoms"][b]); bv = (fb - fa) * A; bv /= np.linalg.norm(bv)
        pts = [fa + t * (fb - fa) for t in np.linspace(0.3, 0.7, 5)]
        d = np.mean([[interp(M[i], p) for i in (1, 2, 3)] for p in pts], axis=0); r0 = np.mean([interp(M[0], p) for p in pts])
        res.append((d / np.linalg.norm(d), np.linalg.norm(d) / r0))
    return res
scen = {"complete": np.ones(len(h), bool)}
for p in (0.10, 0.25):
    for s in range(3): scen[f"random {int(p * 100)}% (seed {s})"] = np.random.default_rng(s).random(len(h)) > p
for ang in (20, 30): scen[f"missing cone {ang} deg about c*"] = np.abs(u[:, 2]) < np.cos(np.deg2rad(ang))
for dm in (0.8, 1.0): scen[f"truncated to {dm} A"] = dsp >= dm
scen["low-res d>3 A removed"] = dsp <= 3.0
base = desc(maps(Ch, scen["complete"])); rows = []
acute = lambda x, y: np.degrees(np.arccos(min(1, abs(x @ y))))
for name, mask in scen.items():
    dh = desc(maps(Ch, mask)); di = desc(maps(Ci, mask))
    dang = [acute(x[0], y[0]) for x, y in zip(dh, base)]; hari = [acute(x[0], y[0]) for x, y in zip(dh, di)]
    ratio = np.array([x[1] / y[1] for x, y in zip(dh, di)])
    rows.append(dict(scenario=name, completeness=float(mask.mean()), median_dir_change=float(np.median(dang)), max_dir_change=float(np.max(dang)),
                     max_HAR_vs_IAM_dir=float(np.max(hari)), ratio_min=float(ratio.min()), ratio_max=float(ratio.max()), n_ratio_below_1=int((ratio < 1).sum())))
    print("%-34s compl %.2f | dir change med %5.1f max %5.1f | HAR-IAM max %4.1f | D ratio %.2f-%.2f (%d/14 <1)" % (
        name, mask.mean(), np.median(dang), np.max(dang), np.max(hari), ratio.min(), ratio.max(), (ratio < 1).sum()))
json.dump(rows, open(OUT / "incompleteness_glyala.json", "w"), indent=1)
