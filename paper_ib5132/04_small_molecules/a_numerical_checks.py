"""Numerical checks of the implementation (Supplementary Table 'Numerical checks', main text Section 2.3), Gly-Ala:
(1) sph-map channels against an independent numpy FFT synthesis of Eq. (2) (correlation, relative r.m.s.);
(2) the identity d = sqrt(3/4pi)/(2pi) grad rho_{1/k} (Eq. 4), gradient by central finite differences;
(3) sign: d points towards the atom at 0.3 A from each non-H atom; (4) translation equivariance;
(5) Plancherel: sum_m int rho_lm^2 dV = (2l+1)/(4pi) int (rho - mean)^2 dV for l = 0, 1.
Output: WORK/04_small_molecules/numerical_checks.json"""
import _paths, json, numpy as np
from _paths import step
import molecules, maps as MP
from expand import load, expand, parse_ops
OUT = step("04_small_molecules")
m = molecules.load("glyala"); A = m["cell"]
cell, hkl, C = load(str(step("02_prepare") / "gly-ala_Fphi.mtz")); k, v = expand(hkl, C, parse_ops(m["ops"]), 1)
Cpos = np.conj(v)                                    # A exp(-i phi_ccp4): coefficients of the positive synthesis
kc = k / A; kn = np.linalg.norm(kc, axis=1); u = kc / kn[:, None]
M = MP.load_set("glyala", "obs"); sph = {key: np.array(M[c], copy=True) for key, c in (("L0", "L0"), ("L1x", "x"), ("L1y", "y"), ("L1z", "z"))}
grid = sph["L0"].shape
def synth(coef):
    G = np.zeros(grid, complex); np.add.at(G, (k[:, 0] % grid[0], k[:, 1] % grid[1], k[:, 2] % grid[2]), coef)
    return np.fft.ifftn(G) * np.prod(grid)            # sum C exp(+2 pi i h.x)
c = np.sqrt(3 / (4 * np.pi))
mine = {"L0": synth(Cpos / np.sqrt(4 * np.pi)), "L1x": synth(Cpos * 1j * c * u[:, 0]), "L1y": synth(Cpos * 1j * c * u[:, 1]), "L1z": synth(Cpos * 1j * c * u[:, 2])}
out = {"grid": list(grid)}
out["max_imag_over_real"] = float(max(np.abs(x.imag).max() / np.abs(x.real).max() for x in mine.values()))
def cc(a, b): a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
for key in mine:
    x = mine[key].real; s = sph[key]; sc = (x * s).sum() / (x * x).sum()
    out["CC_numpy_vs_sph_" + key] = cc(x, s); out["relrms_" + key] = float(np.sqrt(((s - sc * x) ** 2).mean()) / np.sqrt((s ** 2).mean()))
r1k = synth(Cpos / kn).real
for ax, key in enumerate(("L1x", "L1y", "L1z")):
    h = A[ax] / grid[ax]; g = (np.roll(r1k, -1, ax) - np.roll(r1k, 1, ax)) / (2 * h)
    out["CC_gradient_identity_" + key] = cc(c / (2 * np.pi) * g, mine[key].real)
toward = []
for nm, x in m["atoms"].items():
    if nm.startswith("H"): continue
    for e in np.eye(3):
        for sgn in (1, -1):
            p = np.array(x) + sgn * e * 0.3 / A; toward.append(MP.vec(M, p) @ (-sgn * e) > 0)
out["sph_fraction_d_toward_atom"] = float(np.mean(toward))
t = np.array([7 / grid[0], 13 / grid[1], 29 / grid[2]])
sh = synth(Cpos * 1j * c * u[:, 0] * np.exp(-2j * np.pi * (k @ t))).real
ref = np.roll(mine["L1x"].real, (7, 13, 29), axis=(0, 1, 2))
out["translation_maxabsdiff_rel"] = float(np.abs(sh - ref).max() / np.abs(ref).max())
rho = synth(Cpos).real; var = ((rho - rho.mean()) ** 2).sum()
out["plancherel_l0_rel_error"] = float(abs((mine["L0"].real ** 2).sum() / (var / (4 * np.pi)) - 1))
out["plancherel_l1_rel_error"] = float(abs(sum((mine[q].real ** 2).sum() for q in ("L1x", "L1y", "L1z")) / (3 * var / (4 * np.pi)) - 1))
json.dump(out, open(OUT / "numerical_checks.json", "w"), indent=1); print(json.dumps(out, indent=1))
