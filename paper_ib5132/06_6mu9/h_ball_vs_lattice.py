"""Sphere average on the 0.1 A lattice (Lambda, Eq. 8 of the paper) against the continuous ball average Phi = 3 j1(x)/x,
x = 2 pi k r0 (main text Section 3.6): largest difference of the two weights over the 6MU9 reflections, and the change
of the observed-model angles at the 263 carbonyl O atoms of the deposited model when Phi replaces Lambda (r0 = 0.6 A).
Output: WORK/06_6mu9/ball_vs_lattice.json"""
import _paths, json, numpy as np
from _paths import step
import protein as PR
OUT = step("06_6mu9"); r0 = 0.6
mtz, cols, pdb, _ = PR.MODELS["deposited"]; H, Cd, Cm, Fr, kc = PR.load(PR.D6 / mtz, *cols); names, O, _ = PR.carbonyl_sites(PR.D6 / pdb)
k = np.linalg.norm(kc, axis=1); kh = kc / k[:, None]; x = 2 * np.pi * k * r0; Phi = 3 * (np.sin(x) - x * np.cos(x)) / x ** 3
off = PR.sphere_offsets(r0); Lam = np.concatenate([np.exp(2j * np.pi * (kc[i:i + 4000] @ off.T)).mean(1) for i in range(0, len(kc), 4000)])
M = {"lattice": 1j * np.sqrt(3 / (4 * np.pi)) * kh * Lam[:, None], "ball": 1j * np.sqrt(3 / (4 * np.pi)) * kh * Phi[:, None]}
Fd = PR.fields(H, Cd, M, Fr, O); Fm = PR.fields(H, Cm, M, Fr, O); A = {f: PR.ang(Fd[f], Fm[f]) for f in M}; i = names.index("A299LYS")
d = np.abs(A["ball"] - A["lattice"])
out = dict(n_lattice_points=len(off), max_abs_Phi_minus_Lambda=float(np.abs(Phi - Lam.real).max()), max_abs_imag_Lambda=float(np.abs(Lam.imag).max()),
           site_change_median=float(np.median(d)), site_change_max=float(d.max()),
           **{f"O299_{f}": [round(float(A[f][i]), 2), PR.rank(A[f], i)] for f in M}, **{f"median_{f}": float(np.median(A[f])) for f in M})
print(json.dumps(out, indent=1)); json.dump(out, open(OUT / "ball_vs_lattice.json", "w"), indent=1)
