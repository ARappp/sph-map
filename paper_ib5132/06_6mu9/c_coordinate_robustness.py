"""Robustness of the O299 rank to coordinate errors (Supplementary Table 'robustness'): every carbonyl O of the deposited
model displaced at random (normal, total r.m.s. 0.01, 0.02, 0.05 A, and 0.02 A per axis = 0.035 A total), 50 sets each,
exact site values recomputed for r0 = 0.3 and 0.6 A. About 5 min. Output: WORK/06_6mu9/coordinate_robustness.json"""
import _paths, json, numpy as np
from _paths import step
import protein as PR
OUT = step("06_6mu9"); NTRIAL = 50
mtz, cols, pdb, _ = PR.MODELS["deposited"]
H, Cd, Cm, Fr, kc = PR.load(PR.D6 / mtz, *cols); names, O, Cc = PR.carbonyl_sites(PR.D6 / pdb); i299 = names.index("A299LYS")
Mult = {r: PR.mult(kc, r)[0] for r in (0.3, 0.6)}
SET = {"0.01_total": (0.01 / np.sqrt(3), 1), "0.02_total": (0.02 / np.sqrt(3), 2), "0.05_total": (0.05 / np.sqrt(3), 5), "0.02_per_axis": (0.02, 22)}
out = {}
for label, (sd, seed) in SET.items():
    AA = []
    for t in range(NTRIAL):
        X = O + np.random.default_rng([2510, seed, t]).normal(scale=sd, size=O.shape)
        Fd = PR.fields(H, Cd, Mult, Fr, X); Fm = PR.fields(H, Cm, Mult, Fr, X); AA.append([PR.ang(Fd[r], Fm[r]) for r in (0.3, 0.6)])
    AA = np.array(AA)
    for j, r in enumerate((0.3, 0.6)):
        A = AA[:, j]; rk = (A > A[:, [i299]]).sum(1) + 1; sdv = A.std(0)
        out[f"{r}_{label}"] = dict(n=NTRIAL, angle=[round(float(A[:, i299].min()), 1), round(float(A[:, i299].max()), 1)],
                                   rank=[int(rk.min()), int(rk.max())], rank_median=float(np.median(rk)), top13_pct=float((rk <= 13).mean() * 100),
                                   sd_median=round(float(np.median(sdv)), 2), sd_p90=round(float(np.percentile(sdv, 90)), 2))
        print(r, label, out[f"{r}_{label}"], flush=True)
json.dump(out, open(OUT / "coordinate_robustness.json", "w"), indent=1)
