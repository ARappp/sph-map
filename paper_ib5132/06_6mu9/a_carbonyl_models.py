"""Observed-model angles (Eq. 9) at the 263 carbonyl O atoms of 6MU9 for four reference models: deposited (REFMAC, no H),
(a) Phenix no H, (b) riding H, (c) (b) + H on O299; and for the deposited coefficients with the coordinates of an
intermediate refinement file (6mu9_final.pdb). Filters: field at the O atom, sphere averages r0 = 0.3 and 0.6 A.
Supplementary Tables 'reference models' and 'O299 (models b and c)'; main text Sections 4.3, 4.4.
Output (WORK/06_6mu9): carbonyl_models.csv (every site), carbonyl_models.json (summary)"""
import _paths, json, csv, numpy as np
from _paths import step
import protein as PR
OUT = step("06_6mu9")
JOBS = {k: (PR.D6 / v[0], v[1], PR.D6 / v[2]) for k, v in PR.MODELS.items()}
JOBS["deposited_intermediate_coords"] = (PR.D6 / "6mu9_final.mtz", PR.REFMAC, PR.D6 / "6mu9_final.pdb")
KEY = ["A75LYS", "A298ALA", "A299LYS", "A302VAL"]
res, rows, A = {}, [], {}
for tag, (mtz, cols, pdb) in JOBS.items():
    H, Cd, Cm, Fr, kc = PR.load(mtz, *cols); names, O, Cc = PR.carbonyl_sites(pdb)
    Mult = {"point": PR.mult(kc)[0], "r0_0.3": PR.mult(kc, 0.3)[0], "r0_0.6": PR.mult(kc, 0.6)[0]}
    Fd = PR.fields(H, Cd, Mult, Fr, O); Fm = PR.fields(H, Cm, Mult, Fr, O); res[tag] = {"n_refl": len(H), "n_sites": len(names)}
    for f in Mult:
        a = PR.ang(Fd[f], Fm[f]); A[(tag, f)] = (names, a)
        d = dict(median=float(np.median(a)), p90=float(np.percentile(a, 90)), median_dobs_to_CO=float(np.median(PR.ang(Fd[f], O - Cc))))
        for s in KEY: d[s] = [round(float(a[names.index(s)]), 2), PR.rank(a, names.index(s))]
        d["top10"] = [(names[j], round(float(a[j]), 1)) for j in np.argsort(-a)[:10]]; res[tag][f] = d
        rows += [(tag, f, n, round(float(a[j]), 3), PR.rank(a, j)) for j, n in enumerate(names)]
    print(tag, {f: (round(res[tag][f]["median"], 1), res[tag][f]["A299LYS"]) for f in Mult}, flush=True)
# hydrogen test: change from model (b) to model (c) at every site
res["b_to_c"] = {}
for f in ("point", "r0_0.3", "r0_0.6"):
    nb, ab = A[("b_ridingH", f)]; nc, ac = A[("c_plusH_O299", f)]; assert nb == nc
    dec = ab - ac; i = nb.index("A299LYS"); other = np.delete(np.abs(dec), i)
    res["b_to_c"][f] = dict(O299_decrease=float(dec[i]), rank_of_O299_decrease=PR.rank(dec, i), median_abs_change_other=float(np.median(other)),
                            largest_other=(nb[int(np.argmax(np.where(np.arange(len(dec)) == i, -np.inf, dec)))], float(np.max(np.delete(dec, i)))))
print(json.dumps(res["b_to_c"], indent=1))
# R factors as written by the refinement programs
for tag, (mtz, cols, pdb) in JOBS.items():
    lines = [l for l in open(pdb) if l.startswith("REMARK   3   R VALUE            (WORKING SET)") or l.startswith("REMARK   3   FREE R VALUE      ")]
    res[tag]["R_work_R_free_pdb"] = [float(l.split(":")[1]) for l in lines[:2]]
json.dump(res, open(OUT / "carbonyl_models.json", "w"), indent=1)
with open(OUT / "carbonyl_models.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["model", "filter", "site", "angle_obs_model_deg", "rank"]); w.writerows(rows)
