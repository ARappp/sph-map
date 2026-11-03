"""Resolution sweep for the deposited 6MU9 model (Fig. 4, Fig. S5, Supplementary Table 'sites with the largest angles',
6MU9 part of the Tapering table): observed-model angles at the 263 carbonyl O atoms for 11 resolution cut-offs
(1.0-2.0 A) and full resolution, r0 = 0.3 and 0.6 A, without taper and with a cosine or Kaiser taper to each cut-off.
About 3 min. Output (WORK/06_6mu9): sweep_sites.csv (untapered, every site and cut-off), sweep_summary.json"""
import _paths, json, csv, numpy as np
from _paths import step
import protein as PR
OUT = step("06_6mu9")
CUTS = [round(1.0 + 0.1 * i, 1) for i in range(11)]
mtz, cols, pdb, _ = PR.MODELS["deposited"]
H, Cd, Cm, Fr, kc = PR.load(PR.D6 / mtz, *cols); names, O, Cc = PR.carbonyl_sites(PR.D6 / pdb); co = O - Cc
kfull = np.linalg.norm(kc, axis=1).max(); Mult = {}
for r in (0.3, 0.6):
    B, k = PR.mult(kc, r)
    for tp in ("none", "cosine", "kaiser"):
        Mult[(r, tp, "full")] = B * PR.taper(tp, k, kfull)[:, None]
        for c in CUTS:
            kmax = k[k <= 1 / c + 1e-12].max(); Mult[(r, tp, c)] = B * PR.taper(tp, k, kmax)[:, None]
Fd = PR.fields(H, Cd, Mult, Fr, O); Fm = PR.fields(H, Cm, Mult, Fr, O)
rows = []; summ = {}; i299 = names.index("A299LYS"); rank = PR.rank; ang = PR.ang
for r in (0.3, 0.6):
    for tp in ("none", "cosine", "kaiser"):
        A = np.array([ang(Fd[(r, tp, c)], Fm[(r, tp, c)]) for c in CUTS]).T                     # sites x cut-offs
        Aco = np.array([ang(Fd[(r, tp, c)], co) for c in CUTS]).T; Acm = np.array([ang(Fm[(r, tp, c)], co) for c in CUTS]).T
        af = ang(Fd[(r, tp, "full")], Fm[(r, tp, "full")]); m = A.mean(1)
        s = dict(full_O299=[round(float(af[i299]), 2), rank(af, i299)], full_median=float(np.median(af)),
                 sweep_mean_O299=[round(float(m[i299]), 2), rank(m, i299)], sweep_mean_median=float(np.median(m)),
                 O299_by_cut=[round(float(x), 1) for x in A[i299]], O299_rank_by_cut=[rank(A[:, j], i299) for j in range(len(CUTS))],
                 median_by_cut=[round(float(x), 1) for x in np.median(A, 0)],
                 sd_dir_O299_obs=round(float(Aco[i299].std()), 2), sd_dir_O299_model=round(float(Acm[i299].std()), 2),
                 sd_dir_median_obs=round(float(np.median(Aco.std(1))), 2), sd_dir_median_model=round(float(np.median(Acm.std(1))), 2),
                 top10_sweep=[(names[j], round(float(m[j]), 1), round(float(Aco[j].std()), 1), round(float(Acm[j].std()), 1), rank(m, j)) for j in np.argsort(-m)[:10]],
                 top_full=[(names[j], round(float(af[j]), 1)) for j in np.argsort(-af)[:13]], _top=set(np.argsort(-af)[:13].tolist()))
        summ[f"{r}_{tp}"] = s
        if tp == "none":
            rows += [(n, r, c, round(float(Aco[i, j]), 2), round(float(Acm[i, j]), 2), round(float(A[i, j]), 2),
                      float(np.linalg.norm(Fd[(r, tp, c)][i])), float(np.linalg.norm(Fm[(r, tp, c)][i]))) for i, n in enumerate(names) for j, c in enumerate(CUTS)]
for r in (0.3, 0.6):
    for tp in ("cosine", "kaiser"): summ[f"{r}_{tp}"]["top13_in_common_with_untapered"] = len(summ[f"{r}_none"]["_top"] & summ[f"{r}_{tp}"]["_top"])
for v in summ.values(): v.pop("_top")
json.dump(summ, open(OUT / "sweep_summary.json", "w"), indent=1)
with open(OUT / "sweep_sites.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["site", "r0", "dmin", "angle_CO_FWT", "angle_CO_FCALL", "angle_FWT_vs_FCALL", "mag_FWT", "mag_FCALL"]); w.writerows(rows)
for k, v in summ.items(): print(k, "full", v["full_O299"], "sweep", v["sweep_mean_O299"], "ranks", min(v["O299_rank_by_cut"]), "-", max(v["O299_rank_by_cut"]), "sd", v["sd_dir_O299_obs"])
