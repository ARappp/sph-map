"""Known-H test and model-phase control for urea and Gly-Ala (Supplementary Tables 'Known-H test' and 'Model-phase
control'; main text Sections 3.7 and 4.2). Controls: P1 all phases random; P2 phases of F_H random; P3 |F_obs| with
H-free phases; P4 scrambled H amplitudes with H-free phases; 20 random sets (seeds 1000-1019; set NSEED to change).
About 15 min. Output (WORK/05_hydrogen_tests): htests_by_site.csv, pooled.csv, per_dataset.csv, scale.json"""
import _paths, os, json, time, numpy as np, pandas as pd
from _paths import step
import direct as D, htest as H
OUT = step("05_hydrogen_tests"); NSEED = int(os.environ.get("NSEED", 20))
res = {}; rows = []
for ds in ("urea", "glyala"):
    t0 = time.time(); c = H.context(ds); coI = c["coI"]; X, E = c["X"], c["E"]
    sets = {"data": D.with_f000(coI, c["CD"], c["F0"], "d"), "model_I": D.with_f000(coI, coI.C, c["F0"], "I"),
            "model_noH": D.with_f000(coI, c["CnoH"], c["F0"] - c["nH_e"], "noH"), "data_omit": D.with_f000(coI, np.abs(c["CD"]) * c["phnoH"], c["F0"], "omit")}
    V = {inp: H.exactA(co, X) for inp, co in sets.items()}
    for m in V["data"]:
        for j, lab in enumerate(c["labels"]):
            real = V["data"][m][j] - V["model_noH"][m][j]; full = V["data"][m][j] - V["model_I"][m][j]; omit = V["data_omit"][m][j] - V["model_noH"][m][j]
            rows.append(dict(dataset=ds, site=lab, nH=c["nH"][j], method=m, control="real", seed=-1, angle=H.ang(real, E[j]), ratio=np.linalg.norm(real) / np.linalg.norm(full),
                             angle_omit=H.ang(omit, E[j]), ratio_omit_to_real=np.linalg.norm(omit) / np.linalg.norm(real), angle_geom=H.ang(-V["model_noH"][m][j], E[j])))
    print(ds, list(zip(c["labels"], c["nH"])), c["scale"], "%.0fs" % (time.time() - t0), flush=True)
    for ctrl in ("P1", "P2", "P4"):
        for seed in range(NSEED):
            ph = D.random_phases(coI.hkl, 1000 + seed)
            if ctrl == "P1": C = np.abs(c["CD"]) * ph
            elif ctrl == "P2": C = c["CD"] - c["FHy"] + np.abs(c["FHy"]) * ph
            else: C = np.abs(c["CD"] - c["FHy"] + np.abs(c["FHy"]) * ph) * c["phnoH"]
            W = H.fftA(D.with_f000(coI, C, c["F0"], ctrl), X, c["Ainv"])
            for m in W:
                for j, lab in enumerate(c["labels"]):
                    dc = W[m][j] - V["model_noH"][m][j]; full = V["data"][m][j] - V["model_I"][m][j]
                    rows.append(dict(dataset=ds, site=lab, nH=c["nH"][j], method=m, control=ctrl, seed=seed, angle=H.ang(dc, E[j]), ratio=np.linalg.norm(dc) / np.linalg.norm(full)))
        print(ds, ctrl, "%.0fs" % (time.time() - t0), flush=True)
    res[ds] = c["scale"]
t = pd.DataFrame(rows); t.to_csv(OUT / "htests_by_site.csv", index=False)
pooled = t.groupby(["method", "control"], sort=False).agg(n=("angle", "size"), med=("angle", "median"), le30=("angle", lambda a: np.mean(a <= 30)), ratio=("ratio", "median")).reset_index()
per = t.groupby(["dataset", "method", "control"], sort=False).agg(n=("angle", "size"), med=("angle", "median"), le30=("angle", lambda a: int((a <= 30).sum())), ratio=("ratio", "median"),
      omit=("angle_omit", "median"), omit_le30=("angle_omit", lambda a: int((a <= 30).sum())), omit_ratio=("ratio_omit_to_real", "median"), geom=("angle_geom", "median")).reset_index()
pooled.to_csv(OUT / "pooled.csv", index=False); per.to_csv(OUT / "per_dataset.csv", index=False); json.dump(res, open(OUT / "scale.json", "w"), indent=1)
pd.set_option("display.width", 250); print(pooled.round(2).to_string(index=False)); print(per.round(2).to_string(index=False))
