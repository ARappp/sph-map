"""Step 07c: amide-H test for the PDB-REDO structures (Supplementary Table 'amide H'): difference dipolar field
(data_noH - model_noH, H-omit coefficients of a_prepare.py) at the amide N atoms against the expected N->H direction;
field at N and averages over spheres r0 = 0.3, 0.6 A (FFT maps, 3 points per d_min). For 4Y44 and 3X3R pooled, also the
reversed field of the H-free model alone ('model only') and observed amplitudes with random phases (P1, 5 sets, seeds
1000-1004). Needs a_prepare.py. About 10 min. Output: WORK/07_pdbredo/amide_h.json"""
import _paths, json, numpy as np
from _paths import INPUT, step
import direct as D, amide
W = step("07_pdbredo"); IDS = ["4y44", "3x3r", "3ume", "3la9", "4gyb"]
res = {}; pool = {}
for p in IDS:
    sites = amide.amide_N(INPUT / "pdbredo" / f"{p}_final.pdb"); X = np.array([s[1] for s in sites]); E = np.array([s[2] for s in sites])
    INP = {"data_noH": (W / p / f"{p}_data_noH_P1.mtz", "F", "PHI"), "model_noH": (W / p / f"{p}_model_noH_P1.mtz", "F", "PHI"),
           "data": (INPUT / "pdbredo" / f"{p}_final.mtz", "FWT", "PHWT"), "model": (INPUT / "pdbredo" / f"{p}_final.mtz", "FC_ALL", "PHIC_ALL")}
    co = {k: D.load(f, a, ph) for k, (f, a, ph) in INP.items()}
    V = {k: amide.vectors(c, X) for k, c in co.items()}; res[p] = amide.summary(V, E)
    print(p, len(sites), {k: round(v["frac_le30"], 3) for k, v in res[p].items()}, flush=True)
    if p in ("4y44", "3x3r"):
        pool.setdefault("real", {}); pool.setdefault("model only", {}); pool.setdefault("P1", {})
        for pr in V["data"]:
            m0 = V["model_noH"][pr]
            pool["real"].setdefault(pr, []).extend(amide.angles(V["data_noH"][pr] - m0, E))
            pool["model only"].setdefault(pr, []).extend(amide.angles(-m0, E))
        c0 = co["data_noH"]
        for seed in range(5):
            cr = D.Coefficients(c0.cell, c0.hkl, np.abs(c0.C) * D.random_phases(c0.hkl, 1000 + seed), dict(source=f"P1 seed {seed}"))
            Vc = amide.vectors(cr, X)
            for pr in Vc: pool["P1"].setdefault(pr, []).extend(amide.angles(Vc[pr] - V["model_noH"][pr], E))
        print(p, "P1 done", flush=True)
res["4y44+3x3r"] = {c: {pr: dict(n=len(a), frac_le30=float(np.mean(np.array(a) <= 30)), median_angle=float(np.median(a))) for pr, a in d.items()} for c, d in pool.items()}
print(json.dumps(res["4y44+3x3r"], indent=1)); json.dump(res, open(W / "amide_h.json", "w"), indent=1)
