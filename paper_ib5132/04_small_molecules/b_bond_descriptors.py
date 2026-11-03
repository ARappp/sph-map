"""Bond descriptors of urea (4 bonds) and Gly-Ala (15 bonds) for the three inputs:
obs   = F_obs with HAR phases,  iamph = F_obs with IAM phases,  iam = IAM structure factors.
Supplementary Table 'Bond descriptors' (angles, D_AB, HAR/IAM), Fig. 2, Fig. S4, and the summary numbers of
Sections 4.1, 4.2 and 4.5 (ratio ranges, amplitude and phase parts, angle agreement).
Output (WORK/04_small_molecules): bond_descriptors.csv, bond_summary.json"""
import _paths, json, numpy as np, pandas as pd
from _paths import step
import molecules, maps as MP
OUT = step("04_small_molecules")
cls = lambda k: "C-H" if "-H" in k and k.startswith("C") else ("N-H" if "-H" in k else ("C=O/C-O" if "O" in k else ("C-N" if "N" in k else "C-C")))
rows = []
for mol in ("urea", "glyala"):
    m = molecules.load(mol); A = m["cell"]; V = float(np.prod(A))
    R = {t: MP.descriptors(MP.load_set(mol, t), m["atoms"], A, m["bonds"]) for t in ("obs", "iamph", "iam")}
    b0 = molecules.F000[mol] / (V * np.sqrt(4 * np.pi))          # rho_00 of F(000) (orthorhombic/tetragonal cells)
    for k in R["obs"]:
        r = {t: R[t][k] for t in R}
        rows.append(dict(mol=mol, bond=k, cls=cls(k), excl=(mol, k) in molecules.EXCLUDED,
                         ang_H=MP.acute(r["obs"]["angle"]), ang_M=MP.acute(r["iamph"]["angle"]), ang_I=MP.acute(r["iam"]["angle"]),
                         D_H=r["obs"]["D"], D_M=r["iamph"]["D"], D_I=r["iam"]["D"],
                         ratio=r["obs"]["D"] / r["iam"]["D"], ratio_M=r["iamph"]["D"] / r["iam"]["D"],
                         ratio_F000=(r["obs"]["dnorm"] / (r["obs"]["rho0"] + b0)) / (r["iam"]["dnorm"] / (r["iam"]["rho0"] + b0)),
                         vHM=MP.angle(np.array(r["obs"]["d"]), np.array(r["iamph"]["d"])),
                         vHI=MP.angle(np.array(r["obs"]["d"]), np.array(r["iam"]["d"])),
                         vMI=MP.angle(np.array(r["iamph"]["d"]), np.array(r["iam"]["d"])),
                         dnorm_rel=r["obs"]["dnorm"] / r["iam"]["dnorm"] - 1, rho_rel=r["obs"]["rho0"] / r["iam"]["rho0"] - 1))
t = pd.DataFrame(rows)
t["amp"] = (t.D_I - t.D_M) / t.D_I; t["phase"] = (t.D_M - t.D_H) / t.D_I
t["spread"] = t[["ang_H", "ang_M", "ang_I"]].max(1) - t[["ang_H", "ang_M", "ang_I"]].min(1)
t.to_csv(OUT / "bond_descriptors.csv", index=False)
s = t[~t.excl]; out = {}
out["ratio_range"] = [s.ratio.min(), s.ratio.max()]; out["reduction_pct"] = [100 * (1 - s.ratio.max()), 100 * (1 - s.ratio.min())]
out["by_class"] = {c: [g.ratio.min(), g.ratio.max()] for c, g in s.groupby("cls")}
out["ratio_F000"] = [s.ratio_F000.min(), s.ratio_F000.max()]; out["ratio_IAMph"] = [s.ratio_M.min(), s.ratio_M.max()]
out["amp_pct"] = [100 * s.amp.min(), 100 * s.amp.max(), 100 * s.amp.mean()]; out["phase_pct"] = [100 * s.phase.min(), 100 * s.phase.max(), 100 * s.phase.mean()]
out["total_mean_pct"] = 100 * (s.amp + s.phase).mean(); out["amp_share"] = s.amp.mean() / (s.amp + s.phase).mean()
out["max_ang_H"] = s.ang_H.max(); out["max_spread"] = s.spread.max(); out["max_vHM"] = s.vHM.max(); out["max_vHI"] = s.vHI.max()
out["max_v_all"] = s[["vHM", "vHI", "vMI"]].max().max()
out["excluded"] = t[t.excl][["mol", "bond", "ang_H", "D_H", "ratio"]].to_dict("records")
out["C4"] = t[t.bond.isin(["C4-O3", "C4-N1"])][["bond", "dnorm_rel", "rho_rel"]].to_dict("records")
json.dump(out, open(OUT / "bond_summary.json", "w"), indent=1, default=float)
pd.set_option("display.width", 200)
print(t[["mol", "bond", "ang_H", "ang_M", "ang_I", "D_H", "D_M", "D_I", "ratio"]].round(3).to_string(index=False))
print(f"\n17 bonds: D_HAR/D_IAM {out['ratio_range'][0]:.2f}-{out['ratio_range'][1]:.2f}; amplitude part {out['amp_pct'][2]:.1f} %, "
      f"phase part {out['phase_pct'][2]:.1f} % (total {out['total_mean_pct']:.1f} %); angles within {out['max_spread']:.1f} deg")
