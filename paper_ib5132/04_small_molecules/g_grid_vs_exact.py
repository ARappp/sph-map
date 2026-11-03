"""Accuracy of the grid path: bond descriptors from the interpolated sph-map maps (6 points per d_min, step 03;
and 3 points per d_min, run here) against exact direct sums of Eq. (2) at the same points
(Supplementary Table 'Radial weight', caption). Output: WORK/04_small_molecules/grid_vs_exact.json"""
import _paths, json, numpy as np, gemmi
from _paths import step
import molecules, maps as MP, sphmap, direct
OUT = step("04_small_molecules"); IN = step("02_prepare")
res = {}
for mol in ("urea", "glyala"):
    m = molecules.load(mol); A = m["cell"]; cell = gemmi.UnitCell(*A, 90, 90, 90)
    for s, suf in (("obs", ""), ("iam", "_iam")):
        co = direct.load(IN / f"{mol}{suf}_fullP1.mtz", "F", "PHI")
        g3 = sphmap.run(IN / f"{mol}{suf}_fullP1.mtz", OUT / "grid_3_per_dmin", f"{mol}_{s}", 3)
        grids = {"6": MP.descriptors(MP.load_set(mol, s), m["atoms"], A, m["bonds"]),
                 "3": MP.descriptors(MP.loadmaps(g3, f"{mol}_{s}"), m["atoms"], A, m["bonds"])}
        for a, b in m["bonds"]:
            fa, fb = np.array(m["atoms"][a]), np.array(m["atoms"][b])
            pts = np.array([cell.orthogonalize(gemmi.Fractional(*(fa + t * (fb - fa)))).tolist() for t in np.linspace(0.3, 0.7, 5)])
            v = direct.exact(co, pts, 1); d = direct.l1_to_xyz(v[:, 1:4]).mean(0); r0 = v[:, 0].mean()
            e = res.setdefault(f"{mol} {a}-{b}", {}); e[f"D_exact_{s}"] = float(np.linalg.norm(d) / r0)
            for g, R in grids.items():
                gd = np.array(R[f"{a}-{b}"]["d"]); e[f"dir_diff_{g}_{s}"] = MP.angle(d, gd); e[f"D_grid{g}_{s}"] = R[f"{a}-{b}"]["D"]
for k, e in res.items():
    for g in ("6", "3"):
        e[f"ratio_rel_diff_{g}"] = (e[f"D_grid{g}_obs"] / e[f"D_grid{g}_iam"]) / (e["D_exact_obs"] / e["D_exact_iam"]) - 1
inc = [k for k in res if tuple(k.split()) not in {(a, b) for a, b in molecules.EXCLUDED}]
summ = {g: dict(max_dir_diff=max(max(res[k][f"dir_diff_{g}_obs"], res[k][f"dir_diff_{g}_iam"]) for k in inc),
                max_ratio_rel_diff=max(abs(res[k][f"ratio_rel_diff_{g}"]) for k in inc)) for g in ("6", "3")}
summ["3_CO_CN_max_dir_diff"] = max(max(res[k]["dir_diff_3_obs"], res[k]["dir_diff_3_iam"]) for k in inc if "H" not in k.split()[1])
print(json.dumps(summ, indent=1)); json.dump(dict(summary=summ, bonds=res), open(OUT / "grid_vs_exact.json", "w"), indent=1)
