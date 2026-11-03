"""Sensitivity of the bond descriptors to a Fourier taper (Supplementary Table 'Tapering', bond part): all
coefficients multiplied by a cosine taper (start 0.9 k_max) or a Kaiser taper (beta 8) of sph-map before filtering.
Runs sph-map 18 times (about 2 min). Output: WORK/04_small_molecules/taper/ (maps), taper_bonds.csv, taper_summary.csv"""
import _paths, numpy as np, pandas as pd, gemmi
from _paths import step
from sph.analysis.tapering import cosine_taper, kaiser_taper
import molecules, maps as MP, sphmap
OUT = step("04_small_molecules"); IN = step("02_prepare")
def weight(k, kind):
    return np.ones_like(k) if kind == "none" else (cosine_taper(k, k.max(), 0.9) if kind == "cosine" else kaiser_taper(k, k.max(), 8.0))
rows = []
for kind in ("none", "cosine", "kaiser"):
    for mol in ("urea", "glyala"):
        m = molecules.load(mol); R = {}
        for inp, suf in (("H", "_fullP1"), ("M", "_iamph_fullP1"), ("I", "_iam_fullP1")):
            x = gemmi.read_mtz_file(str(IN / f"{mol}{suf}.mtz")); k = 2 * np.pi / np.asarray(x.make_d_array(), float)
            arr = np.array(x, copy=True); arr[:, x.column_labels().index("F")] *= weight(k, kind); x.set_data(arr)
            d = OUT / "taper" / kind; d.mkdir(parents=True, exist_ok=True); src = d / f"{mol}_{inp}_in.mtz"; x.write_to_file(str(src))
            R[inp] = MP.descriptors(MP.loadmaps(sphmap.run(src, d, f"{mol}_{inp}", 6), f"{mol}_{inp}"), m["atoms"], m["cell"], m["bonds"])
        for key in R["H"]:
            rows.append(dict(taper=kind, molecule=mol, bond=key, in17=(mol, key) not in molecules.EXCLUDED,
                             **{f"incl_{i}": MP.acute(R[i][key]["angle"]) for i in "HMI"}, **{f"D_{i}": R[i][key]["D"] for i in "HMI"}))
        print(kind, mol, flush=True)
w = pd.DataFrame(rows)
w["maxdiff_inputs"] = w[["incl_H", "incl_M", "incl_I"]].max(1) - w[["incl_H", "incl_M", "incl_I"]].min(1)
w["ratio_HI"] = w.D_H / w.D_I; w["amp_part"] = (w.D_I - w.D_M) / w.D_I; w["phase_part"] = (w.D_M - w.D_H) / w.D_I
base = w[w.taper == "none"].set_index(["molecule", "bond"])
w["dincl_H_vs_none"] = [abs(r.incl_H - base.loc[(r.molecule, r.bond), "incl_H"]) for r in w.itertuples()]
w.to_csv(OUT / "taper_bonds.csv", index=False)
s = pd.DataFrame([dict(taper=kind, n=len(g), max_incl_change_H=g.dincl_H_vs_none.max(), max_diff_inputs=g.maxdiff_inputs.max(),
                       ratio_min=g.ratio_HI.min(), ratio_max=g.ratio_HI.max(), mean_amp=100 * g.amp_part.mean(), mean_phase=100 * g.phase_part.mean())
                  for kind, g in w[w.in17].groupby("taper", sort=False)])
s.to_csv(OUT / "taper_summary.csv", index=False); print(s.round(3).to_string(index=False))
