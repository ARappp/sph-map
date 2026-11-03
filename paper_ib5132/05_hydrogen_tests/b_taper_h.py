"""Hydrogen tests with tapered coefficients (Supplementary Table 'Tapering', hydrogen part): HAR phases, H-free
phases (P3) and P2 (20 random sets), field averaged over spheres from FFT maps; taper none, cosine (0.9), Kaiser (8).
Output (WORK/05_hydrogen_tests): taper_h_by_site.csv, taper_h_summary.csv"""
import _paths, numpy as np, pandas as pd
from _paths import step
from sph.analysis.tapering import cosine_taper, kaiser_taper
import direct as D, htest as H
OUT = step("05_hydrogen_tests")
def weight(k, kind):
    return np.ones_like(k) if kind == "none" else (cosine_taper(k, k.max(), 0.9) if kind == "cosine" else kaiser_taper(k, k.max(), 8.0))
rows = []
for ds in ("urea", "glyala"):
    c = H.context(ds); coI = c["coI"]; k = 2 * np.pi * np.linalg.norm(coI.q, axis=1)
    for kind in ("none", "cosine", "kaiser"):
        w = weight(k, kind)
        def S(C, f0): return D.with_f000(coI, w * C, f0, kind)
        V = {n: H.fftA(S(C, f0), c["X"], c["Ainv"]) for n, (C, f0) in dict(data=(c["CD"], c["F0"]), noH=(c["CnoH"], c["F0"] - c["nH_e"]),
                                                                           P3=(np.abs(c["CD"]) * c["phnoH"], c["F0"])).items()}
        P2 = [H.fftA(S(c["CD"] - c["FHy"] + np.abs(c["FHy"]) * D.random_phases(coI.hkl, 1000 + sd), c["F0"]), c["X"], c["Ainv"]) for sd in range(20)]
        for key in V["data"]:
            for j, (lab, e) in enumerate(zip(c["labels"], c["E"])):
                m0 = V["noH"][key][j]
                rows.append(dict(taper=kind, dataset=ds, site=lab, probe=key, real=H.ang(V["data"][key][j] - m0, e), P3=H.ang(V["P3"][key][j] - m0, e),
                                 P2_median=float(np.median([H.ang(p[key][j] - m0, e) for p in P2]))))
        print(ds, kind, flush=True)
t = pd.DataFrame(rows); t.to_csv(OUT / "taper_h_by_site.csv", index=False)
s = t.groupby(["dataset", "probe", "taper"], sort=False)[["real", "P3", "P2_median"]].median().reset_index(); s.to_csv(OUT / "taper_h_summary.csv", index=False)
print(s.round(1).to_string(index=False))
