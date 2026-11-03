"""Sensitivity of the bond descriptors to the sampling window along the bond (Supplementary Table 'Bond descriptors',
column 'windows'): five points at 30-70 % (paper), 25-75 %, 35-65 %, and 9 or 41 points at 30-70 %.
Output: WORK/04_small_molecules/window_sensitivity.json"""
import _paths, json, numpy as np
from _paths import step
import molecules, maps as MP
OUT = step("04_small_molecules")
WIN = {"30-70/5 (paper)": (0.3, 0.7, 5), "25-75/5": (0.25, 0.75, 5), "35-65/5": (0.35, 0.65, 5), "30-70/9": (0.3, 0.7, 9), "30-70/41": (0.3, 0.7, 41)}
out = {}
for mol in ("urea", "glyala"):
    m = molecules.load(mol); A = m["cell"]
    M = {t: MP.load_set(mol, t) for t in ("obs", "iamph", "iam")}
    for a, b in m["bonds"]:
        key = f"{mol} {a}-{b}"; out[key] = {}
        for w, (lo, hi, n) in WIN.items():
            r = {t: MP.descriptors(M[t], m["atoms"], A, [(a, b)], lo, hi, n)[f"{a}-{b}"] for t in M}
            out[key][w] = dict(har_over_iam=r["obs"]["D"] / r["iam"]["D"], iamph_over_iam=r["iamph"]["D"] / r["iam"]["D"],
                               D={t: r[t]["D"] for t in r}, angle={t: MP.acute(r[t]["angle"]) for t in r})
        rr = [out[key][w]["har_over_iam"] for w in WIN]
        print(f"{key:18s} paper {rr[0]:.3f}  windows {min(rr):.2f}-{max(rr):.2f}")
json.dump(out, open(OUT / "window_sensitivity.json", "w"), indent=1)
