"""Height of the H-omit difference density Delta rho^-H (observed minus H-free IAM, F(000) omitted) at the H atoms
of urea and Gly-Ala, in units of its r.m.s. over the cell (main text Section 4.2): value at the H position and at the
nearest maximum (Nelder-Mead), and the distance between them. Output: WORK/05_hydrogen_tests/homit_peaks.json"""
import _paths, json, numpy as np
from scipy.optimize import minimize
from _paths import step
import htest as H
OUT = step("05_hydrogen_tests"); out = {}
for ds in ("urea", "glyala"):
    c = H.context(ds); cr = c["cr"]; q = c["coI"].q; C = c["CD"] - c["CnoH"]; ok = np.linalg.norm(q, axis=1) > 0
    rms = np.sqrt((abs(C[ok]) ** 2).sum())          # Parseval, same (unit-free) normalisation as the synthesis below
    f = lambda x: -(np.exp(2j * np.pi * (q @ x)) @ C).real
    out[ds] = []
    for a in cr.atoms:
        if a.element != "H" or not a.label.endswith("#1"): continue
        v0 = -f(a.cart); r = minimize(f, a.cart, method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-6))
        out[ds].append(dict(H=a.label[:-2], at_H_rms=float(v0 / rms), peak_rms=float(-r.fun / rms), peak_distance_A=float(np.linalg.norm(r.x - a.cart))))
    print(ds, [(e["H"], round(e["at_H_rms"], 2), round(e["peak_rms"], 2), round(e["peak_distance_A"], 2)) for e in out[ds]])
json.dump(out, open(OUT / "homit_peaks.json", "w"), indent=1)
