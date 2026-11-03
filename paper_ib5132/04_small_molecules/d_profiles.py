"""Profiles along Gly-Ala C4-O3 and C4-N1 of rho_00(obs) - rho_00(IAM) and |d_obs| - |d_IAM| (Fig. S3B and
Section 4.5). Output: WORK/04_small_molecules/profiles_glyala.json (summary) and profiles_glyala.npz (curves)."""
import _paths, json, numpy as np
from _paths import step
import molecules, maps as MP
OUT = step("04_small_molecules")
m = molecules.load("glyala"); M = {t: MP.load_set("glyala", t) for t in ("obs", "iam")}
out = {}; curves = {"t": np.linspace(0, 1, 201)}; ts = curves["t"]
for a, b in (("C4", "O3"), ("C4", "N1")):
    fa, fb = np.array(m["atoms"][a]), np.array(m["atoms"][b])
    r0 = {t: np.array([MP.val(M[t]["L0"], fa + s * (fb - fa)) for s in ts]) for t in M}
    dn = {t: np.array([np.linalg.norm(MP.vec(M[t], fa + s * (fb - fa))) for s in ts]) for t in M}
    drho = r0["obs"] - r0["iam"]; dd = dn["obs"] - dn["iam"]; inner = (ts >= 0.15) & (ts <= 0.85)
    curves[f"{a}-{b}_drho00"] = drho; curves[f"{a}-{b}_dnorm"] = dd
    out[f"{a}-{b}"] = dict(max_drho=float(drho[inner].max()), min_dd=float(dd[inner].min()), max_dd=float(dd[inner].max()),
                           rho_min_mid_obs=float(r0["obs"][(ts > 0.3) & (ts < 0.7)].min()), rho_mid_obs=float(r0["obs"][100]),
                           frac_dd_neg=float((dd[(ts >= 0.2) & (ts <= 0.8)] < 0).mean()))
np.savez_compressed(OUT / "profiles_glyala.npz", **curves)
json.dump(out, open(OUT / "profiles_glyala.json", "w"), indent=1); print(json.dumps(out, indent=1))
