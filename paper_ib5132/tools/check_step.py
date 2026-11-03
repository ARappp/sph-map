"""Compare the output of a step with the reference values of the paper calculations.
Usage: python3 tools/check_step.py 04_small_molecules
       python3 tools/check_step.py 04_small_molecules --write   (writes the reference file; maintainers only)
MTZ and CCP4 files (MTZ amplitude-phase pairs as complex coefficients F cos(phi), F sin(phi), so that +180/-180 and the phase of a zero amplitude do not matter): number of values, sum, sum of squares, min and max must agree to 1e-5 of the largest value.
JSON, CSV and NPZ files (NPZ: size, mean, r.m.s., min, max of each array): every number must agree to 2e-3 + 2e-4 |value|
(at least 50 times finer than the precision quoted in the paper); text entries must be identical.
(Results can differ in the last digits between computers, numpy and FFT builds.)"""
import sys, json, csv
from pathlib import Path
import numpy as np, gemmi
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); from config import WORK, ROOT
name = sys.argv[1]; d = WORK / name; ref = ROOT / name / "reference_values.json"
def num(x):
    try: return float(x)
    except (TypeError, ValueError): return None
def flat(o, p=""):
    if isinstance(o, dict): return {k2: v for k, x in o.items() for k2, v in flat(x, f"{p}/{k}").items()}
    if isinstance(o, list): return {k2: v for i, x in enumerate(o) for k2, v in flat(x, f"{p}[{i}]").items()}
    return {p: o}
def stats(p):
    if p.suffix in (".mtz", ".ccp4"):
        if p.suffix == ".mtz":                  # phases as cos and sin: +180 and -180 are the same phase
            m = gemmi.read_mtz_file(str(p)); a = np.array(m.array, dtype=float); cols = []
            for j, c in enumerate(m.columns):
                if c.type == "P" and j and m.columns[j - 1].type == "F":    # amplitude-phase pair -> complex coefficient
                    cols[-1] = a[:, j - 1] * np.cos(np.radians(a[:, j])); cols.append(a[:, j - 1] * np.sin(np.radians(a[:, j])))
                elif c.type == "P": cols += [np.cos(np.radians(a[:, j])), np.sin(np.radians(a[:, j]))]
                else: cols.append(a[:, j])
            a = np.concatenate(cols)
        else: a = np.array(gemmi.read_ccp4_map(str(p)).grid, copy=False)
        a = a[np.isfinite(a)].astype(float); return dict(kind="array", v=[a.size, a.sum(), (a * a).sum(), a.min(), a.max()])
    if p.suffix == ".json": return dict(kind="table", v=flat(json.load(open(p))))
    if p.suffix == ".npz":                      # per array: size, mean, r.m.s., min, max
        z = np.load(p); v = {}
        for k in z.files:
            a = z[k]
            if a.dtype.kind in "fi" and a.size: a = a.astype(float).ravel(); v[k] = [a.size, a.mean(), np.sqrt((a * a).mean()), a.min(), a.max()]
        return dict(kind="table", v=flat(v))
    rows = list(csv.reader(open(p))); return dict(kind="table", v={f"r{i}c{j}": x for i, r in enumerate(rows) for j, x in enumerate(r)})
files = sorted(p for p in d.rglob("*") if p.suffix in (".mtz", ".ccp4", ".json", ".csv", ".npz") and not p.name.endswith(".manifest.json"))
S = {str(p.relative_to(d)): stats(p) for p in files}
if "--write" in sys.argv:
    json.dump(S, open(ref, "w"), indent=0, default=float); print(f"wrote {ref} ({len(S)} files)"); sys.exit()
R = json.load(open(ref)); bad = 0
for k, r in R.items():
    if k not in S: print("MISSING ", k); bad += 1; continue
    s = S[k]
    if r["kind"] == "array":
        n, m = r["v"][0], max(abs(r["v"][3]), abs(r["v"][4]), 1e-30)
        tol = [0, 1e-5 * n * m, 1e-5 * n * m * m, 1e-5 * m, 1e-5 * m]
        ok = s["v"][0] == n and all(abs(x - y) <= t for x, y, t in zip(s["v"][1:], r["v"][1:], tol[1:]))
        if not ok: print("DIFFERS ", k)
    else:
        diffs = []
        for key, y in r["v"].items():
            x = s["v"].get(key, "<missing>"); fx, fy = num(x), num(y)
            if fx is not None and fy is not None:
                if not (abs(fx - fy) <= 2e-3 + 2e-4 * abs(fy) or (np.isnan(fx) and np.isnan(fy))): diffs.append((key, x, y))
            elif str(x) != str(y): diffs.append((key, x, y))
        ok = not diffs
        if not ok: print("DIFFERS ", k, diffs[:3], f"({len(diffs)} entries)")
    bad += not ok
print(f"{name}: {len(R) - bad} of {len(R)} files agree with the reference values" + ("" if bad else "  -> OK"))
sys.exit(1 if bad else 0)
