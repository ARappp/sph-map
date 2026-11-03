"""Sensitivity of the bond descriptors to the radial weight (Supplementary Table 'Radial weight'): sph-map with
shell weights (80 linear or 80 logarithmic shells) instead of the uniform weight used in the paper.
Runs sph-map 12 times (about 1 min). Output: WORK/04_small_molecules/radial_weight/{linear,log}/ (maps) and
radial_weight.json"""
import _paths, json
from _paths import step
import molecules, maps as MP, sphmap
OUT = step("04_small_molecules"); IN = step("02_prepare")
SRC = {"obs": "", "iamph": "_iamph", "iam": "_iam"}
out = {"uniform": {}}
for mode in ("linear", "log"):
    out[mode] = {}; d = OUT / "radial_weight" / mode
    for mol in ("urea", "glyala"):
        m = molecules.load(mol); R = {}
        for s, suf in SRC.items():
            sphmap.run(IN / f"{mol}{suf}_fullP1.mtz", d, f"{mol}_{s}", 6, ["--shell-weights", "on", "--shell-mode", mode, "--n-shell", "80"])
            R[s] = MP.descriptors(MP.loadmaps(d / f"{mol}_{s}_maps", f"{mol}_{s}"), m["atoms"], m["cell"], m["bonds"])
        U = {s: MP.descriptors(MP.load_set(mol, s), m["atoms"], m["cell"], m["bonds"]) for s in SRC}
        for k in R["obs"]:
            for tag, X in ((mode, R), ("uniform", U)):
                out[tag][f"{mol} {k}"] = dict(angle={s: MP.acute(X[s][k]["angle"]) for s in X}, D={s: X[s][k]["D"] for s in X},
                                              ratio=X["obs"][k]["D"] / X["iam"][k]["D"])
print("bond               angle(HAR) uniform linear log | D_HAR/D_IAM uniform linear log")
for k in out["uniform"]:
    print("%-18s %16.1f %6.1f %5.1f | %19.2f %6.2f %5.2f" % (k, *(out[t][k]["angle"]["obs"] for t in ("uniform", "linear", "log")),
                                                           *(out[t][k]["ratio"] for t in ("uniform", "linear", "log"))))
json.dump(out, open(OUT / "radial_weight.json", "w"), indent=1)
