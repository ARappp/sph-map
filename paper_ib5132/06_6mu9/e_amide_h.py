"""Amide-H test for 6MU9 (Supplementary Table 'amide H', column 6MU9): deposited model (REFMAC, no H: FWT, FC_ALL)
against the re-refined riding-H model (b) (2FOFCWT_no_fill, F-model). Amide N sites from the deposited
coordinates. About 3 min. Output: WORK/06_6mu9/amide_h.json"""
import _paths, json, numpy as np
from _paths import step
import direct as D, protein as PR, amide
OUT = step("06_6mu9")
sites = amide.amide_N(PR.D6 / "6MU9_deposited.pdb"); X = np.array([s[1] for s in sites]); E = np.array([s[2] for s in sites])
INP = {"data_noH": ("6mu9_final.mtz", "FWT", "PHWT"), "model_noH": ("6mu9_final.mtz", "FC_ALL", "PHIC_ALL"),
       "data": ("rerefined_b_ridingH.mtz", "2FOFCWT_no_fill", "PH2FOFCWT_no_fill"), "model": ("rerefined_b_ridingH.mtz", "F-model", "PHIF-model")}
V = {k: amide.vectors(D.load(PR.D6 / f, a, p), X) for k, (f, a, p) in INP.items()}
res = amide.summary(V, E); print(len(sites), "amide N"); [print(k, v) for k, v in res.items()]
json.dump(res, open(OUT / "amide_h.json", "w"), indent=1)
