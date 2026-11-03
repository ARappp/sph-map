"""The 6MU9 row of the Supplementary PDB-REDO table: median (r0 = 0.6 and 0.3 A) and 90th percentile of the observed-model
angle, fraction of sites above 15 deg, fraction of sites with an mFo-DFc peak >= 3 sigma within 1.6 A (DELFWT, gemmi FFT
sample rate 3, 0.2 A lattice) for the 5 % of sites with the largest angles and for 40 random sites (rng 0), and the
difference density at the ideal amide-H positions. Needs a_carbonyl_models.py and f_aromatics.py.
Output: WORK/06_6mu9/pdbredo_table_row.json"""
import _paths, json, csv, numpy as np, gemmi
from _paths import step
import protein as PR
from lattice import ball
OUT = step("06_6mu9")
rows = [r for r in csv.DictReader(open(OUT / "carbonyl_models.csv")) if r["model"] == "deposited"]
a6 = {r["site"]: float(r["angle_obs_model_deg"]) for r in rows if r["filter"] == "r0_0.6"}
a3 = np.array([float(r["angle_obs_model_deg"]) for r in rows if r["filter"] == "r0_0.3"]); v6 = np.array(list(a6.values()))
names, O, _ = PR.carbonyl_sites(PR.D6 / "6MU9_deposited.pdb"); Opos = dict(zip(names, O))
mt = gemmi.read_mtz_file(str(PR.D6 / "6mu9_final.mtz")); dg = mt.transform_f_phi_to_map("DELFWT", "PHDELWT", sample_rate=3); drms = np.array(dg, copy=False).std()
sh = ball(1.6, 0.2)
peak = lambda x: max(dg.interpolate_value(gemmi.Position(*(x + o))) for o in sh) / drms
top = sorted(a6, key=lambda k: -a6[k])[:max(5, len(a6) // 20)]; tp = np.array([peak(Opos[k]) for k in top])
rnd = np.random.default_rng(0).choice(list(a6), size=40, replace=False); rp = np.array([peak(Opos[k]) for k in rnd])
h = np.array(json.load(open(OUT / "aromatics_stats.json"))["amideH"])
out = dict(n_CO=len(v6), median_r06=float(np.median(v6)), median_r03=float(np.median(a3)), p90_r06=float(np.percentile(v6, 90)),
           pct_gt15=float((v6 > 15).mean() * 100), top5pct_peak_ge3_pct=float((tp >= 3).mean() * 100), random_peak_ge3_pct=float((rp >= 3).mean() * 100),
           amideH_mean_rms=float(h.mean()), amideH_n=len(h), amideH_gt3_pct=float((h > 3).mean() * 100))
print(json.dumps(out, indent=1)); json.dump(out, open(OUT / "pdbredo_table_row.json", "w"), indent=1)
