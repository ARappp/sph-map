"""Largest mFo-DFc value within 1.6 A of the carbonyl O of Lys299 and Lys75 for the four 6MU9 models, in units of the
r.m.s. of the difference map (Supplementary Table 'reference models', last column; main text Section 4.3), and the
environment of O75 (B factor, alternate conformers within 4 A). Difference maps: DELFWT (REFMAC) or FOFCWT (Phenix),
gemmi FFT with sample rate 4; maximum over a 0.1 A cubic lattice. Output: WORK/06_6mu9/difference_peaks.json"""
import _paths, json, numpy as np, gemmi
from _paths import step
import protein as PR
from lattice import ball
OUT = step("06_6mu9"); out = {}
off = list(ball(1.6, 0.1))
for tag, (mtz, _, pdb, (fd, pd_)) in PR.MODELS.items():
    st = gemmi.read_structure(str(PR.D6 / pdb)); m = gemmi.read_mtz_file(str(PR.D6 / mtz))
    g = m.transform_f_phi_to_map(fd, pd_, sample_rate=4); rms = float(np.array(g, copy=False).std()); out[tag] = {}
    for num in (299, 75):
        O = [a for a in [r for r in st[0]["A"] if r.seqid.num == num][0] if a.name == "O"][0]
        v = [(g.interpolate_value(gemmi.Position(O.pos.x + d[0], O.pos.y + d[1], O.pos.z + d[2])), np.linalg.norm(d)) for d in off]
        best = max(v); out[tag][f"O{num}"] = dict(peak_rms=round(best[0] / rms, 2), distance_A=round(float(best[1]), 2))
    print(tag, out[tag], flush=True)
st = gemmi.read_structure(str(PR.D6 / "6MU9_deposited.pdb")); st.setup_entities(); ns = gemmi.NeighborSearch(st[0], st.cell, 5).populate()
Bs = [a.b_iso for ch in st[0] for r in ch for a in r if a.name == "O" and r.het_flag != "H"]; out["deposited_environment"] = {"median_B_carbonyl_O": round(float(np.median(Bs)), 2)}
for num in (75, 299):
    r = [r for r in st[0]["A"] if r.seqid.num == num][0]; O = [a for a in r if a.name == "O"][0]
    alts = sorted({f"{mk.to_cra(st[0]).residue.name}{mk.to_cra(st[0]).residue.seqid.num}" for mk in ns.find_atoms(O.pos, "\0", radius=4.0) if mk.to_cra(st[0]).atom.altloc != "\0"})
    out["deposited_environment"][f"O{num}"] = dict(B=O.b_iso, altloc_residues_within_4A=alts)
print(out["deposited_environment"])
json.dump(out, open(OUT / "difference_peaks.json", "w"), indent=1)
