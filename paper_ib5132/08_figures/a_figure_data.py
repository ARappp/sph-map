"""Step 08a: data for the figures, from the results of steps 03-06. Writes WORK/08_figures/data/:
  urea_plane_section.npz, glyala_peptide_plane_section.npz  sections of rho_00 and d (Figs 1, S2, S3)
  sections_SI_small.npz, sections_SI_6mu9_pep.npz            channel sections (Fig. S1; 6MU9 maps filtered here)
  smallmol_homit_sections.npz/.json, glyala_homit_sections.npz, glyala_sections_info.json,
  smallmol_noH_coefficients.npz                              H-omit sections (Fig. 1), exact synthesis
  smallmol_bonds.csv, window_sensitivity.json                 bond descriptors (Figs 2, S4)
  fig4_exact.json, fofc_dep/b/c.ccp4, *.pdb                   O299 region (Fig. 3)
  6mu9_sweep_exact.csv                                        resolution sweep (Figs 4, S5)
  fig6_*_deposited.*                                          aromatics (Figs 5, S6)
Section maps are sampled by trilinear interpolation of the sph-map maps; |d|^2 ('P1power') is either the square of
the interpolated field or, for Fig. S1, interpolated from the grid of |d|^2 (as noted below). About 5 min."""
import _paths, json, csv, shutil, numpy as np, gemmi
from _paths import INPUT, step
import molecules, maps as MP, htest as HT, protein as PR, sphmap
from lattice import ball
OUT = step("08_figures") / "data"; OUT.mkdir(exist_ok=True)
S03, S04, S06 = step("03_filter"), step("04_small_molecules"), step("06_6mu9")
def cart(mol):
    m = molecules.load(mol); cell = gemmi.UnitCell(*m["cell"], 90, 90, 90)
    return m, cell, {k: np.array(cell.orthogonalize(gemmi.Fractional(*f)).tolist()) for k, f in m["atoms"].items()}
def sample(G, cell, O, ex, ey, u, v):
    """G: {key: FloatGrid}; returns {key: (len(v), len(u))} sampled on the plane O + u ex + v ey."""
    out = {k: np.zeros((len(v), len(u))) for k in G}
    for i, vv in enumerate(v):
        for j, uu in enumerate(u):
            f = cell.fractionalize(gemmi.Position(*(O + uu * ex + vv * ey)))
            for k in G: out[k][i, j] = G[k].interpolate_value(f)
    return out
def power_grid(M):
    """grid of |d|^2 = d_x^2 + d_y^2 + d_z^2 (as the Cartesian 'P1power' maps)."""
    a = sum(np.array(M[c], copy=False).astype(np.float64) ** 2 for c in "xyz")
    g = gemmi.FloatGrid(a.astype(np.float32)); g.set_unit_cell(M["L0"].unit_cell); g.spacegroup = gemmi.SpaceGroup("P 1"); return g
def frame(O, a, b):
    ey = (a - O) / np.linalg.norm(a - O); w = b - O; ex = w - (w @ ey) * ey; ex /= np.linalg.norm(ex); return ex, ey, np.cross(ex, ey)
R = {}
# ---------------- urea, molecular plane (Figs 1, S2, S3): origin C, v along C->O, N1 at negative u
m, cell, X = cart("urea"); O = X["C"]; ex = np.array([-1.0, -1.0, 0.0]) / np.sqrt(2); ey = np.array([0.0, 0.0, 1.0]); n = np.cross(ex, ey)
u = np.arange(-2.6, 2.6 + 1e-9, 0.025); v = np.arange(-2.4, 2.9 + 1e-9, 0.025); U = {"u": u, "v": v}
two = lambda p: np.array([(p - O) @ ex, (p - O) @ ey])
mirror = lambda p: np.array([-p[0], p[1]])
at = {"O": two(X["O"]), "N1": two(X["N1"]), "C": two(X["C"]), "H11": two(X["H1"]), "H12": two(X["H3"])}
at.update({"N2": mirror(at["N1"]), "H21": mirror(at["H11"]), "H22": mirror(at["H12"])})
for k, p in at.items(): U["atom_" + k] = p
for tag, s in (("urea", "obs"), ("urea_IAM", "iam")):
    M = MP.load_set("urea", s); M["P1"] = power_grid(M); o = sample(M, cell, O, ex, ey, u, v); d3 = np.stack([o["x"], o["y"], o["z"]], -1)
    U[tag + "_L0"] = o["L0"]; U[tag + "_du"] = d3 @ ex; U[tag + "_dv"] = d3 @ ey; U[tag + "_dn"] = d3 @ n; U[tag + "_P1"] = o["P1"]
np.savez_compressed(OUT / "urea_plane_section.npz", **U); print("urea plane", flush=True)
uU, nU = (u, v, O, ex, ey, n), None
# ---------------- Gly-Ala peptide plane (Figs 1, S1, S2, S3): origin C4, v along C4->O3, N1 in the plane
m, cell, X = cart("glyala"); O = X["C4"]; ey = (X["O3"] - O) / np.linalg.norm(X["O3"] - O); w = X["N1"] - O; ex = w - (w @ ey) * ey; ex /= np.linalg.norm(ex); n = np.cross(ex, ey)
u = np.arange(-2.8, 3.4 + 1e-9, 0.025); v = np.arange(-2.4, 2.2 + 1e-9, 0.025); G = {"u": u, "v": v}
for k, p in X.items(): G["atom_" + k] = np.array([(p - O) @ ex, (p - O) @ ey, (p - O) @ n])
for tag, s in (("glyala", "obs"), ("glyala_IAM", "iam")):
    o = sample(MP.load_set("glyala", s), cell, O, ex, ey, u, v); d3 = np.stack([o["x"], o["y"], o["z"]], -1)
    G[tag + "_L0"] = o["L0"]; G[tag + "_du"] = d3 @ ex; G[tag + "_dv"] = d3 @ ey; G[tag + "_dn"] = d3 @ n
    G[tag + "_L1x"] = o["x"]; G[tag + "_L1y"] = o["y"]; G[tag + "_L1z"] = o["z"]; G[tag + "_P1power"] = (d3 ** 2).sum(-1)
np.savez_compressed(OUT / "glyala_peptide_plane_section.npz", **G); print("glyala plane", flush=True)
# ---------------- H-omit sections by exact synthesis, scaled to the observed map section (Fig. 1)
def synth_plane(q, C, P, chunk=1500):
    qn = np.linalg.norm(q, axis=1); ok = qn > 0; q, C = q[ok], C[ok]; qh = q / qn[ok][:, None]; P2 = P.reshape(-1, 3)
    L0 = np.zeros(len(P2)); d = np.zeros((len(P2), 3))
    for i in range(0, len(P2), chunk):
        E = np.exp(2j * np.pi * (P2[i:i + chunk] @ q.T)); L0[i:i + chunk] = (E @ C).real; d[i:i + chunk] = (E @ (C[:, None] * 1j * qh)).real
    return L0.reshape(P.shape[:-1]), d.reshape(P.shape[:-1] + (3,))
def homit(ds, D, key, O, ex, ey, n):
    ctx = HT.context(ds); q = ctx["coI"].q; uu, vv = np.meshgrid(D["u"], D["v"]); P = O + uu[..., None] * ex + vv[..., None] * ey
    r = {tag: synth_plane(q, C, P) for tag, C in (("obs", ctx["CD"]), ("iam", ctx["coI"].C), ("noH", ctx["CnoH"]))}
    ref = D[key + "_L0"]; kL = (r["obs"][0] * ref).sum() / (r["obs"][0] ** 2).sum(); c0 = np.corrcoef(r["obs"][0].ravel(), ref.ravel())[0, 1]
    du, dv = r["obs"][1] @ ex, r["obs"][1] @ ey; kd = (du * D[key + "_du"] + dv * D[key + "_dv"]).sum() / (du ** 2 + dv ** 2).sum()
    c1 = np.corrcoef(np.r_[du.ravel(), dv.ravel()], np.r_[D[key + "_du"].ravel(), D[key + "_dv"].ravel()])[0, 1]
    H = {f"{ds}_u": D["u"], f"{ds}_v": D["v"]}
    for tag in r:
        H[f"{ds}_{tag}_L0"] = r[tag][0] * kL; d = r[tag][1] * kd; H[f"{ds}_{tag}_du"] = d @ ex; H[f"{ds}_{tag}_dv"] = d @ ey; H[f"{ds}_{tag}_dn"] = d @ n
    C = ctx["CD"] - ctx["CnoH"]; ok = np.linalg.norm(q, axis=1) > 0
    info = dict(kL=float(kL), kd=float(kd), corr_L0=float(c0), corr_d=float(c1), homit_rms=float(kL * np.sqrt((abs(C[ok]) ** 2).sum())))
    return H, info, ctx
Hg, infoG, _ = homit("glyala", G, "glyala", O, ex, ey, n)
np.savez_compressed(OUT / "glyala_homit_sections.npz", **Hg); json.dump(infoG, open(OUT / "glyala_sections_info.json", "w"), indent=1)
u, v, O, ex, ey, n = uU
Hu, infoU, cU = homit("urea", U, "urea", O, ex, ey, n)
np.savez_compressed(OUT / "smallmol_homit_sections.npz", **Hu); json.dump({"urea": infoU}, open(OUT / "smallmol_homit_sections.json", "w"), indent=1)
np.savez(OUT / "smallmol_noH_coefficients.npz", urea_hkl=cU["coI"].hkl, urea_Cobs=cU["CD"], urea_Ciam=cU["coI"].C, urea_CnoH=cU["CnoH"], urea_q=cU["coI"].q)
print("H-omit", infoU, infoG, flush=True)
# ---------------- channel sections for Fig. S1 (|d|^2 interpolated from its grid)
SI = {}
def store(D, key, u, v, o, names, xyz):
    D[key + "_u"] = u; D[key + "_v"] = v
    for k, a in o.items(): D[f"{key}_{k}"] = a
    D[key + "_atomnames"] = np.array(names); D[key + "_atomxyz"] = xyz
m, cell, X = cart("urea"); O = X["C"]; ex, ey, n = frame(O, X["O"], X["N1"])
u = np.arange(-2.4, 2.4 + 1e-9, 0.04); v = np.arange(-2.0, 2.6 + 1e-9, 0.04)
names = ["O", "N1", "C", "H11", "H12", "N2", "H21", "H22"]
xyz = [[(X[a] - O) @ ex, (X[a] - O) @ ey, (X[a] - O) @ n] for a in ("O", "N1", "C", "H1", "H3")]
xyz = np.array(xyz + [[-p[0], p[1], p[2]] for p in (xyz[1], xyz[3], xyz[4])])
for tag, s in (("urea", "obs"), ("urea_IAM", "iam")):
    M = MP.load_set("urea", s); M["P1power"] = power_grid(M); o = sample(M, cell, O, ex, ey, u, v)
    store(SI, tag, u, v, {"L0": o["L0"], "L1x": o["x"], "L1y": o["y"], "L1z": o["z"], "P1power": o["P1power"]}, names, xyz)
gn = [k[5:] for k in G if k.startswith("atom_")]
for tag in ("glyala", "glyala_IAM"):
    store(SI, tag, G["u"], G["v"], {c: G[f"{tag}_{c}"] for c in ("L0", "L1x", "L1y", "L1z", "P1power")}, gn, np.array([G["atom_" + a] for a in gn]))
np.savez_compressed(OUT / "sections_SI_small.npz", **SI); print("SI small", flush=True)
# 6MU9: sph-map maps of FWT and FC_ALL (3 points per d_min), section through the Lys299-Gln300 peptide
D6 = INPUT / "6mu9"; MD = OUT / "6mu9_maps"
Mx = {"6mu9_FWT": MP.loadmaps(sphmap.run(D6 / "6mu9_final.mtz", MD, "6mu9_FWT", 3, amp="FWT", phase="PHWT"), "6mu9_FWT"),
      "6mu9_model": MP.loadmaps(sphmap.run(D6 / "6mu9_final.mtz", MD, "6mu9_model", 3, amp="FC_ALL", phase="PHIC_ALL"), "6mu9_model")}
st = gemmi.read_structure(str(D6 / "6MU9_deposited.pdb")); cell = st.cell; A = {}
for ch in st[0]:
    for r in ch:
        if r.name == "HOH": continue
        for a in r:
            if a.altloc in ("\0", "A"): A.setdefault((r.seqid.num, a.name), np.array(a.pos.tolist()))
Wt = {("W", 767): np.array(a.pos.tolist()) for ch in st[0] for r in ch if r.name == "HOH" and r.seqid.num == 767 for a in r}
O = A[(299, "C")]; ex, ey, n = frame(O, A[(299, "O")], A[(300, "N")]); u = np.arange(-3.0, 3.0 + 1e-9, 0.04); v = np.arange(-2.6, 2.6 + 1e-9, 0.04)
near = {k: p for k, p in {**A, **Wt}.items() if np.linalg.norm(p - O) < 6}
names = [f"{k[0]}:{k[1]}" for k in near]; xyz = np.array([[(p - O) @ ex, (p - O) @ ey, (p - O) @ n] for p in near.values()]); PP = {}
for t, M in Mx.items():
    M["P1power"] = power_grid(M); o = sample(M, cell, O, ex, ey, u, v)
    store(PP, t + "_pep299", u, v, {"L0": o["L0"], "L1x": o["x"], "L1y": o["y"], "L1z": o["z"], "P1power": o["P1power"]}, names, xyz)
    PP[t + "_pep299_frame"] = np.array([ex, ey, n])
np.savez_compressed(OUT / "sections_SI_6mu9_pep.npz", **PP); shutil.rmtree(MD); print("SI 6MU9", flush=True)
# ---------------- bond descriptors (Figs 2, S4)
CLS = {"C=O/C-O": "C=O", "C-N": "C-N", "N-H": "N-H", "C-H": "C-H", "C-C": "C-C"}
with open(S04 / "bond_descriptors.csv") as f, open(OUT / "smallmol_bonds.csv", "w", newline="") as g:
    w = csv.writer(g); w.writerow(["molecule", "bond", "class", "angle_HAR", "angle_IAMphase", "angle_IAM", "A_HAR", "A_IAMphase", "A_IAM"])
    for r in csv.DictReader(f): w.writerow([r["mol"], r["bond"], CLS[r["cls"]], r["ang_H"], r["ang_M"], r["ang_I"], r["D_H"], r["D_M"], r["D_I"]])
shutil.copyfile(S04 / "window_sensitivity.json", OUT / "window_sensitivity.json")
# ---------------- 6MU9 (Figs 3-5, S5, S6)
shutil.copyfile(S06 / "sweep_sites.csv", OUT / "6mu9_sweep_exact.csv")
for a, b in (("aromatics_sections.npz", "fig6_sections_deposited.npz"), ("aromatics_stats.json", "fig6_stats_deposited.json"), ("aromatics_cluster_se.json", "fig6_cluster_se_deposited.json")):
    shutil.copyfile(S06 / a, OUT / b)
F3 = {"dep": "deposited", "b": "b_ridingH", "c": "c_plusH_O299"}; out = {}
for tag, mk in F3.items():
    mtz, cols, pdb, (ff, pf) = PR.MODELS[mk]; shutil.copyfile(D6 / pdb, OUT / pdb)
    H, Cd, Cm, Fr, kc = PR.load(D6 / mtz, *cols); Mt = {"f": PR.mult(kc, 0.6)[0]}
    names, Oa, _ = PR.carbonyl_sites(D6 / pdb); resis = list(range(296, 305)) if tag == "dep" else [299]
    sel = [(nm, o) for nm, o in zip(names, Oa) if int(nm[1:-3]) in resis]; Xs = np.array([o for _, o in sel])
    Fd = PR.fields(H, Cd, Mt, Fr, Xs)["f"]; Fm = PR.fields(H, Cm, Mt, Fr, Xs)["f"]
    arrows = [[int(nm[1:-3]), o.tolist(), (Fd[i] / np.linalg.norm(Fd[i])).tolist(), (Fm[i] / np.linalg.norm(Fm[i])).tolist(),
               round(float(PR.ang(Fd[i:i + 1], Fm[i:i + 1])[0]), 1)] for i, (nm, o) in enumerate(sel)]
    mt = gemmi.read_mtz_file(str(D6 / mtz)); g = mt.transform_f_phi_to_map(ff, pf, sample_rate=4); rms = float(np.array(g, copy=False).std())
    cc = gemmi.Ccp4Map(); cc.grid = g; cc.update_ccp4_header(); cc.write_ccp4_map(str(OUT / f"fofc_{tag}.ccp4"))
    Pt = ball(1.6, 0.05); peaks = {}
    for r, o in [(int(nm[1:-3]), o) for nm, o in sel]:
        vv = np.array([g.interpolate_value(gemmi.Position(*(o + p))) for p in Pt]); i = int(vv.argmax())
        peaks[r] = dict(pos=(o + Pt[i]).tolist(), sigma=round(float(vv[i] / rms), 2), dist=round(float(np.linalg.norm(Pt[i])), 2))
    hpos = None
    for ch in gemmi.read_structure(str(D6 / pdb))[0]:
        for res in ch:
            if res.name == "HOX": hpos = [a.pos.tolist() for a in res if a.element.name == "H"][0]
    out[tag] = dict(arrows=arrows, fofc_rms=rms, peaks=peaks, H_O299=hpos); print(tag, [(x[0], x[4]) for x in arrows], flush=True)
json.dump(out, open(OUT / "fig4_exact.json", "w"), indent=1)
