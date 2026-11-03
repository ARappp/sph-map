# Fig. 1 v4 (31 Oct 2026; single column, 2 x 2): (A) urea, observed - IAM isotropic channel; (B) urea H-omit difference;
# (C) Gly-Ala (new HAR data, Tonto 26.10.18), observed - IAM; (D) Gly-Ala H-omit difference. One colour bar per column,
# colour range = largest |value| in the two panels of the column; grey contours of the observed rho0 at 0.2, 0.5, 1, 2, 5, 10.
# Old panels B (pointwise ratio) and E (Gly-Ala profiles) -> Supplementary Fig. S3 (make_figS3_ratio_profiles.py).
import numpy as np, matplotlib, json
import matplotlib.patheffects as pe
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7})
Du = np.load('urea_plane_section.npz'); Hu = np.load('smallmol_homit_sections.npz'); Zu = np.load('smallmol_noH_coefficients.npz')
Dg = np.load('glyala_peptide_plane_section.npz'); Hg = np.load('glyala_homit_sections.npz'); infoG = json.load(open('glyala_sections_info.json'))
LEV = [0.2, 0.5, 1, 2, 5, 10]
kLu = json.load(open('smallmol_homit_sections.json'))['urea']['kL']
C = Zu['urea_Cobs'] - Zu['urea_CnoH']; ok = np.linalg.norm(Zu['urea_q'], axis=1) > 0; RMS = {'urea': kLu * np.sqrt((abs(C[ok]) ** 2).sum()), 'glyala': infoG['homit_rms']}
COL = {'C': '0.35', 'O': '#d62728', 'N': '#1f4e9c', 'H': '0.75'}
atU = {k[5:]: Du[k] for k in Du.files if k.startswith('atom_')}
bondsU = [('C', 'O'), ('C', 'N1'), ('C', 'N2'), ('N1', 'H11'), ('N1', 'H12'), ('N2', 'H21'), ('N2', 'H22')]
atG = {k[5:]: Dg[k] for k in Dg.files if k.startswith('atom_')}
bondsG = [('C4', 'O3'), ('C4', 'N1'), ('C4', 'C5'), ('N1', 'C2'), ('N1', 'H1N1'), ('C5', 'N2'), ('C5', 'H5A'), ('C5', 'H5B'), ('N2', 'H1N2'), ('N2', 'H2N2'), ('N2', 'H3N2'),
          ('C2', 'C1'), ('C2', 'C3'), ('C2', 'H2'), ('C1', 'O1'), ('C1', 'O2'), ('C3', 'H3A'), ('C3', 'H3B'), ('C3', 'H3C')]
def molU(ax, labels=False, showH=True):
    for a, b in bondsU:
        if not showH and 'H' in a + b: continue
        p, q = atU[a], atU[b]; ax.plot([p[0], q[0]], [p[1], q[1]], color='0.2', lw=0.9, zorder=3)
    for k, p in atU.items():
        if k[0] == 'H' and not showH: ax.scatter(*p, s=16, facecolors='none', edgecolors='k', linewidths=0.8, zorder=4); continue
        ax.scatter(*p, s=26 if k[0] != 'H' else 10, c=COL[k[0]], edgecolors='k', linewidths=0.4, zorder=4)
        if labels and k[0] != 'H': ax.text(p[0] + 0.15, p[1] + 0.12, k, fontsize=6, zorder=5, path_effects=[pe.withStroke(linewidth=1.5, foreground='w')])
def molG(ax, labels=False, showH=True, NEAR=0.45):
    for a, b in bondsG:
        if not showH and ('H' in a or 'H' in b): continue
        p, q = atG[a], atG[b]
        if max(abs(p[2]), abs(q[2])) > 1.6: continue
        near = abs(p[2]) < NEAR and abs(q[2]) < NEAR
        ax.plot([p[0], q[0]], [p[1], q[1]], color='0.2' if near else '0.55', lw=0.9 if near else 0.6, ls='-' if near else (0, (2, 1.5)), zorder=3)
    for k, p in atG.items():
        if abs(p[2]) > 1.6: continue
        e = k[0]; near = abs(p[2]) < NEAR
        if e == 'H' and not showH:
            ax.scatter(p[0], p[1], s=16 if near else 9, facecolors='none', edgecolors='k', linewidths=0.8 if near else 0.5, zorder=4); continue
        s = (26 if e != 'H' else 10) * (1 if near else 0.5)
        ax.scatter(p[0], p[1], s=s, c=COL[e] if near else 'w', edgecolors=COL[e] if not near else 'k', linewidths=0.4 if near else 0.7, zorder=4)
        if labels and near and e != 'H' and Dg['u'][0] < p[0] < Dg['u'][-1] - 0.3: ax.text(p[0] + 0.15, p[1] + 0.12, k, fontsize=6, zorder=5, path_effects=[pe.withStroke(linewidth=1.5, foreground='w')])
# data
uU, vU = np.meshgrid(Du['u'], Du['v']); dU = Du['urea_L0'] - Du['urea_IAM_L0']
uG, vG = np.meshgrid(Dg['u'], Dg['v']); dG = Dg['glyala_L0'] - Dg['glyala_IAM_L0']
def hom(H, ds): 
    U, V = np.meshgrid(H[ds + '_u'], H[ds + '_v'])
    return U, V, (H[ds + '_obs_L0'] - H[ds + '_noH_L0']) / RMS[ds], H[ds + '_obs_du'] - H[ds + '_noH_du'], H[ds + '_obs_dv'] - H[ds + '_noH_dv'], H[ds + '_obs_L0']
hU = hom(Hu, 'urea'); hG = hom(Hg, 'glyala')
# urea window and Gly-Ala window used for the colour range
wU = (uU > -2.4) & (uU < 2.4) & (vU > -2.0) & (vU < 2.6); wUh = (hU[0] > -2.4) & (hU[0] < 2.4) & (hU[1] > -2.0) & (hU[1] < 2.6)
LIM1 = max(abs(dU[wU]).max(), abs(dG).max()); LIM2 = max(abs(hU[2][wUh]).max(), abs(hG[2]).max())
fig = plt.figure(figsize=(3.46, 3.75), constrained_layout=True)
gs = fig.add_gridspec(3, 2, height_ratios=[1, 0.82, 0.07])
axA, axB, axC, axD = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
n1 = TwoSlopeNorm(0, -LIM1, LIM1); n2 = TwoSlopeNorm(0, -LIM2, LIM2)
imA = axA.pcolormesh(uU, vU, dU, cmap='RdBu_r', norm=n1, shading='auto', rasterized=True)
axA.contour(uU, vU, Du['urea_L0'], levels=LEV, colors='0.5', linewidths=0.35); molU(axA, labels=True)
axC.pcolormesh(uG, vG, dG, cmap='RdBu_r', norm=n1, shading='auto', rasterized=True)
axC.contour(uG, vG, Dg['glyala_L0'], levels=LEV, colors='0.5', linewidths=0.35); molG(axC, labels=True)
def homit(ax, h, mol, step):
    U, V, dL, du, dv, L0 = h
    im = ax.pcolormesh(U, V, dL, cmap='RdBu_r', norm=n2, shading='auto', rasterized=True)
    ax.contour(U, V, L0, levels=LEV, colors='0.5', linewidths=0.35)
    sl = (slice(step // 2, None, step), slice(step // 2, None, step))
    ax.quiver(U[sl], V[sl], du[sl], dv[sl], color='k', width=0.006, alpha=0.8, zorder=5); mol(ax); return im
imB = homit(axB, hU, lambda a: molU(a, showH=False), 16)
homit(axD, hG, lambda a: molG(a, showH=False), 16)
for ax, l in zip((axA, axB, axC, axD), 'ABCD'):
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    if ax in (axA, axB): ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.0, 2.6)
    else: ax.set_xlim(Dg['u'][0], Dg['u'][-1]); ax.set_ylim(Dg['v'][0], Dg['v'][-1])
    ax.text(0.03, 0.97, l, transform=ax.transAxes, fontsize=9, fontweight='bold', va='top', path_effects=[pe.withStroke(linewidth=2, foreground='w')])
axA.set_title('observed − IAM', fontsize=7.5); axB.set_title('H-omit difference', fontsize=7.5)
axA.set_ylabel('urea', fontsize=7.5); axC.set_ylabel('Gly–Ala', fontsize=7.5)
axA.plot([-2.2, -1.2], [-1.8, -1.8], 'k', lw=1.3); axA.text(-1.7, -1.7, '1 Å', ha='center', va='bottom', fontsize=6, path_effects=[pe.withStroke(linewidth=1.5, foreground='w')])
c1 = fig.colorbar(imA, cax=fig.add_subplot(gs[2, 0]), orientation='horizontal'); c1.set_label(r'$\Delta\widetilde{\rho}_{00}$ (map units)', fontsize=6.5); c1.ax.tick_params(labelsize=6)
c2 = fig.colorbar(imB, cax=fig.add_subplot(gs[2, 1]), orientation='horizontal'); c2.set_label(r'$\Delta\widetilde{\rho}_{00}^{-\mathrm{H}}$ (r.m.s. units)', fontsize=6.5); c2.ax.tick_params(labelsize=6)
fig.savefig('Fig1_smallmol.pdf', dpi=600); fig.savefig('Fig1_smallmol.png', dpi=300)
json.dump(dict(LIM1=float(LIM1), LIM2=float(LIM2), RMS=RMS, max_hom_urea=float(hU[2][wUh].max()), max_hom_gly=float(hG[2].max())), open('fig1_info.json', 'w'), indent=1)
print(dict(LIM1=LIM1, LIM2=LIM2, RMS=RMS))
