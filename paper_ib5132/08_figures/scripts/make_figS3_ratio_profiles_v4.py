# v3 (2 Nov 2026): pointwise dipolar ratio
# Supplementary Fig. S3 (31 Oct 2026): former main-text Fig. 1B (urea pointwise gradient ratio; renamed 30 Oct 2026) and 1E (Gly-Ala profiles
# along C4-O3 and C4-N1, now from the new Gly-Ala HAR data). Grey contours: observed rho0 at 0.2, 0.5, 1, 2, 5, 10.
import numpy as np, matplotlib, json
import matplotlib.patheffects as pe
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import map_coordinates
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5})
Du = np.load('urea_plane_section.npz'); Dg = np.load('glyala_peptide_plane_section.npz'); LEV = [0.2, 0.5, 1, 2, 5, 10]
atU = {k[5:]: Du[k] for k in Du.files if k.startswith('atom_')}; atG = {k[5:]: Dg[k] for k in Dg.files if k.startswith('atom_')}
COL = {'C': '0.35', 'O': '#d62728', 'N': '#1f4e9c', 'H': '0.75'}
fig = plt.figure(figsize=(6.5, 2.9), constrained_layout=True); gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.1])
ax = fig.add_subplot(gs[:, 0]); U, V = np.meshgrid(Du['u'], Du['v'])
L0h, L0i = Du['urea_L0'], Du['urea_IAM_L0']; mh = np.hypot(Du['urea_du'], Du['urea_dv']); mi = np.hypot(Du['urea_IAM_du'], Du['urea_IAM_dv'])
r = np.ma.array((mh / L0h) / (mi / L0i), mask=(L0i < 0.15) | (L0h < 0.15))
im = ax.pcolormesh(U, V, r, cmap='PuOr_r', norm=TwoSlopeNorm(1.0, 0.5, 1.5), shading='auto', rasterized=True); ax.set_facecolor('0.88')
ax.contour(U, V, L0h, levels=LEV, colors='0.5', linewidths=0.4)
for a, b in [('C', 'O'), ('C', 'N1'), ('C', 'N2'), ('N1', 'H11'), ('N1', 'H12'), ('N2', 'H21'), ('N2', 'H22')]:
    p, q = atU[a], atU[b]; ax.plot([p[0], q[0]], [p[1], q[1]], color='0.2', lw=1, zorder=3)
for k, p in atU.items(): ax.scatter(*p, s=36 if k[0] != 'H' else 14, c=COL[k[0]], edgecolors='k', linewidths=0.4, zorder=4)
ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.0, 2.6)
ax.set_title('urea: pointwise dipolar ratio, observed / IAM', fontsize=7.5); ax.text(0.02, 0.97, 'A', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')
cb = fig.colorbar(im, ax=ax, location='bottom', shrink=0.8, pad=0.01); cb.set_label('pointwise ratio (grey: $\\widetilde{\\rho}_{00}<0.15$)', fontsize=7)
ug, vg = Dg['u'], Dg['v']; full = lambda t: np.sqrt(Dg[f'{t}_du'] ** 2 + Dg[f'{t}_dv'] ** 2 + Dg[f'{t}_dn'] ** 2)
def prof(arr, p, q, n=161):
    t = np.linspace(0, 1, n); pts = p[None, :2] + t[:, None] * (q[:2] - p[:2])[None, :]
    return t, map_coordinates(arr, [(pts[:, 1] - vg[0]) / (vg[1] - vg[0]), (pts[:, 0] - ug[0]) / (ug[1] - ug[0])], order=1)
axE = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]
for (a_, b_), cc in {('C4', 'O3'): '#d62728', ('C4', 'N1'): '#1f4e9c'}.items():
    p, q = atG[a_], atG[b_]; t, r0h = prof(Dg['glyala_L0'], p, q); _, r0i = prof(Dg['glyala_IAM_L0'], p, q); _, dh = prof(full('glyala'), p, q); _, di = prof(full('glyala_IAM'), p, q)
    axE[0].plot(t, r0h - r0i, color=cc, lw=1.2, label=f'{a_}–{b_}'); axE[1].plot(t, dh - di, color=cc, lw=1.2)
    i5 = [int(round(x * (len(t) - 1))) for x in np.linspace(0.3, 0.7, 5)]
    axE[0].plot(t[i5], (r0h - r0i)[i5], 'o', ms=3.5, mfc='w', mec=cc, mew=0.9); axE[1].plot(t[i5], (dh - di)[i5], 'o', ms=3.5, mfc='w', mec=cc, mew=0.9)
for a_ in axE: a_.axvspan(0.3, 0.7, color='0.9', zorder=0); a_.axhline(0, color='0.5', lw=0.6); a_.set_xlim(0, 1)
axE[0].set_ylabel(r'$\Delta\widetilde{\rho}_{00}$ (map units)'); axE[1].set_ylabel(r'$|\mathbf{d}_{\rm obs}|-|\mathbf{d}_{\rm IAM}|$')
axE[0].set_xticklabels([]); axE[1].set_xlabel('fraction of the bond from C4'); axE[0].legend(frameon=False, fontsize=7, loc='lower center', ncol=2)
axE[0].set_title('Gly–Ala: observed − IAM along two bonds', fontsize=7.5); axE[0].text(0.02, 0.93, 'B', transform=axE[0].transAxes, fontsize=10, fontweight='bold', va='top')
fig.savefig('FigS3_ratio_profiles.pdf', dpi=600); fig.savefig('FigS3_ratio_profiles.png', dpi=200); print('ok')
