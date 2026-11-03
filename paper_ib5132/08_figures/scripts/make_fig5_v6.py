# v6 (2 Nov 2026): panels A and B side by side (full width)
# Fig. 5 v4 (31 Oct 2026; single column): (A) difference density in the ring plane of Trp227 (no arrows; colour range to the
# largest |value| in the panel; contours of the observed rho0 at 25, 50, 75, 100 % of its maximum); (B) class means over 6MU9
# (as old panel G). The six sections of the old figure are Supplementary Fig. S6 (Fig6_aromatics.pdf of 28 Oct, unchanged).
import numpy as np, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from common import draw_atoms
plt.rcParams.update({'font.size': 7, 'font.family': 'DejaVu Sans'})
D = np.load('fig6_sections_deposited.npz', allow_pickle=True); J = json.load(open('fig6_stats_deposited.json')); CS = json.load(open('fig6_cluster_se_deposited.json')); rms = float(D['rms'])
fig = plt.figure(figsize=(7.0, 2.9), constrained_layout=True); gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25])
ax = fig.add_subplot(gs[0, 0]); k = 'r227_face'; u, v = D[k + '_u'], D[k + '_v']; U, V = np.meshgrid(u, v); dL = D[k + '_dL0'] / rms; lim = float(np.abs(dL).max())
im = ax.pcolormesh(U, V, dL, cmap='RdBu_r', norm=TwoSlopeNorm(0, -lim, lim), shading='auto', rasterized=True)
L0 = D[k + '_oL0']; ax.contour(U, V, L0, levels=np.linspace(0.25 * L0.max(), L0.max(), 4), colors='0.45', linewidths=0.4)
names = D[k + '_atomnames']; xyz = D[k + '_atomxyz']; sel = [i for i, nm in enumerate(names) if nm.startswith('227:')]
draw_atoms(ax, names[sel], xyz[sel], near=0.6, far=1.6, lw=0.8)
Hx = D[k + '_Hxyz']; nr = np.abs(Hx[:, 2]) < 0.6
ax.scatter(Hx[nr, 0], Hx[nr, 1], s=40, facecolors='none', edgecolors='white', linewidths=2.0, zorder=6); ax.scatter(Hx[nr, 0], Hx[nr, 1], s=40, facecolors='none', edgecolors='black', linewidths=0.8, zorder=7)
ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(u[0], u[-1]); ax.set_ylim(v[0], v[-1])
ax.set_title('Trp227, ring plane (○ expected H)', fontsize=7.5); ax.text(0.03, 0.97, 'A', transform=ax.transAxes, fontsize=9, fontweight='bold', va='top')
x0 = u[0] + 0.3; y0 = v[0] + 0.3; ax.plot([x0, x0 + 1], [y0, y0], 'k', lw=1.3); ax.text(x0 + 0.5, y0 + 0.08, '1 Å', ha='center', va='bottom', fontsize=6)
cb = fig.colorbar(im, ax=ax, location='right', shrink=0.85, pad=0.02); cb.set_label(r'$\Delta\widetilde{\rho}_{00}$ (r.m.s. units)', fontsize=6.5); cb.ax.tick_params(labelsize=6)
ax = fig.add_subplot(gs[0, 1])
cats = [('amideH', 'amide H positions'), ('aromH', 'aromatic C–H positions'), ('ring_mid', 'ring bond midpoints'), ('pi', '±0.7 Å above/below\nring atoms (pair mean)'),
        ('centre', 'ring centres'), ('cn', 'peptide C–N midpoints'), ('co', 'C=O midpoints')]
rng = np.random.default_rng(1)
for i, (key, lab) in enumerate(cats):
    a = np.array(J[key]); y = np.full(len(a), i) + rng.uniform(-0.25, 0.25, len(a))
    ax.scatter(a, y, s=2.5, color='#d62728' if key in ('amideH', 'aromH') else '#1f4e9c', alpha=0.35, lw=0)
    m = a.mean(); se = CS[key]['se_cluster']; ax.errorbar(m, i, xerr=2 * se, fmt='o', color='k', ms=3, capsize=2, lw=1)
    ax.text(5.0, i, f'{m:+.2f} (n = {len(a)})', fontsize=5.8, va='center')
ax.axvline(0, color='0.5', lw=0.6); ax.set_yticks(range(len(cats))); ax.set_yticklabels([c[1] for c in cats], fontsize=6.2)
ax.invert_yaxis(); ax.set_xlim(-3, 7.6); ax.set_xticks([-2, 0, 2, 4]); ax.set_xlabel(r'$\Delta\widetilde{\rho}_{00}$ (r.m.s. units)')
ax.spines[['top', 'right']].set_visible(False); ax.text(-0.5, 1.0, 'B', transform=ax.transAxes, fontsize=9, fontweight='bold', va='top')
fig.savefig('Fig5_aromatics.pdf', dpi=600); fig.savefig('Fig5_aromatics.png', dpi=300); print('lim', lim)
