# Supplementary Fig. S6 (2 Nov 2026): make_fig6_v3 with the symbol rho_00
# Fig. 6 v3 (28 Oct 2026): error bars in G = 2 x cluster-bootstrap SE (fig6_cluster_se.py: rings/residues resampled as units;
# backbone classes in blocks of 5 residues).
# Fig. 6 v2 (27 Oct 2026): data from fig6_exact_data.py (MTZ + deposited 6MU9 coordinates).
# Changes from v1 (fig6_aromatics.py): expected H positions (not in the model) marked in A-F; panel G label
# "above/below ring atoms (±0.7 Å, pair mean)"; panel G values are exact per-position sums; G title.
# Arrows: matplotlib quiver with automatic scaling, separately for each panel (lengths comparable within a panel only).
import sys, numpy as np, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from common import draw_atoms
tag = sys.argv[1] if len(sys.argv) > 1 else 'deposited'
D = np.load(f'fig6_sections_{tag}.npz', allow_pickle=True); J = json.load(open(f'fig6_stats_{tag}.json')); CS = json.load(open(f'fig6_cluster_se_{tag}.json')); rms = float(D['rms'])
res = [(85, 'Phe85'), (210, 'Tyr210'), (227, 'Trp227')]
fig = plt.figure(figsize=(7.2, 6.0), constrained_layout=True)
gs = fig.add_gridspec(3, 3, width_ratios=[1, 0.75, 1.05]); lim = 2.5
for r, (resi, name) in enumerate(res):
    for c, o in enumerate(('face', 'edge')):
        ax = fig.add_subplot(gs[r, c]); k = f'r{resi}_{o}'
        u, v = D[k + '_u'], D[k + '_v']; U, V = np.meshgrid(u, v)
        dL = D[k + '_dL0'] / rms
        im = ax.pcolormesh(U, V, dL, cmap='RdBu_r', norm=TwoSlopeNorm(0, -lim, lim), shading='auto', rasterized=True)
        L0 = D[k + '_oL0']; ax.contour(U, V, L0, levels=np.linspace(0.25 * L0.max(), L0.max(), 4), colors='0.45', linewidths=0.35)
        ex, ey, n = D[k + '_frame']
        dd = np.stack([D[k + '_dL1' + x] for x in 'xyz'], -1); du, dv = dd @ ex, dd @ ey; s = 6
        ax.quiver(U[::s, ::s], V[::s, ::s], du[::s, ::s], dv[::s, ::s], color='k', width=0.003, scale=None, alpha=0.45, zorder=5)
        names = D[k + '_atomnames']; xyz = D[k + '_atomxyz']; sel = [i for i, nm in enumerate(names) if nm.startswith(f'{resi}:')]
        draw_atoms(ax, names[sel], xyz[sel], near=0.6 if o == 'face' else 0.4, far=1.6 if o == 'face' else 1.0, lw=0.8)
        Hx = D[k + '_Hxyz']
        if o == 'face' and len(Hx):
            nr = np.abs(Hx[:, 2]) < 0.6
            ax.scatter(Hx[nr, 0], Hx[nr, 1], s=46, facecolors='none', edgecolors='white', linewidths=2.2, marker='o', zorder=6)
            ax.scatter(Hx[nr, 0], Hx[nr, 1], s=46, facecolors='none', edgecolors='black', linewidths=0.9, marker='o', zorder=7)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(u[0], u[-1]); ax.set_ylim(v[0], v[-1])
        if r == 0: ax.set_title('ring plane (○ expected H)' if o == 'face' else 'perpendicular plane', fontsize=8)
        if c == 0: ax.set_ylabel(name, fontsize=9)
        ax.text(0.03, 0.96, 'ABCDEF'[2 * r + c], transform=ax.transAxes, fontsize=9, fontweight='bold', va='top')
cb = fig.colorbar(im, ax=fig.axes, location='bottom', shrink=0.45, pad=0.01, aspect=30)
cb.set_label(r'$\Delta\widetilde{\rho}_{00}$ (observed $-$ model), units of r.m.s.', fontsize=7); cb.ax.tick_params(labelsize=6)
ax = fig.add_subplot(gs[:, 2])
cats = [('amideH', 'amide H positions\n(not in model)'), ('aromH', 'aromatic C–H positions\n(not in model)'), ('ring_mid', 'ring bond\nmidpoints'),
        ('pi', 'above/below ring atoms\n(±0.7 Å, pair mean)'), ('centre', 'ring centres'), ('cn', 'peptide C–N\nmidpoints'), ('co', 'C=O\nmidpoints')]
rng = np.random.default_rng(1)
for i, (key, lab) in enumerate(cats):
    a = np.array(J[key]); y = np.full(len(a), i) + rng.uniform(-0.25, 0.25, len(a))
    ax.scatter(a, y, s=3, color='#d62728' if key in ('amideH', 'aromH') else '#1f4e9c', alpha=0.35, lw=0)
    m = a.mean(); se = CS[key]['se_cluster']; ax.errorbar(m, i, xerr=2 * se, fmt='o', color='k', ms=3.5, capsize=2, lw=1)
    ax.text(4.3, i, f'{m:+.2f}\n(n={len(a)})', fontsize=6, va='center')
ax.axvline(0, color='0.5', lw=0.6); ax.set_yticks(range(len(cats))); ax.set_yticklabels([c[1] for c in cats], fontsize=6.5)
ax.invert_yaxis(); ax.set_xlim(-3, 5.6); ax.set_xlabel(r'$\Delta\widetilde{\rho}_{00}$ (r.m.s. units)', fontsize=7.5); ax.tick_params(axis='x', labelsize=6.5)
ax.set_title('all positions in 6MU9', fontsize=8); ax.text(0.03, 0.985, 'G', transform=ax.transAxes, fontsize=9, fontweight='bold', va='top')
fig.savefig('FigS6_aromatics.png', dpi=300); fig.savefig('FigS6_aromatics.pdf'); print('ok')
