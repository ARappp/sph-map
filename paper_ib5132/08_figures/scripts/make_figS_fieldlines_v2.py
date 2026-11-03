# SI figure (29 Oct 2026): field lines of the observed and IAM maps for urea and Gly-Ala (formerly main-text Figs 1A,B and 2A,B).
# Same data and drawing as make_fig1_v2.py / make_fig2_v2.py; common logarithmic colour scale and common rho0 contour levels.
import numpy as np, matplotlib
import matplotlib.patheffects as pe
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
Du = np.load('urea_plane_section.npz'); Dg = np.load('glyala_peptide_plane_section.npz')
LEV = [0.2, 0.5, 1, 2, 5, 10]; NORM = LogNorm(0.1, 10)
COL = {'C': '0.35', 'O': '#d62728', 'N': '#1f4e9c', 'H': '0.75'}
fig, axs = plt.subplots(2, 2, figsize=(6.8, 6.4), constrained_layout=True)
# urea
u, v = Du['u'], Du['v']
atU = {k[5:]: Du[k] for k in Du.files if k.startswith('atom_')}
bondsU = [('C', 'O'), ('C', 'N1'), ('C', 'N2'), ('N1', 'H11'), ('N1', 'H12'), ('N2', 'H21'), ('N2', 'H22')]
for ax, tag, title in zip(axs[0], ('urea', 'urea_IAM'), ('urea, observed ($F_{\\rm obs}$, HAR phases)', 'urea, spherical atoms (IAM)')):
    L0 = Du[f'{tag}_L0']; du, dv = Du[f'{tag}_du'], Du[f'{tag}_dv']
    ax.contour(u, v, np.clip(L0, 1e-3, None), levels=LEV, colors='0.72', linewidths=0.45)
    sp = ax.streamplot(u, v, du, dv, color=np.hypot(du, dv), cmap='viridis', norm=NORM, density=1.0, linewidth=0.7, arrowsize=0.6)
    for a, b in bondsU:
        p, q = atU[a], atU[b]; ax.plot([p[0], q[0]], [p[1], q[1]], color='0.25', lw=1.1, zorder=3)
    for k, p in atU.items():
        ax.scatter(*p, s=55 if k[0] != 'H' else 22, c=COL[k[0]], edgecolors='k', linewidths=0.5, zorder=4)
        ax.text(p[0] + 0.12, p[1] + 0.1, k, fontsize=6.5, zorder=5, path_effects=[pe.withStroke(linewidth=2, foreground='w')])
    ax.set_title(title, fontsize=8); ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.0, 2.6)
# Gly-Ala
ug, vg = Dg['u'], Dg['v']
atG = {k[5:]: Dg[k] for k in Dg.files if k.startswith('atom_')}
bondsG = [('C4', 'O3'), ('C4', 'N1'), ('C4', 'C5'), ('N1', 'C2'), ('N1', 'H1N1'), ('C5', 'N2'), ('C5', 'H5A'), ('C5', 'H5B'), ('N2', 'H1N2'), ('N2', 'H2N2'), ('N2', 'H3N2'),
          ('C2', 'C1'), ('C2', 'C3'), ('C2', 'H2'), ('C1', 'O1'), ('C1', 'O2'), ('C3', 'H3A'), ('C3', 'H3B'), ('C3', 'H3C')]
for ax, tag, title in zip(axs[1], ('glyala', 'glyala_IAM'), ('Gly–Ala, observed ($F_{\\rm obs}$, HAR phases)', 'Gly–Ala, spherical atoms (IAM)')):
    L0 = Dg[f'{tag}_L0']; du, dv = Dg[f'{tag}_du'], Dg[f'{tag}_dv']; mag = np.hypot(du, dv)
    ax.contour(ug, vg, np.clip(L0, 1e-3, None), levels=LEV, colors='0.72', linewidths=0.45)
    keep = mag > 0.2
    sp2 = ax.streamplot(ug, vg, np.where(keep, du, np.nan), np.where(keep, dv, np.nan), color=mag, cmap='viridis', norm=NORM, density=1.4, linewidth=0.7, arrowsize=0.6)
    for a, b in bondsG:
        p, q = atG[a], atG[b]; near = abs(p[2]) < 0.45 and abs(q[2]) < 0.45
        if max(abs(p[2]), abs(q[2])) > 1.6: continue
        ax.plot([p[0], q[0]], [p[1], q[1]], color='0.25' if near else '0.6', lw=1.1 if near else 0.7, ls='-' if near else (0, (2, 1.5)), zorder=3)
    for k, p in atG.items():
        if abs(p[2]) > 1.6: continue
        e = k[0]; near = abs(p[2]) < 0.45; s = (55 if e != 'H' else 22) * (1 if near else 0.5)
        ax.scatter(p[0], p[1], s=s, c=COL[e] if near else 'w', edgecolors=COL[e] if not near else 'k', linewidths=0.5 if near else 0.9, zorder=4)
        if near and e != 'H' and ug[0] < p[0] < ug[-1] - 0.3: ax.text(p[0] + 0.12, p[1] + 0.1, k, fontsize=6.5, zorder=5, path_effects=[pe.withStroke(linewidth=2, foreground='w')])
    ax.set_title(title, fontsize=8); ax.set_xlim(ug[0], ug[-1]); ax.set_ylim(vg[0], vg[-1])
for ax, l in zip(axs.ravel(), 'ABCD'):
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.text(0.02, 0.97, l, transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')
cb = fig.colorbar(sp.lines, ax=axs, location='right', shrink=0.6, pad=0.01, extend='min'); cb.set_ticks([0.1, 0.3, 1, 3, 10]); cb.set_ticklabels(['0.1', '0.3', '1', '3', '10']); cb.ax.minorticks_off()
cb.set_label(r'$|\mathbf{d}|$ (in plane)', fontsize=8); cb.ax.tick_params(labelsize=7)
fig.savefig('FigS_fieldlines.pdf'); fig.savefig('FigS_fieldlines.png', dpi=300); print('ok')
