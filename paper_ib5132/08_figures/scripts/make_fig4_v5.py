# v5 (2 Nov 2026): panels A and B side by side (full width)
# v4 (2 Nov 2026): symbol theta_om renamed Delta theta_obs
# Fig. 4 v3 (31 Oct 2026; single column): (a) observed-model angle vs d_min (r0 = 0.6 A) for Lys299, Ala298, Val302 against the
# distribution over 263 sites; (b) distribution of the angle averaged over the 11 cut-offs. Values and ranks are in the text
# (no legend table). Old panel (a) (inclination to C->O) -> Supplementary Fig. S5 (below). Data: 6mu9_sweep_exact.csv (exact sums).
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size': 7, 'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#555', 'grid.color': '#e5e5e2', 'grid.linewidth': 0.6})
BLUE, GREY, ORANGE, GREEN = '#2a78d6', '#555555', '#eb6834', '#1baf7a'
s = pd.read_csv('6mu9_sweep_exact.csv'); s['dmin'] = s['dmin'].astype(float); q = s[s.r0 == 0.6]
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5), constrained_layout=True)
g = q.groupby('dmin').angle_FWT_vs_FCALL; med, lo, hi = g.median(), g.quantile(0.1), g.quantile(0.9)
ax[0].fill_between(med.index, lo, hi, color='#d9d8d2', lw=0, label='all sites, 10–90 %'); ax[0].plot(med.index, med, color='#777', lw=1.3, label='all sites, median')
for site, col, mk, lab in [('A299LYS', BLUE, 'o', 'Lys299'), ('A298ALA', ORANGE, 's', 'Ala298'), ('A302VAL', GREEN, '^', 'Val302')]:
    z = q[q.site == site]; ax[0].plot(z.dmin, z.angle_FWT_vs_FCALL, color=col, marker=mk, ms=3.2, lw=1.3, label=lab, mec='white', mew=0.4)
ax[0].set_xlabel('$d_\\mathrm{min}$ (Å)'); ax[0].set_ylabel('observed–model angle $\\Delta\\theta_\\mathrm{obs}$ (°)'); ax[0].set_ylim(0, 50); ax[0].grid(axis='y')
ax[0].legend(frameon=False, fontsize=6.3, ncol=2, loc='upper right'); ax[0].text(-0.15, 1.02, 'A', transform=ax[0].transAxes, fontsize=9, fontweight='bold', va='top')
m = q.groupby('site').angle_FWT_vs_FCALL.mean()
ax[1].hist(m.values, bins=np.arange(0, max(m.values) + 2.5, 2.5), color='#9ec5f4', edgecolor='white', lw=0.8)
for site, lab, col, lw, ls in [('A184PRO', 'Pro184', '#888', 1.0, ':'), ('A75LYS', 'Lys75', '#888', 1.0, '--'), ('A299LYS', 'Lys299', BLUE, 1.8, '-')]:
    ax[1].axvline(m[site], color=col, lw=lw, ls=ls, label=lab)
ax[1].legend(frameon=False, fontsize=6.3, loc='upper right', bbox_to_anchor=(0.82, 1.0)); ax[1].set_xlabel('$\\Delta\\theta_\\mathrm{obs}$ averaged over 11 cut-offs (°)'); ax[1].set_ylabel('carbonyl O sites')
ax[1].text(-0.15, 1.02, 'B', transform=ax[1].transAxes, fontsize=9, fontweight='bold', va='top')
fig.savefig('Fig4_6mu9_resolution.pdf', dpi=600); fig.savefig('Fig4_6mu9_resolution.png', dpi=300)
# SI Fig. S5: inclination of <d> at O299 to C->O
fig, a = plt.subplots(figsize=(3.46, 2.6), constrained_layout=True); L = s[s.site == 'A299LYS']
for r0, ls in [(0.3, '-'), (0.6, '--')]:
    z = L[L.r0 == r0]; a.plot(z.dmin, z.angle_CO_FWT, ls, color=BLUE, marker='o', ms=3, lw=1.3); a.plot(z.dmin, z.angle_CO_FCALL, ls, color=GREY, marker='^', ms=3, lw=1.3)
a.set_xlabel('$d_\\mathrm{min}$ (Å)'); a.set_ylabel('angle of $\\langle\\mathbf{d}\\rangle$ at O299 to C→O (°)'); a.grid(axis='y')
H = [Line2D([], [], color=BLUE, marker='o', ms=3.5, lw=0, label='$2mF_o-DF_c$'), Line2D([], [], color=GREY, marker='^', ms=3.5, lw=0, label='model only'),
     Line2D([], [], color='#333', ls='-', lw=1.3, label='$r_0$ = 0.3 Å'), Line2D([], [], color='#333', ls='--', lw=1.3, label='$r_0$ = 0.6 Å')]
a.legend(handles=H, frameon=False, fontsize=6.3, ncol=2, loc='lower right'); fig.savefig('FigS5_O299_inclination.pdf', dpi=600); fig.savefig('FigS5_O299_inclination.png', dpi=200)
print(m.sort_values(ascending=False).head(4).round(1).to_dict())
