# v6 (2 Nov 2026): symbol D_AB (bond dipolar index)
# Fig. 2 v5 (30 Oct 2026; renamed bond gradient index; S4 legend moved above the axes): bond gradient index relative to IAM for the 17 analysed bonds (new Gly-Ala HAR data);
# bars = range over alternative sampling windows. The angle panel (old (a)) is Supplementary Fig. S4 (same script, below).
import pandas as pd, numpy as np, matplotlib, json
matplotlib.use('Agg'); import matplotlib.pyplot as plt
WS = json.load(open('window_sensitivity.json'))
plt.rcParams.update({'font.size': 7, 'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#555',
                     'grid.color': '#e5e5e2', 'grid.linewidth': 0.6})
C = {'HAR': '#2a78d6', 'IAMph': '#eb6834', 'IAM': '#1baf7a'}; MK = {'HAR': 'o', 'IAMph': 's', 'IAM': '^'}
LAB = {'HAR': '$F_\\mathrm{obs}$, HAR phases', 'IAMph': '$F_\\mathrm{obs}$, IAM phases', 'IAM': 'IAM $F_\\mathrm{calc}$'}
d = pd.read_csv('smallmol_bonds.csv'); d = d[~d.apply(lambda r: (r.molecule, r.bond) in {('urea', 'C-N1'), ('glyala', 'C1-C2')}, axis=1)]
order = ['C=O', 'C-O', 'C-N', 'N-H', 'C-H']; d['o'] = d['class'].map(order.index); d['m'] = d.molecule.map({'urea': 0, 'glyala': 1})
d = d.sort_values(['m', 'o', 'bond']).reset_index(drop=True); d['label'] = d.bond.str.replace('-', '–'); x = np.arange(len(d)); nU = int((d.molecule == 'urea').sum())
def frame(ax):
    ax.axvline(nU - 0.5, color='#888', lw=0.8)
    for i in range(1, len(d)):
        if d['class'][i] != d['class'][i - 1] and i != nU: ax.axvline(i - 0.5, color='#ddd', lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(d['label'], rotation=90, fontsize=6.5); ax.set_xlim(-0.6, len(d) - 0.4); ax.grid(axis='y')
# main figure: ratio
fig, ax = plt.subplots(figsize=(3.46, 2.7), constrained_layout=True)
for k, col in [('HAR', 'A_HAR'), ('IAMph', 'A_IAMphase')]:
    off = {'HAR': -0.14, 'IAMph': 0.14}[k]; key = {'HAR': 'har_over_iam', 'IAMph': 'iamph_over_iam'}[k]
    ax.plot(x + off, d[col] / d['A_IAM'], MK[k], ms=3.8, color=C[k], mec='white', mew=0.5, ls='none', zorder=3, label=LAB[k])
    for i, r in d.iterrows():
        W = WS[f"{r.molecule} {r.bond}"]; v = [W[w][key] for w in W]; ax.vlines(x[i] + off, min(v), max(v), color=C[k], lw=0.9, alpha=0.55, zorder=2)
ax.axhline(1, color=C['IAM'], lw=1.1, ls='--', label='IAM reference'); frame(ax)
ax.set_ylabel('$D_{AB}$ relative to IAM'); ax.set_ylim(0.6, 1.08)
ax.legend(frameon=False, fontsize=6.3, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0), handletextpad=0.3, columnspacing=0.8)
ax.text((nU - 1) / 2, 1.075, 'urea', ha='center', va='top', fontweight='bold'); ax.text((nU + len(d) - 1) / 2, 1.075, 'Gly–Ala', ha='center', va='top', fontweight='bold')
fig.savefig('Fig2_descriptors.pdf', dpi=600); fig.savefig('Fig2_descriptors.png', dpi=300)
# SI Fig. S4: angle to the bond axis
fig, ax = plt.subplots(figsize=(6.5, 2.4), constrained_layout=True)
for k, col in [('HAR', 'angle_HAR'), ('IAMph', 'angle_IAMphase'), ('IAM', 'angle_IAM')]:
    ax.plot(x + {'HAR': -0.18, 'IAMph': 0, 'IAM': 0.18}[k], d[col], MK[k], ms=4, color=C[k], mec='white', mew=0.5, ls='none', label=LAB[k])
frame(ax); ax.set_ylabel('angle between $\\langle\\mathbf{d}\\rangle_{AB}$\nand the bond axis (°)'); ax.set_ylim(0, 18)
ax.legend(frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=7)
ax.text((nU - 1) / 2, 17.5, 'urea', ha='center', va='top', fontweight='bold'); ax.text((nU + len(d) - 1) / 2, 17.5, 'Gly–Ala', ha='center', va='top', fontweight='bold')
fig.savefig('FigS4_bond_angles.pdf', dpi=600); fig.savefig('FigS4_bond_angles.png', dpi=200); print('ok', len(d))
