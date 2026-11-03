import _paths; from _paths import INPUT, step
"""Dylan's Gly-Ala archive.fcf (Tonto 26.10.18, 2026-10-18) -> gly-ala_Fphi.mtz, Friedel pairs merged as in
filter_negative_hkl.py (F^2 weighted by 1/sigma(F^2)^2; phase of the canonical member)."""
import numpy as np, gemmi, sys, math
fcf=str(INPUT/"gly-ala"/"gly_L_ala.archive.fcf"); cif=str(INPUT/"gly-ala"/"gly-ala.cif"); out=str(step("02_prepare")/"gly-ala_Fphi.mtz")
tags=[];rows=[];inl=False
for l in open(fcf):
    s=l.strip()
    if s.startswith('_refln_'): tags.append(s.split()[0]); inl=True; continue
    if inl and s and not s.startswith(('_','#','loop_')):
        t=s.split()
        if len(t)==len(tags): rows.append([float(x) for x in t])
a=np.array(rows); ix={t:i for i,t in enumerate(tags)}
def canon(h):
    h,k,l=h
    return ((h,k,l),1) if (h>0 or (h==0 and k>0) or (h==0 and k==0 and l>=0)) else ((-h,-k,-l),-1)
M={}
for r in a:
    hkl=tuple(int(x) for x in r[:3]); key,sg=canon(hkl)
    F=r[ix['_refln_F_meas']]; s=r[ix['_refln_F_sigma']]; I=F*F; sI=max(2*F*s, s*s)
    m=M.setdefault(key,dict(sw=0,swI=0,Ic=[],ph=None,ph_m=None,n=0))
    m['n']+=1
    if sI>0: w=1/sI**2; m['sw']+=w; m['swI']+=w*I
    m['Ic'].append(r[ix['_refln_F_calc']]**2)
    if sg==1: m['ph']=r[ix['_refln_phase_calc']]
    else: m['ph_m']=-r[ix['_refln_phase_calc']]
H=[];F=[];P=[];FC=[];npair=0
for k in sorted(M):
    m=M[k]; I=m['swI']/m['sw']; H.append(k); F.append(math.sqrt(max(I,0))); FC.append(math.sqrt(np.mean(m['Ic'])))
    P.append(m['ph'] if m['ph'] is not None else m['ph_m']); npair+=m['n']==2
    if m['ph'] is not None and m['ph_m'] is not None:
        dd=abs((m['ph']-m['ph_m']+180)%360-180); assert dd<0.01, (k,dd)
b=gemmi.cif.read(cif).sole_block(); num=lambda v: float(v.split('(')[0])
cell=gemmi.UnitCell(*[num(b.find_value('_cell_length_'+x)) for x in 'abc'],*[num(b.find_value('_cell_angle_'+x)) for x in ('alpha','beta','gamma')])
mt=gemmi.Mtz(with_base=False); mt.spacegroup=gemmi.SpaceGroup('P 1'); mt.cell=cell; mt.add_dataset('HAR')
for lab,t in [('H','H'),('K','H'),('L','H'),('F','F'),('PHI','P'),('FC','F')]: mt.add_column(lab,t)
mt.set_data(np.column_stack([np.array(H),F,P,FC]).astype(np.float32)); mt.update_reso(); mt.write_to_file(out)
print(f"{len(a)} rows -> {len(H)} Friedel-merged ({npair} pairs); cell {cell}; d_min {mt.resolution_high():.4f}")
