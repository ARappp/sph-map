"""P1 symmetry expansion of the observed HAR coefficients (urea: P-4 2_1 m; Gly-Ala: P2_12_12_1).
Writes <name>_fullP1.mtz (one member per Friedel pair) to WORK/02_prepare."""
import _paths; from _paths import INPUT, step; import os; OUT=step("02_prepare")
from expand import *; import gemmi, numpy as np

jobs={'urea':(str(INPUT/'urea'/'urea_Fphi.mtz'),
        ["+x,+y,+z","+y,-x,-z","+x+1/2,-y+1/2,-z","-x,-y,+z","+y+1/2,+x+1/2,+z","-y,+x,-z","-x+1/2,+y+1/2,-z","-y+1/2,-x+1/2,+z"]),
      'glyala':(str(OUT/'gly-ala_Fphi.mtz'),
        ["+x,+y,+z","-x+1/2,-y,+z+1/2","+x+1/2,-y+1/2,-z","-x,+y+1/2,-z+1/2"])}
for name,(f,ops) in jobs.items():
    cell,hkl,C=load(f); k,v=expand(hkl,C,parse_ops(ops),1)
    # keep one member per Friedel pair (sph will complete with p1_complete_friedel)
    keep={}
    for h,c in zip(map(tuple,k),v):
        if tuple(-x for x in h) not in keep: keep[h]=c
    H=np.array(list(keep.keys())); V=np.array(list(keep.values()))
    m=gemmi.Mtz(with_base=True); m.spacegroup=gemmi.SpaceGroup('P 1'); m.set_cell_for_all(gemmi.UnitCell(*cell))
    m.add_dataset('x'); m.add_column('F','F'); m.add_column('PHI','P')
    m.set_data(np.column_stack([H,np.abs(V),np.degrees(np.angle(V))]).astype(np.float32))
    m.write_to_file(str(OUT/(name+'_fullP1.mtz'))); print(name,"input",len(hkl),"-> unique half-sphere",len(H),"(full sphere",len(k),")")
