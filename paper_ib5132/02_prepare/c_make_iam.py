"""Independent-atom model (IT92 form factors, CIF Uaniso) for urea and Gly-Ala on the fullP1 index sets.
Writes <name>_iam_fullP1.mtz, <name>_iamph_fullP1.mtz (|F_obs|, IAM phase) and <name>_deform_fullP1.mtz (F_obs - F_IAM)."""
import _paths; from _paths import INPUT, step; OUT=step("02_prepare")
import gemmi, numpy as np
from expand import parse_ops

cifs={'urea':str(INPUT/'urea'/'urea.archive.cif'),'glyala':str(INPUT/'gly-ala'/'gly-ala.cif')}
def num(s): return float(s.split('(')[0])
for name,cf in cifs.items():
    b=gemmi.cif.read(cf).sole_block()
    ops=parse_ops([o.strip("'") for o in b.find_loop('_symmetry_equiv_pos_as_xyz')])
    cell=[num(b.find_value('_cell_length_'+x)) for x in 'abc']; A=np.array(cell)
    at=b.find(['_atom_site_label','_atom_site_fract_x','_atom_site_fract_y','_atom_site_fract_z','_atom_site_U_iso_or_equiv'])
    an=b.find(['_atom_site_aniso_label','_atom_site_aniso_U_11','_atom_site_aniso_U_22','_atom_site_aniso_U_33','_atom_site_aniso_U_12','_atom_site_aniso_U_13','_atom_site_aniso_U_23'])
    U={r[0]:[num(r[i]) for i in range(1,7)] for r in an}
    m=gemmi.read_mtz_file(str(OUT/(name+'_fullP1.mtz'))); d=np.array(m.array); H=d[:,:3].astype(int)
    Fo=d[:,3]*np.exp(1j*np.deg2rad(d[:,4]))
    stol=0.5*np.linalg.norm(H/A,axis=1); hs=H/A   # h*a* for orthogonal cell
    F=np.zeros(len(H),complex)
    for r in at:
        lab=r[0]; el=gemmi.Element(''.join(c for c in lab if c.isalpha())[:1] if not lab.startswith('Cl') else 'Cl')
        f0=np.array([el.it92.calculate_sf(s*s) for s in stol])
        x=np.array([num(r[i]) for i in (1,2,3)])
        u=U.get(lab); Uc=np.array([[u[0],u[3],u[4]],[u[3],u[1],u[5]],[u[4],u[5],u[2]]]) if u else np.eye(3)*num(r[4])
        seen=[]
        for R,t in ops:
            xs=(R@x+t)%1
            if any(np.allclose(xs,p,atol=1e-4) for p in seen): continue
            seen.append(xs)
            Rc=R  # orthogonal cells (a=b where R mixes x,y): fractional R == Cartesian R
            Ur=Rc@Uc@Rc.T
            T=np.exp(-2*np.pi**2*np.einsum('ni,ij,nj->n',hs,Ur,hs))
            F+=f0*T*np.exp(2j*np.pi*(H@xs))
    k=(np.abs(Fo)*np.abs(F)).sum()/(np.abs(F)**2).sum(); F*=k
    R1=np.abs(np.abs(Fo)-np.abs(F)).sum()/np.abs(Fo).sum()
    nz=np.abs(Fo)>0   # reflections with F_obs = 0 (negative intensities set to zero) have no defined phase difference
    print(name,"IAM (IT92, Uani) R(F)=%.4f  scale %.3f  mean|dphi| HAR-IAM %.2f deg (%d reflections with F_obs = 0 excluded)"%(R1,k,np.degrees(np.abs(np.angle(Fo[nz]/F[nz]))).mean(),(~nz).sum()))
    for tag,V in [('iam',F),('iamph',np.abs(Fo)*np.exp(1j*np.angle(F))),('deform',Fo-F)]:
        mm=gemmi.Mtz(with_base=True); mm.spacegroup=gemmi.SpaceGroup('P 1'); mm.set_cell_for_all(m.cell)
        mm.add_dataset('x'); mm.add_column('F','F'); mm.add_column('PHI','P')
        mm.set_data(np.column_stack([H,np.abs(V),np.degrees(np.angle(V))]).astype(np.float32)); mm.write_to_file(str(OUT/f'{name}_{tag}_fullP1.mtz'))
