"""Step 07a: prepare the PDB-REDO coefficients (Supplementary Section 'Five PDB-REDO structures').
(1) FWT/PHWT and FC_ALL/PHIC_ALL expanded to a Friedel-unique P1 half set: <id>_data_P1.mtz, <id>_model_P1.mtz
    (checked against the gemmi FFT map in the original space group).
(2) H-omit coefficients. PDB-REDO FC_ALL contains riding H (the deposited 6MU9 model has none); the amide-H and Phe/Tyr
    ring-H contributions are removed from the model (model_noH = FC_ALL - F_H) and the data map is rebuilt as
    2mFo - DFc(noH) with the H-free phases: <id>_data_noH_P1.mtz, <id>_model_noH_P1.mtz. F_H = gemmi Fcalc of the removed
    H atoms, put on the FC_ALL scale with per-shell scale factors fitted on the heavy atoms. PDB-REDO maps use
    2mFo - DFc for centric reflections too (FWT = FC_ALL + DELFWT for all reflections). Applied to the original model the
    recipe reproduces FWT (R printed). The models are not re-refined.
Output: WORK/07_pdbredo/<id>/  (about 2 min)"""
import _paths, gemmi, numpy as np
from _paths import INPUT, step
IDS = ["4y44", "3x3r", "3ume", "3la9", "4gyb"]
def atoms_named(r,name): return [a for a in r if a.name==name]
def match(cands,alt):
    for a in cands:
        if a.altloc==alt: return a
    for a in cands:
        if a.altloc=='\0': return a
    return cands[0] if cands else None
def build_H(st):
    """structure with only the amide H (N-H 0.86) and Phe/Tyr ring H (C-H 0.93), all altlocs"""
    out=gemmi.Structure(); out.cell=st.cell; out.spacegroup_hm=st.spacegroup_hm; m=gemmi.Model('1'); n=0
    P=lambda a: np.array(a.pos.tolist())
    for ch in st[0]:
        c=gemmi.Chain(ch.name); pol=ch.get_polymer()
        for i,r in enumerate(pol):
            rr=gemmi.Residue(); rr.name=r.name; rr.seqid=r.seqid
            def add(x,par):
                a=gemmi.Atom(); a.name='H'; a.element=gemmi.Element('H'); a.pos=gemmi.Position(*x); a.b_iso=par.b_iso*1.2; a.occ=par.occ; a.altloc=par.altloc; rr.add_atom(a)
            if i and r.name!='PRO':
                for N in atoms_named(r,'N'):
                    CA=match(atoms_named(r,'CA'),N.altloc); C=match(atoms_named(pol[i-1],'C'),N.altloc)
                    if not(CA and C) or C.pos.dist(N.pos)>1.5: continue
                    b=(P(N)-P(C))/C.pos.dist(N.pos)+(P(N)-P(CA))/CA.pos.dist(N.pos); b/=np.linalg.norm(b); add(P(N)+0.86*b,N)
            if r.name in ('PHE','TYR'):
                alts=sorted({a.altloc for a in r if a.name=='CZ'})
                for alt in alts:
                    ring=[match(atoms_named(r,nm),alt) for nm in ('CG','CD1','CE1','CZ','CE2','CD2')]
                    if any(a is None for a in ring): continue
                    cen=np.mean([P(a) for a in ring],axis=0)
                    for a in ring[1:3]+ring[4:]+([ring[3]] if r.name=='PHE' else []):
                        u=P(a)-cen; u/=np.linalg.norm(u); add(P(a)+0.93*u,a)
            if len(rr): c.add_residue(rr); n+=len(rr)
        m.add_chain(c)
    out.add_model(m); return out,n
def fcalc(st,dmin,cell,sg,H):
    dc=gemmi.DensityCalculatorX(); dc.d_min=dmin; dc.rate=1.5; dc.set_refmac_compatible_blur(st[0]); dc.set_grid_cell_and_spacegroup(st); dc.put_model_density_on_grid(st[0])
    sf=gemmi.transform_map_to_f_phi(dc.grid,half_l=True); asu=sf.prepare_asu_data(dmin=dmin-1e-3)
    D={tuple(h):x for h,x in zip(asu.miller_array.tolist(),asu.value_array)}
    v=np.array([D.get(tuple(h),np.nan) for h in H.tolist()])
    q2=np.array([cell.calculate_1_d2(h) for h in H.tolist()]); return v*np.exp(dc.blur*q2/4)
for p in IDS:
    W = step("07_pdbredo") / p; W.mkdir(exist_ok=True)
    pin, pout = str(INPUT / "pdbredo" / p), str(W / p)
    # ---- (1) P1 expansion
    src=gemmi.read_mtz_file(f'{pin}_final.mtz')
    for kind,(fl,pl) in {'data':('FWT','PHWT'),'model':('FC_ALL','PHIC_ALL')}.items():
        m=gemmi.read_mtz_file(f'{pin}_final.mtz')
        ref=m.transform_f_phi_to_map(fl,pl,sample_rate=3)
        m.expand_to_p1()
        a=np.array(m,copy=False); lab=[c.label for c in m.columns]
        H=a[:,:3].astype(int); F=a[:,lab.index(fl)]; P=a[:,lab.index(pl)]
        ok=np.isfinite(F)&np.isfinite(P)&(F>0); H,F,P=H[ok],F[ok],P[ok]
        seen=set(); keep=[]
        for i,h in enumerate(map(tuple,H)):
            if h in seen or (-h[0],-h[1],-h[2]) in seen: continue
            seen.add(h); keep.append(i)
        H,F,P=H[keep],F[keep],P[keep]
        o=gemmi.Mtz(with_base=True); o.spacegroup=gemmi.SpaceGroup('P 1'); o.set_cell_for_all(m.cell)
        o.add_dataset('x'); o.add_column('F','F'); o.add_column('PHI','P')
        o.set_data(np.column_stack([H,F,P]).astype(np.float32)); o.write_to_file(f'{pout}_{kind}_P1.mtz')
        # check: P1 FFT (with Friedel) equals original-SG map
        chk=o.transform_f_phi_to_map('F','PHI',exact_size=list(ref.shape))
        r=np.array(ref,copy=False); c=np.array(chk,copy=False)
        print(p,kind,'refl in SG',src.nreflections,'-> P1 half',len(H),' map corr %.6f'%np.corrcoef(r.ravel(),c.ravel())[0,1],flush=True)
    # ---- (2) H-omit coefficients
    st=gemmi.read_structure(f'{pin}_final.pdb'); st.remove_hydrogens()
    mtz=gemmi.read_mtz_file(f'{pin}_final.mtz'); a=np.array(mtz,copy=False).copy(); lab=[c.label for c in mtz.columns]
    col=lambda f,ph: a[:,lab.index(f)]*np.exp(1j*np.radians(a[:,lab.index(ph)]))
    HK=a[:,:3].astype(int); FC=col('FC_ALL','PHIC_ALL'); FW=col('FWT','PHWT'); DF=col('DELFWT','PHDELWT')
    obs=np.isfinite(a[:,lab.index('FP')]); ops=mtz.spacegroup.operations(); cen=np.array([ops.is_reflection_centric(h) for h in HK.tolist()])
    DFc=np.where(~obs,FW,FW-2*DF+DF)   # = FC_ALL here (FWT = FC_ALL + DELFWT); D is reported only
    dmin=mtz.resolution_high(); d=1/np.sqrt(np.array([mtz.cell.calculate_1_d2(h) for h in HK.tolist()]))
    # D per resolution shell (least squares of DFc on FC_ALL over observed acentric reflections; per-reflection ratio is noisy)
    ed=np.quantile(1/d**2,np.linspace(0,1,21)); Dw=np.ones(len(d)); Dsh=[]
    for lo,hi in zip(ed[:-1],ed[1:]):
        s=(1/d**2>=lo)&(1/d**2<=hi); t=s&obs&~cen
        Dsh.append(float(np.clip(((DFc[t]*np.conj(FC[t])).sum()/(np.abs(FC[t])**2).sum()).real,0,1))); Dw[s]=Dsh[-1]
    hs,nH=build_H(st)
    Fheavy=fcalc(st,dmin,mtz.cell,mtz.spacegroup,HK); FH=fcalc(hs,dmin,mtz.cell,mtz.spacegroup,HK)
    # per-shell scale of gemmi Fcalc onto FC_ALL (fit on d<=4 A shells, heavy atoms only)
    edges=np.quantile(1/d**2,np.linspace(0,1,11)); k=np.ones(len(d),complex); ks=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        s=(1/d**2>=lo)&(1/d**2<=hi); ks.append(((FC[s]*np.conj(Fheavy[s])).sum()/(np.abs(Fheavy[s])**2).sum()).real); k[s]=ks[-1]
    klow=np.median([x for x,e in zip(ks,edges[1:]) if 1/np.sqrt(e)<=4]); k[d>4]=klow
    FHs=k*FH
    model=FC-FHs
    # H-omit data map: mFo amplitudes from FWT and FC_ALL, re-phased with the H-free model (as for the deposited 6MU9, whose model had no H)
    mFo_amp=np.abs(FW+FC)/2; ph=np.exp(1j*np.angle(model))   # PDB-REDO: FWT = 2mFo - FC_ALL for ALL reflections (centric included), FC_ALL = DFc
    data=np.where(~obs,model,2*mFo_amp*ph-model)
    # sanity: same recipe with the original (H) model must reproduce REFMAC FWT
    ph0=np.exp(1j*np.angle(FC)); rep=np.where(~obs,FC,2*mFo_amp*ph0-FC)
    repR=np.abs(rep-FW).sum()/np.abs(FW).sum()
    print(f'{p}: removed {nH} H atoms; shell k {np.round(ks,3).tolist()}; D by shell {np.round(Dsh,2).tolist()}; '
          f'|FcH|/|FC_ALL| {np.abs(FHs).sum()/np.abs(FC).sum():.3f}; FWT reproduced to R={repR:.2e}',flush=True)
    for kind,V in (('data',data),('model',model)):
        m=gemmi.read_mtz_file(f'{pin}_final.mtz'); arr=np.array(m,copy=False).copy()
        arr[:,lab.index('FWT')]=np.abs(V); arr[:,lab.index('PHWT')]=np.degrees(np.angle(V)); m.set_data(arr)
        ref=m.transform_f_phi_to_map('FWT','PHWT',sample_rate=3)
        m.expand_to_p1(); b=np.array(m,copy=False); Hp=b[:,:3].astype(int); F=b[:,lab.index('FWT')]; P=b[:,lab.index('PHWT')]
        ok=np.isfinite(F)&np.isfinite(P)&(F>0); Hp,F,P=Hp[ok],F[ok],P[ok]; seen=set(); keep=[]
        for i,h in enumerate(map(tuple,Hp)):
            if h in seen or (-h[0],-h[1],-h[2]) in seen: continue
            seen.add(h); keep.append(i)
        o=gemmi.Mtz(with_base=True); o.spacegroup=gemmi.SpaceGroup('P 1'); o.set_cell_for_all(m.cell); o.add_dataset('x'); o.add_column('F','F'); o.add_column('PHI','P')
        o.set_data(np.column_stack([Hp[keep],F[keep],P[keep]]).astype(np.float32)); o.write_to_file(f'{pout}_{kind}_noH_P1.mtz')
        chk=o.transform_f_phi_to_map('F','PHI',exact_size=list(ref.shape))
        print(f'   {kind}: P1 check corr {np.corrcoef(np.array(ref,copy=False).ravel(),np.array(chk,copy=False).ravel())[0,1]:.6f}',flush=True)
