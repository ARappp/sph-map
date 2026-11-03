import os,sys
import numpy as np, struct, itertools
def read_mtz(f):
    b=open(f,'rb').read()
    hdr=struct.unpack('<i',b[4:8])[0]
    h=b[(hdr-1)*4:].decode('latin1')
    recs=[h[i:i+80] for i in range(0,len(h),80)]
    cols=[];cell=None;sym=[];ncol=nref=0
    for r in recs:
        k=r[:4]
        if k=='NCOL': _,ncol,nref,_=r.split()[:4]; ncol=int(ncol); nref=int(nref)
        elif k=='CELL': cell=list(map(float,r.split()[1:7]))
        elif k=='COLU': cols.append(r.split()[1])
        elif k=='SYMM': sym.append(r[5:].strip())
        elif k=='SYMI': syminf=r
        elif k=='END ': break
    data=np.frombuffer(b[80:80+ncol*nref*4],dtype='<f4').reshape(nref,ncol)
    return dict(cols=cols,cell=cell,sym=sym,data=data,syminf=syminf)
