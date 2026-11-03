import os,sys
from mtz import read_mtz
import numpy as np
from fractions import Fraction
def parse_ops(strs):
    ops=[]
    for s in strs:
        R=np.zeros((3,3));t=np.zeros(3)
        for i,expr in enumerate(s.replace("'","").split(',')):
            e=expr.replace(' ','').replace('+',' +').replace('-',' -').split()
            for term in e:
                for j,ax in enumerate('xyz'):
                    if ax in term:
                        R[i,j]=-1 if term.startswith('-') else 1; break
                else: t[i]=float(Fraction(term))
        ops.append((R,t))
    return ops
def load(f):
    m=read_mtz(f);d=m['data'];c=m['cols']
    hkl=d[:,:3].astype(int);F=d[:,c.index('F')];P=np.deg2rad(d[:,c.index('PHI')])
    return m['cell'],hkl,F*np.exp(1j*P)
def test_sign(hkl,C,ops):
    D={tuple(h):c for h,c in zip(hkl,C)}
    res={}
    for s in [+1,-1]:
        errs=[]
        for h,c in D.items():
            if abs(c)<1e-3*max(abs(v) for v in list(D.values())[:50]): continue
            for R,t in ops[1:]:
                hr=tuple(int(round(x)) for x in np.array(h)@R)
                if hr in D and hr!=h:
                    pred=c*np.exp(s*2j*np.pi*np.dot(h,t))
                    errs.append(abs(np.angle(D[hr]/pred)))
        res[s]=(len(errs),np.degrees(np.median(errs)) if errs else None)
    return res
def expand(hkl,C,ops,s):
    D={}
    for h,c in zip(hkl,C):
        for R,t in ops:
            hr=tuple(int(round(x)) for x in np.array(h)@R)
            v=c*np.exp(s*2j*np.pi*np.dot(h,t))
            D.setdefault(hr,v)
            D.setdefault(tuple(-x for x in hr),np.conj(v))
    k=np.array(list(D.keys()));v=np.array(list(D.values()));return k,v
def friedel(hkl,C):
    D={}
    for h,c in zip(hkl,C):
        D.setdefault(tuple(h),c); D.setdefault(tuple(-h),np.conj(c))
    k=np.array(list(D.keys()));v=np.array(list(D.values()));return k,v
