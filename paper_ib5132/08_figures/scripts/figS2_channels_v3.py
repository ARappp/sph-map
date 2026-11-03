# v3 (2 Nov 2026): column labels |d|^2 instead of P_1
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, PowerNorm
from common import draw_atoms
S=np.load('sections_SI_small.npz'); P=np.load('sections_SI_6mu9_pep.npz')
rows=[('urea','urea_IAM','Urea',S,None),('glyala','glyala_IAM','Gly–Ala',S,None),('6mu9_FWT_pep299','6mu9_model_pep299','6MU9, Lys299',P,lambda n:n.split(':')[0]=='W' or 297<=int(n.split(':')[0])<=301)]
cols=[('L0',r'$\widetilde{\rho}_{00}$'),('L1x',r'$\widetilde{\rho}_{1x}$ ($m=+1$)'),('L1y',r'$\widetilde{\rho}_{1y}$ ($m=-1$)'),('L1z',r'$\widetilde{\rho}_{1z}$ ($m=0$)'),('P1power',r'$|\mathbf{d}|^2$'),('P1ref',r'$|\mathbf{d}|^2$, IAM / model')]
fig,axs=plt.subplots(3,6,figsize=(7.4,4.3),constrained_layout=True)
for r,(k,kref,name,D,filt) in enumerate(rows):
    u,v=D[k+'_u'],D[k+'_v']; U,V=np.meshgrid(u,v)
    dm=max(abs(D[f'{k}_{c}']).max() for c in ('L1x','L1y','L1z'))
    pm=max(D[k+'_P1power'].max(),D[kref+'_P1power'].max())
    names=D[k+'_atomnames']; xyz=D[k+'_atomxyz']
    if filt is not None:
        sel=[i for i,n in enumerate(names) if filt(n)]; names,xyz=names[sel],xyz[sel]
    for c,(ch,title) in enumerate(cols):
        ax=axs[r,c]
        if ch=='L0':
            a=np.clip(D[k+'_L0'],0,None); im=ax.pcolormesh(U,V,a,cmap='Greys',norm=PowerNorm(0.5,vmin=0,vmax=a.max()),shading='auto',rasterized=True)
        elif ch.startswith('L1'):
            im=ax.pcolormesh(U,V,D[f'{k}_{ch}'],cmap='RdBu_r',norm=TwoSlopeNorm(0,-dm,dm),shading='auto',rasterized=True)
        else:
            a=D[(kref if ch=='P1ref' else k)+'_P1power']; im=ax.pcolormesh(U,V,a,cmap='magma_r',norm=PowerNorm(0.5,vmin=0,vmax=pm),shading='auto',rasterized=True)
        draw_atoms(ax,names,xyz,near=0.45,far=0.9,lw=0.7)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(u[0],u[-1]); ax.set_ylim(v[0],v[-1])
        if r==0: ax.set_title(title,fontsize=7.5)
        if c==0: ax.set_ylabel(name,fontsize=8)
        if c in (3,5):
            cb=fig.colorbar(im,ax=axs[r,c],shrink=0.8,pad=0.02,aspect=15); cb.ax.tick_params(labelsize=5.5)
for i,ax in enumerate(axs.ravel()): ax.text(0.03,0.96,chr(65+i),transform=ax.transAxes,fontsize=7,fontweight='bold',va='top')
fig.savefig('FigS2_channels.png',dpi=300); fig.savefig('FigS2_channels.pdf'); print('ok')
