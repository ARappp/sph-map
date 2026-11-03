import numpy as np
COL={'C':'0.35','O':'#d62728','N':'#1f4e9c','H':'0.75','S':'#d4a017'}
def draw_atoms(ax,names,xyz,near=0.5,far=1.6,label=None,bondmax=1.7,lw=1.0,xl=None,yl=None):
    el=['O' if n.startswith('W') else n.split(':')[-1].split('_')[-1][0] for n in names]
    keep=[i for i in range(len(names)) if abs(xyz[i,2])<far]
    for a in keep:
        for b in keep:
            if b<=a: continue
            d=np.linalg.norm(xyz[a]-xyz[b]); hmax=1.25 if 'H' in (el[a],el[b]) else bondmax
            if d<hmax:
                nr=abs(xyz[a,2])<near and abs(xyz[b,2])<near
                ax.plot(xyz[[a,b],0],xyz[[a,b],1],color='0.2' if nr else '0.55',lw=lw if nr else 0.6,ls='-' if nr else (0,(2,1.5)),zorder=3)
    for i in keep:
        nr=abs(xyz[i,2])<near; e=el[i]; s=(36 if e!='H' else 16)*(1 if nr else 0.5)
        ax.scatter(xyz[i,0],xyz[i,1],s=s,c=COL.get(e,'0.5') if nr else 'w',edgecolors='k' if nr else COL.get(e,'0.5'),linewidths=0.5 if nr else 0.8,zorder=4)
        if label and nr and label(names[i]):
            ax.text(xyz[i,0]+0.1,xyz[i,1]+0.08,label(names[i]),fontsize=6,zorder=5,clip_on=True)
