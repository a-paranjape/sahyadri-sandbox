# usage: python compare_quantization.py   (edit D below; output of the 512^3 run: compare_quantization_512.out)
# Effect of .pkc quantization on Rockstar halos: raw (h5g, unquantized HDF5) vs pkc, with pkc vs pkc2
# (same input, two Rockstar runs) as the noise floor. Halos matched by nearest centre (periodic KD tree).
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
D='/storage/shadab/pkc_rockstar/rsval/'; L=50.; mp=8.110260e7
C=['Mvir','Vmax','Rvir','Rs','Spin','X','Y','Z','VX','VY','VZ']
def ld(r,i):
    p=f'{D}{r}/out_{i}.list'
    with open(p) as fh: h=fh.readline()[1:].split()
    return pd.read_csv(p,comment='#',sep=r'\s+',header=None,names=h,usecols=C)[C].to_numpy()
zs=[3,2,1,0.5,0.25,0]
mb=[(1e10,1e11),(1e11,1e12),(1e12,1e13),(1e13,1e16)]
def pair(a,b,tag,z):
    t=cKDTree(b[:,5:8]%L,boxsize=L); out=[]
    for lo,hi in mb:
        m=(a[:,0]>=lo)&(a[:,0]<hi)
        if m.sum()<5: continue
        d,j=t.query(a[m,5:8]%L); ok=d<0.01*(1+ (lo>=1e12)*4)   # 10 kpc/h, 50 kpc/h above 1e12
        A=a[m][ok]; B=b[j[ok]]
        good=abs(A[:,0]/B[:,0]-1)<0.5        # drop host/sub flips for the scatter stats
        A,B,dd=A[good],B[good],d[ok][good]
        r=lambda k: A[:,k]/B[:,k]-1
        dm,dv,drs,dsp=r(0),r(1),r(3),r(4)
        dvel=np.linalg.norm(A[:,8:11]-B[:,8:11],axis=1)
        na=m.sum(); nb=((b[:,0]>=lo)&(b[:,0]<hi)).sum()
        out.append((tag,z,f'{lo:.0e}',na,nb,f'{na/nb-1:+.4f}',f'{1/np.sqrt(nb):.4f}',f'{ok.mean()*100:.2f}',f'{(~good).mean()*100:.2f}',
          f'{np.median(dm):+.1e}',f'{np.median(abs(dm)):.1e}',f'{np.median(abs(dv)):.1e}',f'{np.median(dd)*1e3:.2f}',
          f'{np.median(dvel):.2f}',f'{np.median(abs(drs)):.1e}',f'{np.median(drs):+.1e}',f'{np.median(abs(dsp)):.1e}',f'{np.median(dsp):+.1e}'))
    return out
rows=[]
for i,z in enumerate(zs):
    pk,p2,h5=ld('pkc',i),ld('pkc2',i),ld('h5g',i)
    rows+=pair(pk,h5,'pkc-raw',z); rows+=pair(pk,p2,'pkc-pkc2',z)
cols=['pair','z','Mlo','N_a','N_b','N_a/N_b-1','1/sqrt(N)','match%','flip%','med dM/M','med|dM/M|','med|dVmax|','med dpos[kpc/h]','med dvel[km/s]','med|dRs/Rs|','med dRs/Rs','med|dSpin/S|','med dSpin/S']
df=pd.DataFrame(rows,columns=cols); pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
print(df.to_string(index=False))
# mass function over all z: total counts per bin
